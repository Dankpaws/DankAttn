"""Single-port multi-model router for DankAttn.

Exposes one OpenAI + Anthropic-compatible HTTP port (default :8340) over multiple DankAttn
model servers (`hy4`, `ernie55`, `inkling2`). Because 80+ GB packs are RAM-exclusive on a
96 GB Mac Studio, the router keeps at most one backend process resident at a time: when a
request arrives for a different `model`, it drains and stops the active backend, starts the
requested backend (waking fast via MTLBinaryArchive + SSD prefix cache), and proxies the
request transparently (both streaming SSE and non-streaming).

Usage:
  python3 server/router.py --port 8340 --config router.json
  # or auto-discover packs under ~/dankattn/models:
  python3 server/router.py --port 8340 --models-dir ~/dankattn/models
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from engine_frontend import add_security_args, install_security_middleware, json_body, payload_too_large, PayloadTooLarge


DEFAULT_SPECS = [
    {
        "id": "hy4-flash",
        "aliases": ["hy4", "hy4-v4"],
        "server": os.path.join(_HERE, "hy4_server.py"),
        "engine": os.path.join(_ROOT, "build", "dankattn-hy4"),
        "pack_candidates": ["pack-dfly2b6", "pack-v4-h128-w6"],
        "port": 18351,
        "cap": 262144,
    },
    {
        "id": "ernie-5.5-flash",
        "aliases": ["ernie55", "ernie"],
        "server": os.path.join(_HERE, "ernie55_server.py"),
        "engine": os.path.join(_ROOT, "build", "dankattn-ernie55"),
        "pack_candidates": ["pack-ernie55-w2-g32-w8attn"],
        "port": 18352,
        "cap": 262144,
    },
    {
        "id": "inkling-2-mini",
        "aliases": ["inkling2", "inkling"],
        "server": os.path.join(_HERE, "inkling2_server.py"),
        "engine": os.path.join(_ROOT, "build", "dankattn-inkling2"),
        "pack_candidates": ["pack-inkling2-mini-w8"],
        "port": 18353,
        "cap": 131072,
    },
]


class ModelRouter:
    def __init__(self, models: list[dict], python_bin: str, idle_unload_s: float = 0.0):
        self.models: dict[str, dict] = {}
        self.alias_map: dict[str, str] = {}
        for m in models:
            self.models[m["id"]] = m
            self.alias_map[m["id"]] = m["id"]
            for a in m.get("aliases", ()):
                self.alias_map[a] = m["id"]
        self.python_bin = python_bin
        self.idle_unload_s = idle_unload_s
        self.active_id: str | None = None
        self.proc: subprocess.Popen | None = None
        self.inflight = 0
        self.switch_lock = asyncio.Lock()

    def resolve(self, model_name: str | None) -> dict | None:
        if not model_name:
            if self.active_id:
                return self.models[self.active_id]
            return next(iter(self.models.values()), None)
        # Support Inkling2 :4x-fast / :4x-select suffixes
        base = model_name.split(":", 1)[0]
        mid = self.alias_map.get(model_name) or self.alias_map.get(base)
        return self.models.get(mid) if mid else None

    def _stop_active_sync(self):
        if self.proc is None:
            self.active_id = None
            return
        try:
            self.proc.send_signal(signal.SIGTERM)
            self.proc.wait(timeout=10)
        except Exception:
            try:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except Exception:
                pass
        self.proc = None
        self.active_id = None

    def _wait_ready_sync(self, port: int, timeout_s: float = 90.0) -> bool:
        deadline = time.monotonic() + timeout_s
        url = f"http://127.0.0.1:{port}/health"
        while time.monotonic() < deadline:
            if self.proc and self.proc.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(url, timeout=2.0) as r:
                    data = json.loads(r.read().decode("utf-8"))
                    if data.get("state") == "up":
                        return True
            except Exception:
                pass
            time.sleep(0.5)
        return False

    async def ensure_model(self, spec: dict) -> int:
        async with self.switch_lock:
            if self.active_id == spec["id"] and self.proc and self.proc.poll() is None:
                return spec["port"]
            while self.inflight > 0:
                await asyncio.sleep(0.2)
            await asyncio.to_thread(self._stop_active_sync)
            cmd = [
                self.python_bin, spec["server"],
                "--pack", spec["pack"],
                "--engine", spec["engine"],
                "--port", str(spec["port"]),
                "--cap", str(spec.get("cap", 131072)),
                "--model-id", spec["id"],
            ]
            if self.idle_unload_s > 0:
                cmd += ["--idle-unload", str(self.idle_unload_s)]
            self.proc = subprocess.Popen(cmd, env=os.environ.copy())
            self.active_id = spec["id"]
            ok = await asyncio.to_thread(self._wait_ready_sync, spec["port"])
            if not ok:
                await asyncio.to_thread(self._stop_active_sync)
                raise RuntimeError(f"backend {spec['id']} failed to start")
            return spec["port"]


def discover_models(models_dir: str, config_path: str | None) -> list[dict]:
    if config_path:
        with open(config_path) as f:
            return json.load(f)
    out = []
    for s in DEFAULT_SPECS:
        if not os.path.exists(s["engine"]):
            continue
        pack = None
        for cand in s["pack_candidates"]:
            p = os.path.join(models_dir, cand)
            if os.path.isdir(p):
                pack = p
                break
        if pack:
            out.append({**s, "pack": pack})
    return out


def build_app(router: ModelRouter, allow_hosts=(), api_key="", max_body_mb=128) -> FastAPI:
    app = FastAPI()
    install_security_middleware(app, allow_hosts=allow_hosts, api_key=api_key, max_body_mb=max_body_mb)

    @app.get("/health")
    def health():
        return {"state": "up", "active_model": router.active_id, "models": list(router.models.keys())}

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {"id": m["id"], "object": "model", "owned_by": "DankAttn", "max_model_len": m.get("cap", 131072)}
                for m in router.models.values()
            ],
        }

    async def _proxy(request: Request, path: str):
        try:
            body = await json_body(request)
        except PayloadTooLarge as e:
            return payload_too_large(str(e))
        if not isinstance(body, dict):
            return JSONResponse({"error": {"message": "JSON object required", "type": "invalid_request_error"}}, status_code=400)
        spec = router.resolve(body.get("model"))
        if spec is None:
            return JSONResponse(
                {"error": {"message": f"unknown model {body.get('model')!r}; available: {list(router.models.keys())}",
                           "type": "invalid_request_error"}},
                status_code=404,
            )
        try:
            port = await router.ensure_model(spec)
        except Exception as e:
            return JSONResponse({"error": {"message": str(e), "type": "server_error"}}, status_code=503)

        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}", data=raw,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        router.inflight += 1
        if body.get("stream"):
            try:
                resp = await asyncio.to_thread(urllib.request.urlopen, req, 600.0)
            except urllib.error.HTTPError as e:
                router.inflight -= 1
                return Response(content=e.read(), status_code=e.code, media_type="application/json")
            except Exception as e:
                router.inflight -= 1
                return JSONResponse({"error": {"message": str(e), "type": "server_error"}}, status_code=502)

            async def _stream():
                try:
                    while True:
                        line = await asyncio.to_thread(resp.readline)
                        if not line:
                            break
                        yield line
                finally:
                    resp.close()
                    router.inflight -= 1

            return StreamingResponse(_stream(), media_type="text/event-stream")

        try:
            def _do():
                with urllib.request.urlopen(req, 600.0) as r:
                    return r.status, r.read()
            status, payload = await asyncio.to_thread(_do)
            return Response(content=payload, status_code=status, media_type="application/json")
        except urllib.error.HTTPError as e:
            return Response(content=e.read(), status_code=e.code, media_type="application/json")
        except Exception as e:
            return JSONResponse({"error": {"message": str(e), "type": "server_error"}}, status_code=502)
        finally:
            router.inflight -= 1

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await _proxy(request, "/v1/chat/completions")

    @app.post("/v1/messages")
    async def messages(request: Request):
        return await _proxy(request, "/v1/messages")

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        return await _proxy(request, "/v1/messages/count_tokens")

    return app


def main():
    ap = argparse.ArgumentParser(description="Single-port multi-model DankAttn router")
    ap.add_argument("--port", type=int, default=8340)
    ap.add_argument("--models-dir", default=os.path.join(_ROOT, "models"))
    ap.add_argument("--config", default=None, help="optional JSON list of model backend specs")
    ap.add_argument("--idle-unload", type=float, default=0.0)
    add_security_args(ap)
    args = ap.parse_args()

    models = discover_models(os.path.expanduser(args.models_dir), args.config)
    router = ModelRouter(models, sys.executable, idle_unload_s=args.idle_unload)
    app = build_app(router, allow_hosts=args.allow_host, api_key=args.api_key, max_body_mb=args.max_body_mb)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
