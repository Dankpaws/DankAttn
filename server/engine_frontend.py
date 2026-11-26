"""Shared engine supervision, binary framing, prefix splicing, and OpenAI HTTP helpers for DankAttn front ends."""
import asyncio, hmac, itertools, json, os, re, struct, subprocess, sys, threading, time, urllib.parse, uuid

import jinja2
import numpy as np
from jinja2.sandbox import ImmutableSandboxedEnvironment
from fastapi import Request
from fastapi.responses import JSONResponse


class Unavailable(Exception):
    pass


class BadRequest(Exception):
    pass


class ClientGone(Exception):
    pass


def make_jinja_env():
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, **k: json.dumps(
        x, indent=indent, ensure_ascii=ensure_ascii
    )
    def _raise(msg):
        raise jinja2.exceptions.TemplateError(msg)
    env.globals["raise_exception"] = _raise
    return env


def normalize_messages(body, keep_vision=False):
    msgs = []
    for m in body["messages"]:
        if not isinstance(m, dict):
            raise BadRequest("each message must be an object")
        m = dict(m)
        if isinstance(m.get("content"), list):
            if not all(isinstance(p, dict) for p in m["content"]):
                raise BadRequest("content parts must be objects")
            if keep_vision and any(
                p.get("type") in ("image_url", "image", "video_url", "video")
                or "image_url" in p or "image" in p or "video_url" in p or "video" in p
                for p in m["content"]
            ):
                m["content"] = list(m["content"])
            else:
                m["content"] = "".join(p.get("text", "") for p in m["content"] if p.get("type") == "text")
        tcs = m.get("tool_calls") or []
        if not isinstance(tcs, list) or not all(isinstance(tc, dict) and isinstance(tc.get("function", {}), dict) for tc in tcs):
            raise BadRequest("tool_calls must be a list of {function: {...}} objects")
        for tc in tcs:
            f = tc.get("function", {})
            if isinstance(f.get("arguments"), str):
                try:
                    f["arguments"] = json.loads(f["arguments"])
                except Exception:
                    f["arguments"] = {}
        msgs.append(m)
    return msgs


def read_exact(f, n):
    b = f.read(n)
    if len(b) < n:
        raise EOFError
    return b


class EngineProc:
    """Supervises a DankAttn engine subprocess and multiplexes its OPUS/OPCX binary stream."""

    CRASH_WINDOW_S = 300
    CRASH_MAX = 3

    def __init__(self, cmd, env=None, default_stops=(), think_end=0, think_force=(), idle_unload_s=0):
        self.cmd = list(cmd)
        self.env = env
        self.default_stops = list(default_stops)
        self.think_end = think_end
        self.think_force = list(think_force)
        self.idle_unload_s = max(0.0, float(idle_unload_s or 0))
        self.write_lock = threading.Lock()
        self.streams_lock = threading.Lock()
        self.load_lock = threading.Lock()
        self.state_cv = threading.Condition()
        self._sleep_req = False
        self._wake_req = False
        self.last_active = time.monotonic()
        self.streams = {}
        self.rids = itertools.count(1)
        self.engine = None
        self.state = {"state": "starting", "restarts": 0, "pid": 0, "since": time.time(), "idle_unload_s": self.idle_unload_s}
        self.load = {"instance": uuid.uuid4().hex, "running": 0, "waiting": 0, "decode": 0, "prefill": 0, "prefill_us": 0}
        threading.Thread(target=self._supervise, daemon=True).start()
        if self.idle_unload_s > 0:
            threading.Thread(target=self._idle_watchdog, daemon=True).start()

    def metered(self, cb):
        """Wrap a request callback to track waiting, running, decode, and prefill counters."""
        first = [True]
        with self.load_lock:
            self.load["waiting"] += 1

        def f(item):
            if isinstance(item, tuple):
                cb(item)
                return
            with self.load_lock:
                if isinstance(item, int):
                    if first[0]:
                        self.load["waiting"] -= 1
                        self.load["running"] += 1
                        first[0] = False
                    else:
                        self.load["decode"] += 1
                else:
                    if first[0]:
                        self.load["waiting"] -= 1
                        first[0] = False
                    else:
                        self.load["running"] -= 1
                    if "prompt" in item:
                        self.load["prefill"] += item["prompt"] - item["reused"]
                        self.load["prefill_us"] += int(item["prefill_ms"] * 1000)
            cb(item)

        return f

    def _demux(self, proc):
        out = proc.stdout
        try:
            while True:
                rid, w = struct.unpack("<II", read_exact(out, 8))
                if w == 0xFFFFFFFD:
                    with self.state_cv:
                        self.state.update(state="up", since=time.time())
                        self.last_active = time.monotonic()
                        self.state_cv.notify_all()
                    print(f"dankattn: engine pid {proc.pid} ready", file=sys.stderr, flush=True)
                    continue
                final = w >= 0xFFFFFFFE
                if w == 0xFFFFFFFF:
                    s = struct.unpack("<6I2f", read_exact(out, 32))
                    item = dict(
                        reused=s[0], prompt=s[1], generated=s[2], rounds=s[3], accepted=s[4],
                        length=bool(s[5] & 1), prefill_ms=s[6], wall_ms=s[7],
                    )
                elif w == 0xFFFFFFFE:
                    (n,) = struct.unpack("<I", read_exact(out, 4))
                    item = {"error": read_exact(out, n).decode("utf-8", "replace")}
                elif w == 0xFFFFFFFC:
                    conf, cflags = struct.unpack("<fI", read_exact(out, 8))
                    item = ("cand_sep", conf, cflags)
                else:
                    item = w
                with self.streams_lock:
                    cb = self.streams.pop(rid, None) if final else self.streams.get(rid)
                if final:
                    self.last_active = time.monotonic()
                if cb:
                    cb(item)
        except (EOFError, struct.error, OSError, ValueError):
            pass
        with self.state_cv:
            if not self._sleep_req:
                self.state["state"] = "down"
            self.state_cv.notify_all()
        with self.streams_lock:
            dead = list(self.streams.values())
            self.streams.clear()
        for cb in dead:
            cb({"error": "engine exited"})

    def _idle_watchdog(self):
        poll = min(5.0, max(0.5, self.idle_unload_s / 4.0))
        while True:
            time.sleep(poll)
            proc_to_close = None
            with self.state_cv:
                if self.state["state"] != "up" or self._sleep_req:
                    continue
                if time.monotonic() - self.last_active < self.idle_unload_s:
                    continue
                with self.streams_lock, self.load_lock:
                    if self.streams or self.load["waiting"] > 0 or self.load["running"] > 0:
                        continue
                self._sleep_req = True
                self.state.update(state="sleeping", since=time.time())
                self.state_cv.notify_all()
                proc_to_close = self.engine
            if proc_to_close is not None:
                try:
                    with self.write_lock:
                        proc_to_close.stdin.close()
                except (OSError, ValueError):
                    pass

    def ensure_awake(self, timeout=60.0):
        deadline = time.monotonic() + timeout
        with self.state_cv:
            self.last_active = time.monotonic()
            if self.state["state"] == "up" and not self._sleep_req:
                return
            if self.state["state"] == "sleeping" or self._sleep_req:
                self._wake_req = True
                self.state_cv.notify_all()
            while self.state["state"] != "up" or self._sleep_req:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    raise Unavailable(self.state["state"])
                if self.state["state"] == "down" and not self._wake_req:
                    raise Unavailable("down")
                self.state_cv.wait(timeout=min(rem, 1.0))
            self.last_active = time.monotonic()

    def _supervise(self):
        exits = []
        while True:
            with self.state_cv:
                self.state.update(state="starting", since=time.time())
                self.state_cv.notify_all()
            self.engine = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=self.env)
            self.state["pid"] = self.engine.pid
            self._demux(self.engine)
            code = self.engine.wait()
            with self.state_cv:
                if self._sleep_req:
                    self._sleep_req = False
                    self.state.update(state="sleeping", pid=0, since=time.time())
                    self.state_cv.notify_all()
                    print(
                        f"dankattn: engine sleeping after {self.idle_unload_s:.0f}s idle (RAM released, exit={code})",
                        file=sys.stderr, flush=True,
                    )
                    while not self._wake_req:
                        self.state_cv.wait()
                    self._wake_req = False
                    print("dankattn: waking engine...", file=sys.stderr, flush=True)
                    continue
            now = time.time()
            exits = [t for t in exits if now - t < self.CRASH_WINDOW_S] + [now]
            print(
                f"dankattn: engine pid {self.engine.pid} exited with {code} ({len(exits)} in {self.CRASH_WINDOW_S} s)",
                file=sys.stderr, flush=True,
            )
            if len(exits) >= self.CRASH_MAX:
                print("dankattn: engine keeps exiting; stopping the front end", file=sys.stderr, flush=True)
                os._exit(1)
            self.state["restarts"] += 1
            time.sleep(2)

    def submit(self, ids, max_new, spec, temp, top_p, top_k, seed, cb,
               group=None, think_budget=0, strict=False, min_p=None, extra_stops=(), vision_imgs=None, fpr_suffixes=None):
        self.ensure_awake()
        rid = next(self.rids)
        stops = list(self.default_stops)
        for t in extra_stops:
            if t not in stops:
                stops.append(t)
        req = struct.pack("<III", 0x5355504F, rid, len(ids)) + struct.pack(f"<{len(ids)}I", *ids)
        flags = (
            (1 if spec else 0)
            | (2 if group else 0)
            | (4 if think_budget else 0)
            | (8 if strict else 0)
            | (16 if min_p is not None else 0)
            | (32 if vision_imgs else 0)
        )
        req += struct.pack("<III", max_new, flags, len(stops)) + struct.pack(f"<{len(stops)}I", *stops)
        req += struct.pack("<ffIQ", temp, top_p, top_k, seed)
        if group:
            req += struct.pack("<III", *group)
            if (group[2] & 0xF) == 2 and fpr_suffixes is not None:
                for suf in fpr_suffixes:
                    req += struct.pack("<I", len(suf)) + (struct.pack(f"<{len(suf)}I", *suf) if suf else b"")
        if think_budget:
            req += struct.pack("<III", think_budget, self.think_end, len(self.think_force))
            req += struct.pack(f"<{len(self.think_force)}I", *self.think_force)
        if min_p is not None:
            req += struct.pack("<f", min_p)
        if vision_imgs:
            parts = [req, struct.pack("<I", len(vision_imgs))]
            for im in vision_imgs:
                parts.append(struct.pack("<IIIIQ", im["T"], im["h"], im["w"], im["tok_off"], im["hash_u64"]))
                parts.append(im["pixels_bf16"])
            req = b"".join(parts)
        with self.streams_lock:
            self.streams[rid] = cb
        try:
            with self.write_lock:
                self.engine.stdin.write(req)
                self.engine.stdin.flush()
        except (OSError, ValueError):
            with self.streams_lock:
                self.streams.pop(rid, None)
            raise Unavailable("engine exited")
        return rid

    def cancel(self, rid):
        with self.streams_lock:
            if rid not in self.streams:
                return
        try:
            with self.write_lock:
                self.engine.stdin.write(struct.pack("<II", 0x5843504F, rid))
                self.engine.stdin.flush()
        except (OSError, ValueError):
            pass


class Splitter:
    """Incremental detokeniser that routes text inside thinking tags to reasoning."""

    CTX = 6

    def __init__(self, tok, thinking, think_close, strip_tokens=()):
        self.tok = tok
        self.think_close = think_close
        self.strip_tokens = strip_tokens
        self.ids, self.think = [], thinking
        self.prefix, self.read = 0, 0
        self.reasoning, self.content = "", ""
        self.n_reasoning = 0

    def push(self, t):
        self.ids.append(t)
        self.n_reasoning += self.think
        before = self.tok.decode(self.ids[self.prefix:self.read], skip_special_tokens=False)
        text = self.tok.decode(self.ids[self.prefix:], skip_special_tokens=False)
        if text.endswith("\ufffd") or len(text) <= len(before):
            return "", ""
        new = text[len(before):]
        self.read = len(self.ids)
        self.prefix = max(0, self.read - self.CTX)
        r = c = ""
        while new:
            if self.think:
                i = new.find(self.think_close)
                if i < 0:
                    r += new
                    break
                r += new[:i]
                new = new[i + len(self.think_close):]
                self.think = False
            else:
                c += new
                break
        for s in self.strip_tokens:
            c = c.replace(s, "")
        self.reasoning += r
        self.content += c
        return r, c


class PrefixSplicer:
    """Reuses recent request token ids as far as decoded text matches, avoiding BPE boundary drift."""

    def __init__(self, tok, max_held=16):
        self.tok = tok
        self.max_held = max_held
        self.held = []
        self.lock = threading.Lock()

    @staticmethod
    def _lcp(a, b):
        n = min(len(a), len(b))
        if n == 0:
            return 0
        d = np.flatnonzero(a[:n] != b[:n])
        return int(d[0]) if len(d) else n

    def splice(self, prompt):
        ids = self.tok.encode(prompt, add_special_tokens=False).ids
        with self.lock:
            cands = list(self.held)
        if not cands:
            return ids
        a = np.asarray(ids, dtype=np.uint32)
        lcp, held = max(((self._lcp(h, a), h) for h in cands), key=lambda x: x[0])
        if lcp >= len(held) - 1:
            return ids
        lo, hi = lcp, len(held)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if prompt.startswith(self.tok.decode(held[:mid].tolist(), skip_special_tokens=False)):
                lo = mid
            else:
                hi = mid - 1
        if lo <= lcp + 8:
            return ids
        text = self.tok.decode(held[:lo].tolist(), skip_special_tokens=False)
        if text.endswith("\ufffd") or not prompt.startswith(text):
            return ids
        return held[:lo].tolist() + self.tok.encode(prompt[len(text):], add_special_tokens=False).ids

    def hold(self, ids):
        a = np.asarray(ids, dtype=np.uint32)
        with self.lock:
            self.held[:] = [
                h for h in self.held if not (len(h) <= len(a) and self._lcp(h, a) == len(h))
            ][-(self.max_held - 1):]
            self.held.append(a)


CTX_RE = re.compile(r"prompt of (\d+) tokens exceeds the context capacity of (\d+)")


def unavailable(why):
    return JSONResponse(
        {"error": {"message": f"DankAttn is not serving ({why}); retry shortly",
                   "type": "server_error", "param": None, "code": "engine_unavailable"}},
        status_code=503, headers={"Retry-After": "10"},
    )


def api_error(msg):
    m = CTX_RE.search(msg)
    if m:
        return 400, {
            "message": f"This model's maximum context length is {m.group(2)} tokens. However, your messages "
                       f"resulted in {m.group(1)} tokens. ({msg})",
            "type": "invalid_request_error", "param": "messages", "code": "context_length_exceeded",
        }
    if msg.startswith("empty prompt"):
        return 400, {"message": msg, "type": "invalid_request_error", "param": None, "code": "invalid_request"}
    if msg.startswith("group"):
        return 409, {"message": msg, "type": "group_refused", "param": "start_group", "code": "group_refused"}
    return 500, {"message": msg, "type": "server_error", "param": None, "code": "engine_error"}


def error_response(msg):
    if msg == "engine exited":
        return unavailable(msg)
    status, err = api_error(msg)
    return JSONResponse({"error": err}, status_code=status)


def bad_request(msg):
    return JSONResponse(
        {"error": {"message": msg, "type": "invalid_request_error", "param": None, "code": "invalid_request"}},
        status_code=400,
    )


class PayloadTooLarge(Exception):
    pass


DEFAULT_MAX_BODY_MB = 128
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_RESERVED_TEMPLATE_KWARGS = frozenset({"messages", "tools", "add_generation_prompt"})


def add_security_args(ap):
    """Register shared HTTP security CLI flags on an ArgumentParser."""
    ap.add_argument(
        "--allow-host", action="append", default=[],
        help="additional allowed Host/Origin hostname for reverse proxies (repeatable, or '*' to disable; env DANKATTN_ALLOW_HOSTS)",
    )
    ap.add_argument(
        "--api-key", default=os.environ.get("DANKATTN_API_KEY", ""),
        help="require Authorization: Bearer <key> on API routes (env DANKATTN_API_KEY; default: off)",
    )
    ap.add_argument(
        "--max-body-mb", type=int, default=int(os.environ.get("DANKATTN_MAX_BODY_MB", str(DEFAULT_MAX_BODY_MB))),
        help=f"maximum HTTP request body size in MiB (default: {DEFAULT_MAX_BODY_MB}; env DANKATTN_MAX_BODY_MB)",
    )


def _normalize_host(raw: str) -> str:
    h = (raw or "").strip().lower()
    if not h:
        return ""
    if h.startswith("["):
        end = h.find("]")
        return h[1:end] if end > 0 else h
    if ":" in h:
        return h.rsplit(":", 1)[0]
    return h


def _parse_allow_hosts(cli_hosts) -> set[str]:
    out = set(_LOCAL_HOSTS)
    env_val = os.environ.get("DANKATTN_ALLOW_HOSTS", "")
    for item in list(cli_hosts or ()) + env_val.split(","):
        tok = _normalize_host(item)
        if tok:
            out.add(tok)
    return out


def install_security_middleware(app, allow_hosts=(), api_key="", max_body_mb=DEFAULT_MAX_BODY_MB):
    """Enforce Host/Origin checks (DNS rebinding & CSRF), Content-Type on POST, optional Bearer auth, and body size cap."""
    allowed = _parse_allow_hosts(allow_hosts)
    allow_any_host = "*" in allowed
    max_mb = max(1, int(max_body_mb))
    max_bytes = max_mb * 1024 * 1024
    app.state.max_body_bytes = max_bytes

    @app.middleware("http")
    async def _security_mw(request: Request, call_next):
        if not allow_any_host:
            host = _normalize_host(request.headers.get("host", ""))
            if host not in allowed:
                return JSONResponse(
                    {"error": {"message": "forbidden Host header (pass --allow-host or DANKATTN_ALLOW_HOSTS for reverse proxies)",
                               "type": "invalid_request_error", "param": None, "code": "invalid_host"}},
                    status_code=403,
                )
            origin = request.headers.get("origin")
            if origin:
                ohost = (urllib.parse.urlparse(origin).hostname or "").lower()
                if ohost not in allowed:
                    return JSONResponse(
                        {"error": {"message": "forbidden Origin header",
                                   "type": "invalid_request_error", "param": None, "code": "invalid_origin"}},
                        status_code=403,
                    )

        if api_key and request.url.path != "/health":
            auth = request.headers.get("authorization", "")
            if not auth.startswith("Bearer ") or not hmac.compare_digest(auth[7:], api_key):
                return JSONResponse(
                    {"error": {"message": "invalid or missing API key",
                               "type": "authentication_error", "param": None, "code": "invalid_api_key"}},
                    status_code=401, headers={"WWW-Authenticate": "Bearer"},
                )

        if request.method == "POST":
            ctype = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if ctype != "application/json":
                return JSONResponse(
                    {"error": {"message": "Content-Type must be application/json",
                               "type": "invalid_request_error", "param": None, "code": "unsupported_media_type"}},
                    status_code=415,
                )
            cl = (request.headers.get("content-length") or "").strip()
            if cl and cl.isdigit() and int(cl) > max_bytes:
                return JSONResponse(
                    {"error": {"message": f"request body exceeds {max_mb} MiB limit",
                               "type": "invalid_request_error", "param": None, "code": "payload_too_large"}},
                    status_code=413,
                )

        return await call_next(request)


def safe_template_kwargs(body):
    kw = body.get("chat_template_kwargs") or {}
    if not isinstance(kw, dict):
        raise BadRequest("chat_template_kwargs must be an object")
    return {k: v for k, v in kw.items() if k not in _RESERVED_TEMPLATE_KWARGS}


async def json_body(request: Request):
    max_bytes = getattr(getattr(request.app, "state", None), "max_body_bytes", DEFAULT_MAX_BODY_MB * 1024 * 1024)
    chunks = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > max_bytes:
            raise PayloadTooLarge(f"request body exceeds {max_bytes // (1024 * 1024)} MiB limit")
        chunks.append(chunk)
    try:
        return json.loads(b"".join(chunks))
    except ValueError:
        return None


def payload_too_large(msg):
    return JSONResponse(
        {"error": {"message": msg, "type": "invalid_request_error", "param": None, "code": "payload_too_large"}},
        status_code=413,
    )


def num(body, key, kind, default, lo=None, hi=None):
    v = body.get(key)
    if v is None:
        return default
    try:
        if isinstance(v, bool) or (kind is int and isinstance(v, float) and not v.is_integer()):
            raise ValueError
        v = kind(v)
    except (TypeError, ValueError):
        raise BadRequest(f"{key} must be a{'n integer' if kind is int else ' number'}")
    if v != v or (lo is not None and v < lo) or (hi is not None and v > hi):
        raise BadRequest(f"{key} {v} is out of range" + (f" [{lo}, {hi}]" if lo is not None and hi is not None else ""))
    return v


class Watch:
    """Reads request items while checking ~1/s whether the HTTP client is still connected."""

    def __init__(self, q, request):
        self.q, self.request, self.at = q, request, time.monotonic()

    async def get(self):
        while True:
            if time.monotonic() - self.at >= 1.0:
                self.at = time.monotonic()
                if await self.request.is_disconnected():
                    raise ClientGone
            try:
                return await asyncio.wait_for(self.q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                pass


def register_common_routes(app, eng: EngineProc, model_id: str, cap: int, lanes_fn, render_fn, tok,
                           extra_model_fields=None):
    @app.get("/health")
    def health():
        st = dict(eng.state, since=round(time.time() - eng.state["since"], 1))
        return JSONResponse(st, status_code=200 if eng.state["state"] in ("up", "sleeping") else 503)

    @app.get("/v1/models")
    def models():
        if eng.state["state"] not in ("up", "sleeping"):
            return unavailable(eng.state["state"])
        entry = {"id": model_id, "object": "model", "owned_by": "DankAttn", "max_model_len": cap}
        if extra_model_fields:
            entry.update(extra_model_fields)
        return {"object": "list", "data": [entry]}

    @app.get("/v1/load")
    def load():
        with eng.load_lock:
            return {
                "model": model_id, "instance": eng.load["instance"], "parked": eng.state["state"] == "sleeping", "lanes": lanes_fn(),
                "running": eng.load["running"], "waiting": eng.load["waiting"],
                "tokens": {"decode": eng.load["decode"], "prefill_compute": eng.load["prefill"]},
                "busy_us": {"prefill": eng.load["prefill_us"]},
            }

    @app.post("/tokenize")
    async def tokenize(request: Request):
        try:
            body = await json_body(request)
        except PayloadTooLarge as e:
            return payload_too_large(str(e))
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
            return bad_request("messages must be a non-empty list")
        try:
            ids = tok.encode(render_fn(body), add_special_tokens=False).ids
        except BadRequest as e:
            return bad_request(str(e))
        except jinja2.exceptions.TemplateError as e:
            return bad_request(f"chat template: {e}")
        return {"tokens": ids, "count": len(ids), "max_model_len": cap}
