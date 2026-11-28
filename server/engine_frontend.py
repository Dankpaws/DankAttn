"""Shared engine supervision, binary framing, prefix splicing, and OpenAI HTTP helpers for DankAttn front ends."""
import asyncio, hmac, itertools, json, os, re, struct, subprocess, sys, threading, time, urllib.parse, uuid

import jinja2
import numpy as np
from jinja2.sandbox import ImmutableSandboxedEnvironment
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from anthropic_compat import register_anthropic_routes
from json_schema_dfa import JsonSchemaDFA, VocabMasker, parse_response_format, schema_system_hint, validate_json_schema


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
    rf = parse_response_format(body)
    hint = schema_system_hint(rf) if rf else ""
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
    if hint:
        if msgs and msgs[0].get("role") == "system" and isinstance(msgs[0].get("content"), str):
            msgs[0] = dict(msgs[0], content=msgs[0]["content"] + "\n\n" + hint)
        else:
            msgs.insert(0, {"role": "system", "content": hint})
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
        self.metrics = {
            "requests_total": 0, "requests_errors": 0,
            "prompt_tokens_total": 0, "reused_prompt_tokens_total": 0,
            "generated_tokens_total": 0, "spec_rounds_total": 0, "accepted_drafts_total": 0,
            "prefill_ms_total": 0.0, "wall_ms_total": 0.0,
            "last_decode_tps": 0.0, "last_prefill_tps": 0.0,
        }
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
                        pf_toks = max(0, item["prompt"] - item["reused"])
                        self.load["prefill"] += pf_toks
                        self.load["prefill_us"] += int(item["prefill_ms"] * 1000)
                        self.metrics["requests_total"] += 1
                        self.metrics["prompt_tokens_total"] += item["prompt"]
                        self.metrics["reused_prompt_tokens_total"] += item["reused"]
                        self.metrics["generated_tokens_total"] += item["generated"]
                        self.metrics["spec_rounds_total"] += item["rounds"]
                        self.metrics["accepted_drafts_total"] += item["accepted"]
                        self.metrics["prefill_ms_total"] += float(item["prefill_ms"])
                        self.metrics["wall_ms_total"] += float(item["wall_ms"])
                        dec_ms = max(0.0, float(item["wall_ms"]) - float(item["prefill_ms"]))
                        if item["generated"] > 0 and dec_ms > 0:
                            self.metrics["last_decode_tps"] = round(item["generated"] * 1000.0 / dec_ms, 2)
                        if pf_toks > 0 and item["prefill_ms"] > 0:
                            self.metrics["last_prefill_tps"] = round(pf_toks * 1000.0 / float(item["prefill_ms"]), 2)
                    elif "error" in item:
                        self.metrics["requests_errors"] += 1
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
                elif w == 0xFFFFFFFB:
                    tok_id, lp, n_top = struct.unpack("<IfI", read_exact(out, 12))
                    tops = []
                    for _ in range(n_top):
                        tid, tlp = struct.unpack("<If", read_exact(out, 8))
                        tops.append((tid, tlp))
                    item = ("logprob", tok_id, lp, tops)
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
               group=None, think_budget=0, strict=False, min_p=None, extra_stops=(),
               vision_imgs=None, fpr_suffixes=None, constrained=False, top_logprobs=None):
        self.ensure_awake()
        rid = next(self.rids)
        stops = list(self.default_stops)
        for t in extra_stops:
            if t not in stops:
                stops.append(t)
        req = struct.pack("<III", 0x5355504F, rid, len(ids)) + struct.pack(f"<{len(ids)}I", *ids)
        flags = (
            (1 if (spec and not constrained) else 0)
            | (2 if group else 0)
            | (4 if think_budget else 0)
            | (8 if strict else 0)
            | (16 if min_p is not None else 0)
            | (32 if vision_imgs else 0)
            | (64 if constrained else 0)
            | (128 if top_logprobs is not None else 0)
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
        if top_logprobs is not None:
            req += struct.pack("<I", max(0, min(20, int(top_logprobs))))
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

    def send_mask(self, rid, allowed_ids):
        """Send an OPMK allowed-token-ID list for a constrained-decoding request."""
        n = len(allowed_ids)
        pkt = struct.pack("<III", 0x4B4D504F, rid, n) + (struct.pack(f"<{n}I", *allowed_ids) if n else b"")
        try:
            with self.write_lock:
                self.engine.stdin.write(pkt)
                self.engine.stdin.flush()
        except (OSError, ValueError):
            pass

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

    def _metrics_snapshot():
        with eng.load_lock:
            m = dict(eng.metrics)
            ld = dict(eng.load)
        st = dict(eng.state)
        uptime_s = round(max(0.0, time.time() - st["since"]), 1)
        acc_rate = round(m["accepted_drafts_total"] / m["spec_rounds_total"], 3) if m["spec_rounds_total"] > 0 else 0.0
        reuse_rate = round(m["reused_prompt_tokens_total"] / m["prompt_tokens_total"], 4) if m["prompt_tokens_total"] > 0 else 0.0
        return {
            "model": model_id,
            "state": st["state"],
            "pid": st["pid"],
            "uptime_seconds": uptime_s,
            "restarts_total": st["restarts"],
            "lanes_capacity": lanes_fn(),
            "requests_running": ld["running"],
            "requests_waiting": ld["waiting"],
            "spec_acceptance_per_round": acc_rate,
            "prefix_reuse_ratio": reuse_rate,
            **m,
        }

    @app.get("/v1/metrics")
    def metrics_json():
        return _metrics_snapshot()

    @app.get("/metrics")
    def metrics_prom(request: Request):
        snap = _metrics_snapshot()
        if request.query_params.get("format") == "json" or "application/json" in request.headers.get("accept", ""):
            return JSONResponse(snap)
        up = 1 if snap["state"] == "up" else 0
        lines = [
            f'# HELP dankattn_up Engine readiness (1=up, 0=sleeping/down)',
            f'# TYPE dankattn_up gauge',
            f'dankattn_up{{model="{model_id}"}} {up}',
            f'dankattn_uptime_seconds{{model="{model_id}"}} {snap["uptime_seconds"]}',
            f'dankattn_restarts_total{{model="{model_id}"}} {snap["restarts_total"]}',
            f'dankattn_lanes_capacity{{model="{model_id}"}} {snap["lanes_capacity"]}',
            f'dankattn_requests_running{{model="{model_id}"}} {snap["requests_running"]}',
            f'dankattn_requests_waiting{{model="{model_id}"}} {snap["requests_waiting"]}',
            f'dankattn_requests_total{{model="{model_id}"}} {snap["requests_total"]}',
            f'dankattn_requests_errors_total{{model="{model_id}"}} {snap["requests_errors"]}',
            f'dankattn_prompt_tokens_total{{model="{model_id}"}} {snap["prompt_tokens_total"]}',
            f'dankattn_reused_prompt_tokens_total{{model="{model_id}"}} {snap["reused_prompt_tokens_total"]}',
            f'dankattn_generated_tokens_total{{model="{model_id}"}} {snap["generated_tokens_total"]}',
            f'dankattn_spec_rounds_total{{model="{model_id}"}} {snap["spec_rounds_total"]}',
            f'dankattn_accepted_drafts_total{{model="{model_id}"}} {snap["accepted_drafts_total"]}',
            f'dankattn_spec_acceptance_per_round{{model="{model_id}"}} {snap["spec_acceptance_per_round"]}',
            f'dankattn_prefix_reuse_ratio{{model="{model_id}"}} {snap["prefix_reuse_ratio"]}',
            f'dankattn_last_decode_tps{{model="{model_id}"}} {snap["last_decode_tps"]}',
            f'dankattn_last_prefill_tps{{model="{model_id}"}} {snap["last_prefill_tps"]}',
        ]
        return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4; charset=utf-8")

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

    register_anthropic_routes(app, model_id, render_fn, tok, json_body, payload_too_large, PayloadTooLarge)


def parse_chat_params(body, *, default_max_new=4096, default_temp=1.0, default_top_p=1.0,
                      default_top_k=0, default_min_p=None, parse_stops=True):
    """Parse common OpenAI sampling, budget, and stop-sequence fields from a chat request body."""
    max_new = num(body, "max_tokens", int, 0, 0, 0xFFFFFFFF) or num(body, "max_completion_tokens", int, 0, 0, 0xFFFFFFFF) or default_max_new
    temp = num(body, "temperature", float, default_temp, 0.0, 100.0)
    top_p = num(body, "top_p", float, default_top_p, 0.0, 1.0)
    top_k = max(0, num(body, "top_k", int, default_top_k, -1, 0xFFFFFFFF))
    min_p = num(body, "min_p", float, default_min_p, 0.0, 1.0)
    seed = body.get("seed")
    seed = (num(body, "seed", int, 0) if seed is not None else uuid.uuid4().int) & ((1 << 64) - 1)
    tb = num(body, "thinking_budget", int, None, 0, 0xFFFFFFFF)
    stop_strs = []
    if parse_stops:
        raw_stop = body.get("stop")
        if raw_stop is None:
            stop_strs = []
        elif isinstance(raw_stop, str):
            stop_strs = [raw_stop] if raw_stop else []
        elif isinstance(raw_stop, list) and all(isinstance(s, str) for s in raw_stop) and len(raw_stop) <= 16:
            stop_strs = [s for s in raw_stop if s]
        else:
            raise BadRequest("stop must be a string or list of up to 16 strings")
    logprobs = bool(body.get("logprobs"))
    top_logprobs = num(body, "top_logprobs", int, 0, 0, 20) if logprobs else None
    rf = parse_response_format(body)
    return {
        "max_new": max_new, "temp": temp, "top_p": top_p, "top_k": top_k,
        "min_p": min_p, "seed": seed, "thinking_budget": tb, "stop_strs": stop_strs,
        "logprobs": logprobs, "top_logprobs": top_logprobs, "response_format": rf,
    }


_MASKER_CACHE: dict[int, VocabMasker] = {}
_MASKER_LOCK = threading.Lock()


def get_vocab_masker(tok) -> VocabMasker:
    key = id(tok)
    with _MASKER_LOCK:
        m = _MASKER_CACHE.get(key)
        if m is None:
            m = VocabMasker(tok)
            _MASKER_CACHE[key] = m
        return m


class ConstrainedDriver:
    """Drives OPMK token-level JSON-schema masks from the demux thread on every emitted token."""

    def __init__(self, eng: EngineProc, tok, rf: dict, eos_ids: tuple[int, ...], thinking: bool = False,
                 think_close: str = ""):
        self.eng = eng
        self.tok = tok
        self.masker = get_vocab_masker(tok)
        self.dfa = JsonSchemaDFA(rf.get("schema"))
        self.eos_ids = tuple(int(x) for x in eos_ids if x is not None)
        self.in_think = bool(thinking and think_close)
        self.think_close = think_close
        self._think_buf = ""
        self.rid: int | None = None

    def start(self, rid: int):
        self.rid = rid
        if self.in_think:
            self.eng.send_mask(rid, ())
        else:
            self.eng.send_mask(rid, self.masker.allowed_ids(self.dfa, self.eos_ids))

    def on_token(self, tok_id: int):
        if self.rid is None:
            return
        if tok_id in self.eos_ids:
            return
        piece = self.tok.decode([tok_id], skip_special_tokens=False)
        if self.in_think:
            self._think_buf += piece
            if self.think_close in self._think_buf:
                idx = self._think_buf.find(self.think_close) + len(self.think_close)
                tail = self._think_buf[idx:]
                self.in_think = False
                self._think_buf = ""
                if tail:
                    self.dfa.feed(tail)
                self.eng.send_mask(self.rid, self.masker.allowed_ids(self.dfa, self.eos_ids))
            else:
                self.eng.send_mask(self.rid, ())
            return
        self.dfa.feed(piece)
        self.eng.send_mask(self.rid, self.masker.allowed_ids(self.dfa, self.eos_ids))


def format_logprob_entry(tok, tok_id: int, lp: float, tops) -> dict:
    s = tok.decode([tok_id], skip_special_tokens=False)
    top_list = []
    for tid, tlp in tops:
        ts = tok.decode([tid], skip_special_tokens=False)
        top_list.append({"token": ts, "logprob": round(float(tlp), 6), "bytes": list(ts.encode("utf-8"))})
    return {
        "token": s, "logprob": round(float(lp), 6), "bytes": list(s.encode("utf-8")),
        "top_logprobs": top_list,
    }


def find_stop_cut(text: str, stop_strs) -> int | None:
    cut_at = None
    for s in stop_strs:
        pos_s = text.find(s)
        if pos_s >= 0 and (cut_at is None or pos_s < cut_at):
            cut_at = pos_s
    return cut_at


def finish_reason_of(item, calls, max_new: int) -> str:
    return "tool_calls" if calls else ("length" if item.get("length") or item["generated"] >= max_new else "stop")


def make_sse_chunk(rid: str, created: int, model_id: str, delta: dict, finish=None, usage=None, logprobs=None) -> str:
    choice = {"index": 0, "delta": delta, "finish_reason": finish}
    if logprobs is not None:
        choice["logprobs"] = logprobs
    d = {
        "id": rid, "object": "chat.completion.chunk", "created": created, "model": model_id,
        "choices": [choice],
    }
    if usage:
        d["usage"] = usage
    return "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"


def make_usage(st: dict, n_reasoning: int) -> dict:
    return {
        "prompt_tokens": st["prompt"],
        "completion_tokens": st["generated"],
        "total_tokens": st["prompt"] + st["generated"],
        "completion_tokens_details": {"reasoning_tokens": n_reasoning},
        "dankattn": {
            "reused_prompt_tokens": st["reused"],
            "prefill_ms": round(st["prefill_ms"], 1),
            "wall_ms": round(st["wall_ms"], 1),
            "spec_rounds": st["rounds"],
            "accepted_drafts": st["accepted"],
        },
    }


async def run_chat_completion(
    request: Request,
    body: dict,
    eng: EngineProc,
    eid: int,
    q: asyncio.Queue,
    sp,
    *,
    rid: str,
    created: int,
    model_id: str,
    max_new: int,
    stop_strs,
    parse_tools_fn,
    use_tools: bool,
    tool_start: str | None = None,
    tool_strip_re=None,
    tool_lookback: int = 24,
):
    """Shared streaming and non-streaming OpenAI chat completion driver."""
    watch = Watch(q, request)
    max_stop_len = max((len(s) for s in stop_strs), default=0)

    def chunk(delta, finish=None, usage=None, lp=None):
        return make_sse_chunk(rid, created, model_id, delta, finish, usage, logprobs=lp)

    def usage_of(st):
        return make_usage(st, sp.n_reasoning)

    tok_ref = getattr(sp, "tok", None)
    want_lp = bool(body.get("logprobs")) and tok_ref is not None
    lp_content: list[dict] = []

    if body.get("stream"):
        try:
            first = await watch.get()
        except ClientGone:
            eng.cancel(eid)
            return Response(status_code=499)
        except asyncio.CancelledError:
            eng.cancel(eid)
            raise
        if isinstance(first, dict) and "error" in first:
            return error_response(first["error"])

        async def gen():
            done = False
            try:
                yield chunk({"role": "assistant", "content": ""})
                held = ""
                in_tool_call = False
                stopped_by_str = False
                pending = [first]
                pending_lp = None
                while True:
                    item = pending.pop() if pending else await q.get()
                    if isinstance(item, tuple) and item and item[0] == "logprob":
                        if want_lp:
                            pending_lp = format_logprob_entry(tok_ref, item[1], item[2], item[3])
                        continue
                    if isinstance(item, dict):
                        if "error" in item:
                            yield "data: " + json.dumps({"error": api_error(item["error"])[1]}, ensure_ascii=False) + "\n\n"
                            done = True
                            break
                        if stopped_by_str:
                            yield chunk({}, "stop", usage_of(item))
                            yield "data: [DONE]\n\n"
                            break
                        _, calls = parse_tools_fn()
                        if calls:
                            if tool_strip_re is not None:
                                rest = tool_strip_re.sub("", held).strip()
                                if rest:
                                    yield chunk({"content": rest})
                            elif held:
                                yield chunk({"content": held})
                            yield chunk({"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]})
                        elif held:
                            yield chunk({"content": held})
                        yield chunk({}, finish_reason_of(item, calls, max_new), usage_of(item))
                        yield "data: [DONE]\n\n"
                        break
                    if stopped_by_str:
                        continue
                    r, c = sp.push(item)
                    lp_obj = {"content": [pending_lp]} if pending_lp is not None else None
                    pending_lp = None
                    if r:
                        yield chunk({"reasoning_content": r}, lp=lp_obj)
                        lp_obj = None
                    if c:
                        held += c
                        if stop_strs:
                            cut_at = find_stop_cut(held, stop_strs)
                            if cut_at is not None:
                                if cut_at > 0:
                                    yield chunk({"content": held[:cut_at]}, lp=lp_obj)
                                held = ""
                                stopped_by_str = True
                                eng.cancel(eid)
                                continue
                        if tool_start and use_tools and not in_tool_call and tool_start in held:
                            in_tool_call = True
                        if not in_tool_call:
                            keep = 0
                            if tool_start and use_tools and "<" in held[-tool_lookback:]:
                                keep = max(keep, len(held) - held.rfind("<"))
                            if max_stop_len > 1:
                                keep = max(keep, max_stop_len - 1)
                            if len(held) > keep:
                                emit_s = held[:-keep] if keep else held
                                held = held[-keep:] if keep else ""
                                yield chunk({"content": emit_s}, lp=lp_obj)
                done = True
            finally:
                if not done:
                    eng.cancel(eid)

        return StreamingResponse(gen(), media_type="text/event-stream")

    stopped_by_str = False
    try:
        while True:
            item = await watch.get()
            if isinstance(item, tuple) and item and item[0] == "logprob":
                if want_lp:
                    lp_content.append(format_logprob_entry(tok_ref, item[1], item[2], item[3]))
                continue
            if isinstance(item, dict):
                if "error" in item:
                    return error_response(item["error"])
                break
            if stopped_by_str:
                continue
            sp.push(item)
            if stop_strs and sp.content:
                cut_at = find_stop_cut(sp.content, stop_strs)
                if cut_at is not None:
                    sp.content = sp.content[:cut_at]
                    stopped_by_str = True
                    eng.cancel(eid)
    except ClientGone:
        eng.cancel(eid)
        return Response(status_code=499)
    except asyncio.CancelledError:
        eng.cancel(eid)
        raise

    content, calls = parse_tools_fn()
    rf = parse_response_format(body)
    schema_meta = None
    if rf and content:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 3 and lines[-1].strip() == "```":
                cleaned = "\n".join(lines[1:-1]).strip()
        try:
            parsed_obj = json.loads(cleaned)
            ok, err = validate_json_schema(parsed_obj, rf.get("schema"))
            content = cleaned
            schema_meta = {"valid": ok, **({"error": err} if err else {})}
        except Exception as e:
            schema_meta = {"valid": False, "error": f"invalid json: {e}"}
    msg = {"role": "assistant", "content": content or None}
    if sp.reasoning:
        msg["reasoning_content"] = sp.reasoning
    if calls:
        msg["tool_calls"] = calls
    usage = usage_of(item)
    if schema_meta is not None:
        usage["dankattn"]["schema"] = schema_meta
    choice_obj = {
        "index": 0, "message": msg,
        "finish_reason": "stop" if stopped_by_str else finish_reason_of(item, calls, max_new),
    }
    if want_lp:
        choice_obj["logprobs"] = {"content": lp_content}
    return {
        "id": rid, "object": "chat.completion", "created": created, "model": model_id,
        "choices": [choice_obj],
        "usage": usage,
    }
