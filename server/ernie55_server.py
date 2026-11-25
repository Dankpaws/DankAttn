"""OpenAI-compatible HTTP front end for DankAttn ERNIE-5.5-Flash-Preview (Mac Studio).

Uses server/engine_frontend.py for engine supervision, binary framing, prefix splicing, and common routes;
uses Baidu's official encoding_ernie.py and decoding_ernie.py from the pack directory for exact chat
rendering, reasoning_effort (low | medium | high), <ernie:think> splitting, and <ernie:tool_call> parsing.

  python3 server/ernie55_server.py --pack ~/pack-ernie55-w2-gptq --port 8343
"""
import argparse, asyncio, json, os, re, struct, sys, threading, time, uuid
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
from tokenizers import Tokenizer
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
import uvicorn

from engine_frontend import (
    BadRequest, ClientGone, EngineProc, Unavailable, Watch,
    api_error, bad_request, error_response, json_body,
    normalize_messages, num, register_common_routes, unavailable,
)

ap = argparse.ArgumentParser()
ap.add_argument("--pack", required=True)
ap.add_argument("--engine", default=os.path.abspath(os.path.join(_HERE, "..", "build", "dankattn-ernie55")))
ap.add_argument("--cap", type=int, default=262144)
ap.add_argument("--port", type=int, default=8343)
ap.add_argument("--model-id", default="ernie-5.5-flash")
ap.add_argument("--no-spec", action="store_true")
ap.add_argument("--thinking-budget", type=int,
                default=int(os.environ.get("E5_THINK_BUDGET", "0")),
                help="default cap on thinking tokens (0: none); a request's thinking_budget overrides it")
args = ap.parse_args()

# Import official ERNIE-5.5 chat encoder/decoder bundled with the pack (handles relative import)
import importlib.util, types
_pkg = types.ModuleType("ernie_chat")
_pkg.__path__ = [args.pack]
sys.modules["ernie_chat"] = _pkg
def _load_mod(name):
    spec = importlib.util.spec_from_file_location(f"ernie_chat.{name}", os.path.join(args.pack, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"ernie_chat.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod
E_ENC = _load_mod("encoding_ernie")
E_DEC = _load_mod("decoding_ernie")

tok = Tokenizer.from_file(os.path.join(args.pack, "tokenizer.json"))
IM_END = "<|ernie:im_end|>"
PAD_TOK = "<|ERNIE_PAD|>"
THINK_OPEN, THINK_CLOSE = "<ernie:think>", "</ernie:think>"
TOOL_START, TOOL_END = "<ernie:tool_call>", "</ernie:tool_call>"


class ErnieTokenizerAdapter:
    """Source-aware ERNIE 5.5 tokenizer matching Baidu's tokenization_ernie.encode_segments.

    Only segments with allow_special=True can emit added/control token IDs; all caller
    text is BPE-encoded with added_tokens stripped so literal marker strings in prompts,
    files, or tool outputs never become live control tokens. Also implements the small
    tokenizer protocol expected by decoding_ernie.parse_completion(token_ids, ...).
    """

    def __init__(self, backend_tok: Tokenizer):
        self.tok = backend_tok
        cfg = json.loads(backend_tok.to_str())
        added = cfg.get("added_tokens", [])
        if any(t.get(k) for t in added for k in ("single_word", "lstrip", "rstrip", "normalized")):
            raise ValueError("ERNIE segment encoding requires literal added tokens")
        self._mapping = {t["content"]: t["id"] for t in added}
        self._pattern = (
            re.compile("|".join(re.escape(s) for s in sorted(self._mapping, key=len, reverse=True)))
            if self._mapping else None
        )
        cfg["added_tokens"] = []
        cfg["padding"] = None
        cfg["truncation"] = None
        self._ordinary = Tokenizer.from_str(json.dumps(cfg))
        self.pad_token_id = backend_tok.token_to_id(PAD_TOK)
        self._vocab_size = backend_tok.get_vocab_size()

    def __len__(self):
        return self._vocab_size

    def convert_tokens_to_ids(self, token: str):
        return self.tok.token_to_id(token)

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return self.tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def encode_segments(self, segments) -> list[int]:
        runs: list[str | int] = []
        buf: list[str] = []

        def flush():
            if buf:
                runs.append("".join(buf))
                buf.clear()

        for seg in segments:
            text = seg.text
            if seg.allow_special and self._pattern is not None:
                cursor = 0
                for m in self._pattern.finditer(text):
                    if m.start() > cursor:
                        buf.append(text[cursor:m.start()])
                    flush()
                    runs.append(self._mapping[m.group()])
                    cursor = m.end()
                text = text[cursor:]
            if text:
                buf.append(text)
        flush()

        out: list[int] = []
        for run in runs:
            if isinstance(run, int):
                out.append(run)
            else:
                out.extend(self._ordinary.encode(run, add_special_tokens=False).ids)
        return out


class SegmentPrefixSplicer:
    """Reuses recent request token IDs on matching prefixes while keeping tail encoding source-aware."""

    def __init__(self, adapter: ErnieTokenizerAdapter, max_held=16):
        self.adapter = adapter
        self.tok = adapter.tok
        self.max_held = max_held
        self.held: list[np.ndarray] = []
        self.lock = threading.Lock()

    @staticmethod
    def _lcp(a, b):
        n = min(len(a), len(b))
        if n == 0:
            return 0
        d = np.flatnonzero(a[:n] != b[:n])
        return int(d[0]) if len(d) else n

    def splice(self, rendered) -> list[int]:
        ids = self.adapter.encode_segments(rendered.segments)
        with self.lock:
            cands = list(self.held)
        if not cands:
            return ids
        prompt = rendered.text
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
        tail_segs = E_ENC._slice(rendered.segments, len(text), len(prompt))
        return held[:lo].tolist() + self.adapter.encode_segments(tail_segs)

    def hold(self, ids):
        a = np.asarray(ids, dtype=np.uint32)
        with self.lock:
            self.held[:] = [
                h for h in self.held if not (len(h) <= len(a) and self._lcp(h, a) == len(h))
            ][-(self.max_held - 1):]
            self.held.append(a)


class _IncDecoder:
    """Incremental UTF-8-safe detokenizer over ordinary token IDs."""

    CTX = 6

    def __init__(self, backend_tok: Tokenizer):
        self.tok = backend_tok
        self.ids: list[int] = []
        self.prefix = 0
        self.read = 0

    def push(self, t: int) -> str:
        self.ids.append(t)
        before = self.tok.decode(self.ids[self.prefix:self.read], skip_special_tokens=False)
        text = self.tok.decode(self.ids[self.prefix:], skip_special_tokens=False)
        if text.endswith("\ufffd") or len(text) <= len(before):
            return ""
        new = text[len(before):]
        self.read = len(self.ids)
        self.prefix = max(0, self.read - self.CTX)
        return new


class ErnieTokenSplitter:
    """Token-ID-based splitter for ERNIE 5.5 thinking and tool-call spans.

    Matches control tokens by exact token ID rather than decoded text, so literal
    </ernie:think> or <ernie:tool_call> strings in prose/code never corrupt state.
    """

    def __init__(self, backend_tok: Tokenizer, thinking: bool, use_tools: bool):
        self.think = thinking
        self.use_tools = use_tools
        self.in_tool_call = False
        self.has_tool_call = False
        self.all_ids: list[int] = []
        self.tool_call_ids: list[int] = []
        self.n_reasoning = 0
        self.reasoning = ""
        self.content = ""
        self._rdec = _IncDecoder(backend_tok)
        self._cdec = _IncDecoder(backend_tok)

    def push(self, t: int) -> tuple[str, str]:
        self.all_ids.append(t)
        if self.think:
            self.n_reasoning += 1
            if t == THINK_END:
                self.think = False
                return "", ""
            if t in _IGNORE_IDS:
                return "", ""
            r = self._rdec.push(t)
            if r:
                self.reasoning += r
            return r, ""
        if self.use_tools and t == TOOL_START_ID:
            self.in_tool_call = True
            self.has_tool_call = True
            self.tool_call_ids.append(t)
            return "", ""
        if self.in_tool_call:
            self.tool_call_ids.append(t)
            if t == TOOL_END_ID:
                self.in_tool_call = False
            return "", ""
        if t in _IGNORE_IDS:
            return "", ""
        c = self._cdec.push(t)
        if c:
            self.content += c
        return "", c


ernie_tok = ErnieTokenizerAdapter(tok)
IM_END_ID = tok.token_to_id(IM_END)
PAD_ID = tok.token_to_id(PAD_TOK)
THINK_OPEN_ID = tok.token_to_id(THINK_OPEN)
THINK_END = tok.token_to_id(THINK_CLOSE) or 20
TOOL_START_ID = tok.token_to_id(TOOL_START)
TOOL_END_ID = tok.token_to_id(TOOL_END)
_IGNORE_IDS = frozenset(x for x in (IM_END_ID, PAD_ID, THINK_OPEN_ID, THINK_END, 2) if x is not None)

STOPS = [x for x in (IM_END_ID, PAD_ID, 2, 8) if x is not None]
STOPS = list(dict.fromkeys(STOPS))
THINK_FORCE = ernie_tok.encode_segments([E_ENC._text("\n\n"), E_ENC._control(THINK_CLOSE)])
EFFORTS = ("low", "medium", "high")

eng = EngineProc(
    [args.engine, args.pack, "serve"],
    env=dict(os.environ, E5_CAP=str(args.cap)),
    default_stops=STOPS,
    think_end=THINK_END,
    think_force=THINK_FORCE,
)
splicer = SegmentPrefixSplicer(ernie_tok, max_held=16)


def render(body):
    kw = body.get("chat_template_kwargs") or {}
    if not isinstance(kw, dict):
        raise BadRequest("chat_template_kwargs must be an object")
    msgs = normalize_messages(body, keep_vision=False)
    enable_thinking = kw.get("enable_thinking", None)
    eff = body.get("reasoning_effort", kw.get("reasoning_effort", E_ENC.MISSING))
    if eff == "no_think" or eff == "none":
        enable_thinking = False
        eff = E_ENC.MISSING
    elif eff not in EFFORTS and eff is not E_ENC.MISSING:
        eff = "medium"
    return E_ENC.render_chat(
        msgs,
        tools=body.get("tools"),
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        reasoning_effort=eff,
    )


def _format_calls(raw_calls):
    calls = []
    for tc in raw_calls:
        fn = tc.get("function") or {}
        arg_val = fn.get("arguments", {})
        arg_str = arg_val if isinstance(arg_val, str) else json.dumps(arg_val, ensure_ascii=False)
        calls.append({
            "id": tc.get("id") or ("call_" + uuid.uuid4().hex[:12]),
            "type": "function",
            "function": {"name": fn.get("name", ""), "arguments": arg_str},
        })
    return calls


def parse_tools_from_ids(sp: ErnieTokenSplitter, rendered, tools, enabled=True):
    if not enabled or not sp.has_tool_call:
        return sp.content.strip(), []
    try:
        parsed = E_DEC.parse_completion(
            sp.all_ids,
            tokenizer=ernie_tok,
            rendered=rendered,
            tools=tools,
            argument_mode="schema",
            allow_incomplete=True,
        )
        calls = _format_calls(parsed.get("tool_calls", []))
        if calls:
            return (parsed.get("content") or "").strip(), calls
    except Exception:
        pass
    # Fallback XML parser scoped strictly to captured tool-call token IDs
    raw_call_text = tok.decode(sp.tool_call_ids, skip_special_tokens=False)
    calls = []
    for blk in re.finditer(r"<ernie:tool_call>\s*(.*?)\s*</ernie:tool_call>", raw_call_text, re.S):
        for inv in re.finditer(r'<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>', blk.group(1), re.S):
            name = inv.group(1)
            a = {}
            for p in re.finditer(r'<parameter\s+name="([^"]+)"\s*>(.*?)</parameter>', inv.group(2), re.S):
                k, v = p.group(1), p.group(2)
                try:
                    a[k] = json.loads(v)
                except Exception:
                    a[k] = v
            calls.append({
                "id": "call_" + uuid.uuid4().hex[:12],
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(a, ensure_ascii=False)},
            })
    return sp.content.strip(), calls


class _CommonTokShim:
    def encode(self, rendered, add_special_tokens=False):
        return SimpleNamespace(ids=ernie_tok.encode_segments(rendered.segments))


app = FastAPI()
register_common_routes(app, eng, args.model_id, args.cap, lambda: 1, render, _CommonTokShim())


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await json_body(request)
    if body is None:
        return bad_request("the request body must be JSON")
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
        return bad_request("messages must be a non-empty list")
    try:
        max_new = num(body, "max_tokens", int, 0, 0, 0xFFFFFFFF) or num(body, "max_completion_tokens", int, 0, 0, 0xFFFFFFFF) or 4096
        temp = num(body, "temperature", float, 1.0, 0.0, 100.0)
        top_p = num(body, "top_p", float, 0.95, 0.0, 1.0)
        top_k = max(0, num(body, "top_k", int, 0, -1, 0xFFFFFFFF))
        min_p = num(body, "min_p", float, None, 0.0, 1.0)
        seed = body.get("seed")
        seed = (num(body, "seed", int, 0) if seed is not None else uuid.uuid4().int) & ((1 << 64) - 1)
        tb = num(body, "thinking_budget", int, None, 0, 0xFFFFFFFF)
        raw_stop = body.get("stop")
        if raw_stop is None:
            stop_strs = []
        elif isinstance(raw_stop, str):
            stop_strs = [raw_stop] if raw_stop else []
        elif isinstance(raw_stop, list) and all(isinstance(s, str) for s in raw_stop) and len(raw_stop) <= 16:
            stop_strs = [s for s in raw_stop if s]
        else:
            raise BadRequest("stop must be a string or list of up to 16 strings")
    except BadRequest as e:
        return bad_request(str(e))
    try:
        rendered = render(body)
    except (BadRequest, ValueError, TypeError) as e:
        return bad_request(str(e))

    ids = await asyncio.to_thread(splicer.splice, rendered)
    extra_stops = []
    for s in stop_strs:
        sids = ernie_tok._ordinary.encode(s, add_special_tokens=False).ids
        if len(sids) == 1:
            extra_stops.append(sids[0])
    max_stop_len = max((len(s) for s in stop_strs), default=0)
    thinking = bool(rendered.generation_prefix) and rendered.generation_prefix[-1].text.endswith(THINK_OPEN)
    tools = body.get("tools")
    use_tools = bool(tools)
    spec = not args.no_spec
    budget = (tb if tb is not None else args.thinking_budget) if thinking else 0
    rid, created = "chatcmpl-" + uuid.uuid4().hex[:16], int(time.time())
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    held_ids = list(ids)

    def track(item):
        if isinstance(item, int):
            held_ids.append(item)
        elif "error" not in item:
            splicer.hold(held_ids)
        loop.call_soon_threadsafe(q.put_nowait, item)

    cb = eng.metered(track)
    try:
        eid = eng.submit(ids, max_new, spec, temp, top_p, top_k, seed, cb,
                         think_budget=budget, min_p=min_p, extra_stops=extra_stops)
    except Unavailable as e:
        cb({"error": str(e)})
        return unavailable(str(e))
    except struct.error as e:
        cb({"error": str(e)})
        return bad_request(f"request does not fit the engine protocol: {e}")

    sp = ErnieTokenSplitter(tok, thinking, use_tools)
    watch = Watch(q, request)

    def finish_of(item, calls):
        return "tool_calls" if calls else ("length" if item.get("length") or item["generated"] >= max_new else "stop")

    def chunk(delta, finish=None, usage=None):
        d = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": args.model_id,
             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if usage:
            d["usage"] = usage
        return "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"

    def usage_of(st):
        return {"prompt_tokens": st["prompt"], "completion_tokens": st["generated"],
                "total_tokens": st["prompt"] + st["generated"],
                "completion_tokens_details": {"reasoning_tokens": sp.n_reasoning},
                "dankattn": {"reused_prompt_tokens": st["reused"], "prefill_ms": round(st["prefill_ms"], 1),
                             "wall_ms": round(st["wall_ms"], 1), "spec_rounds": st["rounds"], "accepted_drafts": st["accepted"]}}

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
                stopped_by_str = False
                pending = [first]
                while True:
                    item = pending.pop() if pending else await q.get()
                    if isinstance(item, dict):
                        if "error" in item:
                            yield "data: " + json.dumps({"error": api_error(item["error"])[1]}, ensure_ascii=False) + "\n\n"
                            done = True
                            break
                        if stopped_by_str:
                            yield chunk({}, "stop", usage_of(item))
                            yield "data: [DONE]\n\n"
                            break
                        if held:
                            yield chunk({"content": held})
                            held = ""
                        _, calls = parse_tools_from_ids(sp, rendered, tools, use_tools)
                        if calls:
                            yield chunk({"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]})
                        yield chunk({}, finish_of(item, calls), usage_of(item))
                        yield "data: [DONE]\n\n"
                        break
                    if stopped_by_str:
                        continue
                    r, c = sp.push(item)
                    if r:
                        yield chunk({"reasoning_content": r})
                    if c:
                        held += c
                        if stop_strs:
                            cut_at = None
                            for s in stop_strs:
                                pos_s = held.find(s)
                                if pos_s >= 0 and (cut_at is None or pos_s < cut_at):
                                    cut_at = pos_s
                            if cut_at is not None:
                                if cut_at > 0:
                                    yield chunk({"content": held[:cut_at]})
                                held = ""
                                stopped_by_str = True
                                eng.cancel(eid)
                                continue
                        keep = max(0, max_stop_len - 1) if max_stop_len > 1 else 0
                        if len(held) > keep:
                            emit_s = held[:-keep] if keep else held
                            held = held[-keep:] if keep else ""
                            yield chunk({"content": emit_s})
                done = True
            finally:
                if not done:
                    eng.cancel(eid)

        return StreamingResponse(gen(), media_type="text/event-stream")

    stopped_by_str = False
    try:
        while True:
            item = await watch.get()
            if isinstance(item, dict):
                if "error" in item:
                    return error_response(item["error"])
                break
            if stopped_by_str:
                continue
            sp.push(item)
            if stop_strs and sp.content:
                cut_at = None
                for s in stop_strs:
                    pos_s = sp.content.find(s)
                    if pos_s >= 0 and (cut_at is None or pos_s < cut_at):
                        cut_at = pos_s
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

    content, calls = parse_tools_from_ids(sp, rendered, tools, use_tools)
    msg = {"role": "assistant", "content": content or None}
    if sp.reasoning:
        msg["reasoning_content"] = sp.reasoning
    if calls:
        msg["tool_calls"] = calls
    return {"id": rid, "object": "chat.completion", "created": created, "model": args.model_id,
            "choices": [{"index": 0, "message": msg,
                         "finish_reason": "stop" if stopped_by_str else finish_of(item, calls)}],
            "usage": usage_of(item)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
