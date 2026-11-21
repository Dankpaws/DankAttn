"""OpenAI-compatible HTTP front end for DankAttn Inkling-2-Mini (engine/inkling2, Mac Studio / MacBook Pro).

Uses server/engine_frontend.py for engine supervision, binary framing, prefix splicing, and common routes;
this module carries Inkling's chat template (reasoning_effort none .. max) and parses its output by message
tokens: <|message_model|>[name]<|content_thinking|> -> reasoning_content, <|content_text|> -> content,
<|content_invoke_tool_json|> {"name", "args"} -> tool_calls, each closed by <|end_message|>; the turn ends at
<|content_model_end_sampling|>.

  python3 server/inkling2_server.py --pack ./pack-inkling2-mini-w8 --port 8342
"""
import argparse, asyncio, json, os, re, struct, sys, time, uuid

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import jinja2
from tokenizers import Tokenizer
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
import uvicorn

from engine_frontend import (
    BadRequest, ClientGone, EngineProc, PrefixSplicer, Unavailable, Watch,
    api_error, bad_request, error_response, json_body, make_jinja_env,
    normalize_messages, num, register_common_routes, unavailable,
)

ap = argparse.ArgumentParser()
ap.add_argument("--pack", required=True)
ap.add_argument("--engine", default=os.path.abspath(os.path.join(_HERE, "..", "build", "dankattn-inkling2")))
ap.add_argument("--cap", type=int, default=int(os.environ.get("INK_CAP", "32768")))
ap.add_argument("--lanes", type=int, default=int(os.environ.get("INK_LANES", "4")))
ap.add_argument("--port", type=int, default=8342)
ap.add_argument("--model-id", default="inkling-2-mini")
ap.add_argument("--no-spec", action="store_true")
ap.add_argument("--min-p", type=float, default=float(os.environ.get("DANKATTN_MIN_P", "0.02")),
                help="min_p when a request sets none (0: off); at temperature 1.0 over a 200k vocabulary it keeps junk tail tokens out")
ap.add_argument("--thinking-budget", type=int, default=int(os.environ.get("DANKATTN_THINK_BUDGET", "0")),
                help="default cap on thinking tokens (0: none); a request's thinking_budget overrides it")
args = ap.parse_args()

tok = Tokenizer.from_file(os.path.join(args.pack, "tokenizer.json"))
# {% generation %} is the transformers training-mask tag (marks assistant spans); rendering ignores it.
_tpl = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", open(os.path.join(args.pack, "chat_template.jinja")).read())
_env = make_jinja_env()
# Inkling's template serializes tool specs and call args with tojson(sort_keys=true, separators=(",", ":"));
# the shared filter drops those keywords, so honour them here (the training format has no spaces).
_env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, sort_keys=False, separators=None, **k: json.dumps(
    x, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys, separators=separators)
template = _env.from_string(_tpl)
T = tok.token_to_id
MSG_MODEL, END_MSG, EOS = T("<|message_model|>"), T("<|end_message|>"), T("<|content_model_end_sampling|>")
CONTENT = {T("<|content_text|>"): "text", T("<|content_thinking|>"): "think",
           T("<|content_invoke_tool_json|>"): "tool_json", T("<|content_invoke_tool_text|>"): "tool_text",
           T("<|content_xml|>"): "xml", T("<|content_tool_error|>"): "text"}
# The model never legitimately opens a user / system / tool turn: stop there too.
STOPS = [EOS, T("<|message_user|>"), T("<|message_system|>"), T("<|message_tool|>")]
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

eng = EngineProc(
    [args.engine, args.pack, "serve"],
    env=dict(os.environ, INK_CAP=str(args.cap), INK_LANES=str(args.lanes)),
    default_stops=STOPS,
    think_end=END_MSG,
    think_force=[END_MSG],
)
splicer = PrefixSplicer(tok, max_held=16)


def render(body):
    kw = body.get("chat_template_kwargs") or {}
    if not isinstance(kw, dict):
        raise BadRequest("chat_template_kwargs must be an object")
    kw = dict(kw)
    eff = body.get("reasoning_effort")
    if eff is not None and "reasoning_effort" not in kw:
        if eff not in EFFORTS:
            raise BadRequest(f"reasoning_effort must be one of {', '.join(EFFORTS)}")
        kw["reasoning_effort"] = eff
    if kw.pop("enable_thinking", True) is False and "reasoning_effort" not in kw:
        kw["reasoning_effort"] = "none"
    return template.render(messages=normalize_messages(body), tools=body.get("tools"), **kw)


class InkParser:
    """Routes generated token ids into reasoning / content / tool calls by Inkling's message tokens."""

    CTX = 6

    def __init__(self):
        self.state, self.chan = "idle", None
        self.header, self.body = [], []
        self.prefix = self.read = 0
        self.reasoning, self.content = "", ""
        self.calls, self.n_reasoning = [], 0

    def _start(self, chan):
        self.chan, self.state, self.body = chan, "body", []
        self.prefix = self.read = 0

    def _delta(self):
        before = tok.decode(self.body[self.prefix:self.read], skip_special_tokens=False)
        text = tok.decode(self.body[self.prefix:], skip_special_tokens=False)
        if text.endswith("�") or len(text) <= len(before):
            return ""
        self.read = len(self.body)
        self.prefix = max(0, self.read - self.CTX)
        return text[len(before):]

    def _close(self):
        ev = []
        if self.state == "body" and self.chan in ("tool_json", "tool_text"):
            raw = tok.decode(self.body, skip_special_tokens=False).strip()
            name = tok.decode(self.header, skip_special_tokens=False).strip()
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and "args" in obj:
                    name = obj.get("name") or name
                    argv = obj["args"]
                else:
                    argv = obj
            except ValueError:
                argv = {"input": raw}
            call = {"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
                    "function": {"name": name, "arguments": json.dumps(argv, ensure_ascii=False)}}
            self.calls.append(call)
            ev.append(("tool", call))
        self.state, self.chan, self.header = "idle", None, []
        return ev

    def push(self, t):
        """Returns [(kind, value)] with kind in reasoning / content / tool."""
        if t == MSG_MODEL:
            ev = self._close()
            self.state = "header"
            return ev
        if t in CONTENT:
            ev = self._close() if self.state == "body" else []
            hdr = self.header if self.state == "header" else []
            self._start(CONTENT[t])
            self.header = hdr
            return ev
        if t == END_MSG or t == EOS:
            return self._close()
        if self.state == "idle":  # bare text without a header: treat as content
            self._start("text")
        if self.state == "header":
            self.header.append(t)
            return []
        self.body.append(t)
        if self.chan == "think":
            self.n_reasoning += 1
        if self.chan not in ("think", "text"):
            return []
        d = self._delta()
        if not d:
            return []
        if self.chan == "think":
            self.reasoning += d
            return [("reasoning", d)]
        self.content += d
        return [("content", d)]


app = FastAPI()
register_common_routes(app, eng, args.model_id, args.cap, lambda: args.lanes, render, tok)


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await json_body(request)
    if body is None:
        return bad_request("the request body must be JSON")
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
        return bad_request("messages must be a non-empty list")
    try:
        max_new = num(body, "max_tokens", int, 0, 0, 0xFFFFFFFF) or num(body, "max_completion_tokens", int, 0, 0, 0xFFFFFFFF) or 8192
        temp = num(body, "temperature", float, 1.0, 0.0, 100.0)
        top_p = num(body, "top_p", float, 1.0, 0.0, 1.0)
        top_k = max(0, num(body, "top_k", int, 0, -1, 0xFFFFFFFF))
        min_p = num(body, "min_p", float, None, 0.0, 1.0)
        if min_p is None:
            min_p = args.min_p
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
        prompt = render(body)
    except BadRequest as e:
        return bad_request(str(e))
    except jinja2.exceptions.TemplateError as e:
        return bad_request(f"chat template: {e}")
    ids = await asyncio.to_thread(splicer.splice, prompt)
    extra_stops = []
    for s in stop_strs:
        sids = tok.encode(s, add_special_tokens=False).ids
        if len(sids) == 1:
            extra_stops.append(sids[0])
    max_stop_len = max((len(s) for s in stop_strs), default=0)
    budget = tb if tb is not None else args.thinking_budget
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
        eid = eng.submit(ids, max_new, not args.no_spec, temp, top_p, top_k, seed, cb,
                         think_budget=budget, min_p=min_p, extra_stops=extra_stops)
    except Unavailable as e:
        cb({"error": str(e)})
        return unavailable(str(e))
    except struct.error as e:
        cb({"error": str(e)})
        return bad_request(f"request does not fit the engine protocol: {e}")
    ps = InkParser()
    watch = Watch(q, request)

    def finish_of(item):
        return "tool_calls" if ps.calls else ("length" if item.get("length") or item["generated"] >= max_new else "stop")

    def usage_of(st):
        return {"prompt_tokens": st["prompt"], "completion_tokens": st["generated"],
                "total_tokens": st["prompt"] + st["generated"],
                "completion_tokens_details": {"reasoning_tokens": ps.n_reasoning},
                "dankattn": {"reused_prompt_tokens": st["reused"], "prefill_ms": round(st["prefill_ms"], 1),
                             "wall_ms": round(st["wall_ms"], 1), "spec_rounds": st["rounds"], "accepted": st["accepted"]}}

    def first_stop(text):
        cut = None
        for s in stop_strs:
            p = text.find(s)
            if p >= 0 and (cut is None or p < cut):
                cut = p
        return cut

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

        def chunk(delta, finish=None, usage=None):
            d = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": args.model_id,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if usage:
                d["usage"] = usage
            return "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"

        async def gen():
            done = False
            try:
                yield chunk({"role": "assistant", "content": ""})
                held, stopped, n_calls = "", False, 0
                pending = [first]
                while True:
                    item = pending.pop() if pending else await q.get()
                    if isinstance(item, dict):
                        if "error" in item:
                            yield "data: " + json.dumps({"error": api_error(item["error"])[1]}, ensure_ascii=False) + "\n\n"
                            done = True
                            break
                        if held and not stopped:
                            yield chunk({"content": held})
                        yield chunk({}, "stop" if stopped else finish_of(item), usage_of(item))
                        yield "data: [DONE]\n\n"
                        break
                    if stopped:
                        continue
                    for kind, v in ps.push(item):
                        if kind == "reasoning":
                            yield chunk({"reasoning_content": v})
                        elif kind == "tool":
                            yield chunk({"tool_calls": [dict(v, index=n_calls)]})
                            n_calls += 1
                        else:
                            held += v
                            cut = first_stop(held) if stop_strs else None
                            if cut is not None:
                                if cut > 0:
                                    yield chunk({"content": held[:cut]})
                                held, stopped = "", True
                                eng.cancel(eid)
                                break
                            keep = max_stop_len - 1 if max_stop_len > 1 else 0
                            if len(held) > keep:
                                yield chunk({"content": held[:len(held) - keep] if keep else held})
                                held = held[len(held) - keep:] if keep else ""
                done = True
            finally:
                if not done:
                    eng.cancel(eid)

        return StreamingResponse(gen(), media_type="text/event-stream")

    stopped = False
    try:
        while True:
            item = await watch.get()
            if isinstance(item, dict):
                if "error" in item:
                    return error_response(item["error"])
                break
            if stopped:
                continue
            ps.push(item)
            if stop_strs:
                cut = first_stop(ps.content)
                if cut is not None:
                    ps.content = ps.content[:cut]
                    stopped = True
                    eng.cancel(eid)
    except ClientGone:
        eng.cancel(eid)
        return Response(status_code=499)
    except asyncio.CancelledError:
        eng.cancel(eid)
        raise
    msg = {"role": "assistant", "content": ps.content.strip() or None}
    if ps.reasoning:
        msg["reasoning_content"] = ps.reasoning
    if ps.calls:
        msg["tool_calls"] = ps.calls
    return {"id": rid, "object": "chat.completion", "created": created, "model": args.model_id,
            "choices": [{"index": 0, "message": msg, "finish_reason": "stop" if stopped else finish_of(item)}],
            "usage": usage_of(item)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
