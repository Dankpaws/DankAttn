"""OpenAI-compatible HTTP front end for DankAttn Hy4 (Hy4-flash on DankAttn, Mac Studio).

Uses server/engine_frontend.py for engine supervision, binary framing, prefix splicing, and common routes;
this module carries Hy4's chat template (reasoning_effort max | high | no_think), <think:opensource> reasoning
split, <tool_call:opensource> tool parsing, and user stop sequences.

  python3 server/hy4_server.py --pack ./pack-g64-gptq-v3 --port 8341
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
    BadRequest, ClientGone, EngineProc, PrefixSplicer, Splitter, Unavailable, Watch,
    api_error, bad_request, error_response, json_body, make_jinja_env,
    normalize_messages, num, register_common_routes, unavailable,
)

ap = argparse.ArgumentParser()
ap.add_argument("--pack", required=True)
ap.add_argument("--engine", default=os.path.abspath(os.path.join(_HERE, "..", "build", "dankattn-hy4")))
ap.add_argument("--cap", type=int, default=262144)
ap.add_argument("--port", type=int, default=8341)
ap.add_argument("--model-id", default="hy4-flash")
ap.add_argument("--no-spec", action="store_true")
ap.add_argument("--thinking-budget", type=int,
                default=int(os.environ.get("DANKATTN_THINK_BUDGET", os.environ.get("OPUS_THINK_BUDGET", "0"))),
                help="default cap on thinking tokens (0: none); a request's thinking_budget overrides it")
args = ap.parse_args()

tok = Tokenizer.from_file(os.path.join(args.pack, "tokenizer.json"))
template = make_jinja_env().from_string(open(os.path.join(args.pack, "chat_template.jinja")).read())
HY_END = "<｜hy_end:opensource｜>"
THINK_OPEN, THINK_CLOSE = "<think:opensource>", "</think:opensource>"
STOPS = [tok.token_to_id(HY_END)]
THINK_END = tok.token_to_id(THINK_CLOSE)
THINK_FORCE = tok.encode("\n\n" + THINK_CLOSE, add_special_tokens=False).ids
EFFORTS = ("max", "high", "no_think")

eng = EngineProc(
    [args.engine, args.pack, "serve"],
    env=dict(os.environ, HY4_CAP=str(args.cap)),
    default_stops=STOPS,
    think_end=THINK_END,
    think_force=THINK_FORCE,
)
splicer = PrefixSplicer(tok, max_held=16)


def render(body):
    kw = body.get("chat_template_kwargs") or {}
    if not isinstance(kw, dict):
        raise BadRequest("chat_template_kwargs must be an object")
    msgs = normalize_messages(body)
    kw = dict(kw)
    eff = body.get("reasoning_effort")
    if eff in EFFORTS and "reasoning_effort" not in kw:
        kw["reasoning_effort"] = eff
    return template.render(messages=msgs, tools=body.get("tools"), add_generation_prompt=True, **kw)


TOOL_RE = re.compile(r"<tool_call:opensource>\s*([^<\s]+)\s*(.*?)</tool_call:opensource>", re.S)
PARAM_RE = re.compile(r"<arg_key:opensource>(.*?)</arg_key:opensource>\s*<arg_value:opensource>(.*?)</arg_value:opensource>", re.S)
CALLS_RE = re.compile(r"<tool_calls:opensource>.*?(</tool_calls:opensource>|$)", re.S)
TOOL_START = "<tool_calls:opensource>"


def parse_tools(text, enabled=True):
    calls = []
    if not enabled:
        return text.strip(), calls
    for m in TOOL_RE.finditer(text):
        a = {}
        for p in PARAM_RE.finditer(m.group(2)):
            k, v = p.group(1).strip(), p.group(2)
            try:
                a[k] = json.loads(v)
            except Exception:
                a[k] = v
        calls.append({"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
                      "function": {"name": m.group(1), "arguments": json.dumps(a, ensure_ascii=False)}})
    content = CALLS_RE.sub("", text).strip() if calls else text.strip()
    return content, calls


def _hy4_lanes():
    n_side = int(os.environ.get("HY4_SIDE_LANES", "3")) if int(os.environ.get("HY4_SIDE_CAP", "32768")) > 0 else 0
    return 1 + n_side


app = FastAPI()
register_common_routes(app, eng, args.model_id, args.cap, _hy4_lanes, render, tok)


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await json_body(request)
    if body is None:
        return bad_request("the request body must be JSON")
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
        return bad_request("messages must be a non-empty list")
    try:
        max_new = num(body, "max_tokens", int, 0, 0, 0xFFFFFFFF) or num(body, "max_completion_tokens", int, 0, 0, 0xFFFFFFFF) or 4096
        temp = num(body, "temperature", float, 0.9, 0.0, 100.0)
        top_p = num(body, "top_p", float, 1.0, 0.0, 1.0)
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
    thinking = prompt.endswith(THINK_OPEN)
    use_tools = bool(body.get("tools"))
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
    sp = Splitter(tok, thinking, THINK_CLOSE, (HY_END, THINK_OPEN))
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
                in_tool_call = False
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
                        content, calls = parse_tools(sp.content, use_tools)
                        if calls:
                            rest = CALLS_RE.sub("", held).strip()
                            if rest:
                                yield chunk({"content": rest})
                            yield chunk({"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]})
                        elif held:
                            yield chunk({"content": held})
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
                        if use_tools and not in_tool_call and TOOL_START in held:
                            in_tool_call = True
                        if not in_tool_call:
                            keep = 0
                            if use_tools and "<" in held[-24:]:
                                keep = max(keep, len(held) - held.rfind("<"))
                            if max_stop_len > 1:
                                keep = max(keep, max_stop_len - 1)
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
    content, calls = parse_tools(sp.content, use_tools)
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
