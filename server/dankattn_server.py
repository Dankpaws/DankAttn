"""OpenAI-compatible HTTP front end for DankAttn (Flash-Next / Mac Studio).

Uses server/engine_frontend.py for engine supervision, binary framing, and common routes;
this module carries Flash-Next's chat template, fanout start_group support, <think> split,
and Qwen XML tool-call parsing.

  python3 server/dankattn_server.py --pack ./pack --port 8340
"""
import argparse, asyncio, itertools, json, os, re, struct, sys, time, uuid

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import jinja2
from tokenizers import Tokenizer
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
import uvicorn

from engine_frontend import (
    BadRequest, ClientGone, EngineProc, Splitter, Unavailable, Watch,
    api_error, bad_request, error_response, json_body, make_jinja_env,
    normalize_messages, num, register_common_routes, unavailable,
)

ap = argparse.ArgumentParser()
ap.add_argument("--pack", required=True)
ap.add_argument("--engine", default=os.path.abspath(os.path.join(_HERE, "..", "build", "dankattn")))
ap.add_argument("--cap", type=int, default=131072)
ap.add_argument("--port", type=int, default=8340)
ap.add_argument("--model-id", default="qwen38-flash-next")
ap.add_argument("--no-spec", action="store_true")
ap.add_argument("--thinking-budget", type=int,
                default=int(os.environ.get("DANKATTN_THINK_BUDGET", os.environ.get("OPUS_THINK_BUDGET", "0"))),
                help="default cap on thinking tokens (0: none); a request's thinking_budget or reasoning_effort overrides it")
args = ap.parse_args()

tok = Tokenizer.from_file(os.path.join(args.pack, "tokenizer.json"))
template = make_jinja_env().from_string(open(os.path.join(args.pack, "chat_template.jinja")).read())
STOPS = [tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")]
THINK_END = tok.token_to_id("</think>")
THINK_FORCE = tok.encode("\n\n</think>\n\n", add_special_tokens=False).ids
EFFORT_BUDGET = {"low": 2048, "medium": 8192}

LANES = int(os.environ.get("OPUS_LANES", "0"))
groups = {}
gids = itertools.count(1)

eng = EngineProc(
    [args.engine, args.pack, "serve", str(args.cap)],
    default_stops=STOPS,
    think_end=THINK_END,
    think_force=THINK_FORCE,
)


def group_of(body):
    g = body.get("start_group")
    if not g or int(g.get("count", 1)) < 2:
        return None
    key = str(g["id"])
    if key not in groups:
        if len(groups) > 1024:
            groups.pop(next(iter(groups)))
        groups[key] = next(gids)
    return groups[key], int(g["count"]), int(g.get("index", 0))


def render(body):
    kw = body.get("chat_template_kwargs") or {}
    if not isinstance(kw, dict):
        raise BadRequest("chat_template_kwargs must be an object")
    msgs = normalize_messages(body)
    return template.render(messages=msgs, tools=body.get("tools"), add_generation_prompt=True, **kw)


TOOL_RE = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_tools(text, enabled=True):
    calls = []
    if not enabled:
        return text.strip(), calls
    for m in TOOL_RE.finditer(text):
        a = {}
        for p in PARAM_RE.finditer(m.group(2)):
            v = p.group(2)
            try:
                a[p.group(1)] = json.loads(v)
            except Exception:
                a[p.group(1)] = v
        calls.append({"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
                      "function": {"name": m.group(1), "arguments": json.dumps(a, ensure_ascii=False)}})
    content = TOOL_RE.sub("", text).strip() if calls else text.strip()
    return content, calls


app = FastAPI()
register_common_routes(
    app, eng, args.model_id, args.cap, lambda: max(LANES, 1), render, tok,
    extra_model_fields={"capabilities": {"synchronized_start": LANES > 1, "max_group": LANES}},
)


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
        top_k = num(body, "top_k", int, 20, -1, 0xFFFFFFFF)
        top_k = top_k if top_k > 0 else 64
        seed = body.get("seed")
        seed = (num(body, "seed", int, 0) if seed is not None else uuid.uuid4().int) & ((1 << 64) - 1)
        tb = num(body, "thinking_budget", int, None, 0, 0xFFFFFFFF)
    except BadRequest as e:
        return bad_request(str(e))
    try:
        prompt = render(body)
    except BadRequest as e:
        return bad_request(str(e))
    except jinja2.exceptions.TemplateError as e:
        return bad_request(f"chat template: {e}")
    ids = tok.encode(prompt, add_special_tokens=False).ids
    thinking = prompt.rstrip().endswith("<think>")
    use_tools = bool(body.get("tools"))
    spec = not args.no_spec
    budget = 0
    if thinking:
        if tb is not None:
            budget = tb
        elif body.get("reasoning_effort") in EFFORT_BUDGET:
            budget = EFFORT_BUDGET[body["reasoning_effort"]]
        else:
            budget = args.thinking_budget
    rid, created = "chatcmpl-" + uuid.uuid4().hex[:16], int(time.time())
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    try:
        grp = group_of(body)
    except (TypeError, ValueError, KeyError, AttributeError):
        return bad_request("start_group must be {id, count, index} with integer count and index")
    cb = eng.metered(lambda item: loop.call_soon_threadsafe(q.put_nowait, item))
    try:
        eid = eng.submit(ids, max_new, spec, temp, top_p, top_k, seed, cb,
                         group=grp, think_budget=budget, strict=body.get("opus_strict") is True)
    except Unavailable as e:
        cb({"error": str(e)})
        return unavailable(str(e))
    except struct.error as e:
        cb({"error": str(e)})
        return bad_request(f"request does not fit the engine protocol: {e}")
    sp = Splitter(tok, thinking, "</think>", ("<|im_end|>", "<|endoftext|>"))
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
                pending = [first]
                while True:
                    item = pending.pop() if pending else await q.get()
                    if isinstance(item, dict):
                        if "error" in item:
                            yield "data: " + json.dumps({"error": api_error(item["error"])[1]}, ensure_ascii=False) + "\n\n"
                            done = True
                            break
                        content, calls = parse_tools(sp.content, use_tools)
                        if calls:
                            rest = TOOL_RE.sub("", held).strip()
                            if rest:
                                yield chunk({"content": rest})
                            yield chunk({"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]})
                        elif held:
                            yield chunk({"content": held})
                        yield chunk({}, finish_of(item, calls), usage_of(item))
                        yield "data: [DONE]\n\n"
                        break
                    r, c = sp.push(item)
                    if r:
                        yield chunk({"reasoning_content": r})
                    if c:
                        held += c
                        if not use_tools or ("<tool_call>" not in sp.content and "<" not in held[-12:]):
                            yield chunk({"content": held})
                            held = ""
                done = True
            finally:
                if not done:
                    eng.cancel(eid)

        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        while True:
            item = await watch.get()
            if isinstance(item, dict):
                if "error" in item:
                    return error_response(item["error"])
                break
            sp.push(item)
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
                         "finish_reason": finish_of(item, calls)}],
            "usage": usage_of(item)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
