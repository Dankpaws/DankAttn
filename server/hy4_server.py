"""OpenAI-compatible HTTP front end for DankAttn Hy4 (Hy4-flash on DankAttn, Mac Studio).

Uses server/engine_frontend.py for engine supervision, binary framing, prefix splicing, and common routes;
this module carries Hy4's chat template (reasoning_effort high | no_think; max -> high, enable_thinking: false -> no_think), <think:opensource> reasoning
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
    BadRequest, EngineProc, PayloadTooLarge, PrefixSplicer, Splitter, Unavailable,
    add_security_args, bad_request, install_security_middleware,
    json_body, make_jinja_env, normalize_messages, parse_chat_params, payload_too_large,
    register_common_routes, run_chat_completion, safe_template_kwargs, unavailable,
)
from vision_preproc import MAX_IMAGES, extract_image_urls, preprocess_image

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
ap.add_argument("--idle-unload", type=float,
                default=float(os.environ.get("DANKATTN_IDLE_UNLOAD", "0")),
                help="stop engine to release GPU RAM after N idle seconds (0: stay loaded; wakes via pcache)")
ap.add_argument("--allow-local-media", default=os.environ.get("DANKATTN_ALLOW_LOCAL_MEDIA", ""),
                help="optional directory from which file:// or local image paths may be loaded (default: disabled)")
add_security_args(ap)
args = ap.parse_args()

tok = Tokenizer.from_file(os.path.join(args.pack, "tokenizer.json"))
template = make_jinja_env().from_string(open(os.path.join(args.pack, "chat_template.jinja")).read())
HY_END = "<｜hy_end:opensource｜>"
IMAGE_PAD = "<｜hy_image▁pad:opensource｜>"
THINK_OPEN, THINK_CLOSE = "<think:opensource>", "</think:opensource>"
STOPS = [tok.token_to_id(HY_END)]
IMAGE_PAD_ID = tok.token_to_id(IMAGE_PAD)
THINK_END = tok.token_to_id(THINK_CLOSE)
THINK_FORCE = tok.encode("\n\n" + THINK_CLOSE, add_special_tokens=False).ids
# Upstream tencent/Hy4-flash (b2498eba, 2026-11-19) dropped 'max' from chat_template.jinja;
# map legacy 'max' -> 'high' and OpenAI-style 'none'/'off' -> 'no_think'.
EFFORT_MAP = {"max": "high", "high": "high", "no_think": "no_think", "none": "no_think", "off": "no_think"}

eng = EngineProc(
    [args.engine, args.pack, "serve"],
    env=dict(os.environ, HY4_CAP=str(args.cap)),
    default_stops=STOPS,
    think_end=THINK_END,
    think_force=THINK_FORCE,
    idle_unload_s=args.idle_unload,
)
splicer = PrefixSplicer(tok, max_held=16)


def render(body):
    kw = safe_template_kwargs(body)
    msgs = normalize_messages(body, keep_vision=True)
    et_kw = kw.pop("enable_thinking", None)
    et_top = body.get("enable_thinking", None)
    et = et_kw if et_kw is not None else et_top
    if "reasoning_effort" in kw:
        if kw["reasoning_effort"] in EFFORT_MAP:
            kw["reasoning_effort"] = EFFORT_MAP[kw["reasoning_effort"]]
    else:
        eff = body.get("reasoning_effort")
        if eff in EFFORT_MAP:
            kw["reasoning_effort"] = EFFORT_MAP[eff]
    if et is False and "reasoning_effort" not in kw:
        kw["reasoning_effort"] = "no_think"
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
install_security_middleware(app, allow_hosts=args.allow_host, api_key=args.api_key, max_body_mb=args.max_body_mb)
register_common_routes(app, eng, args.model_id, args.cap, _hy4_lanes, render, tok)


@app.post("/v1/chat/completions")
async def chat(request: Request):
    try:
        body = await json_body(request)
    except PayloadTooLarge as e:
        return payload_too_large(str(e))
    if body is None:
        return bad_request("the request body must be JSON")
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
        return bad_request("messages must be a non-empty list")
    try:
        p = parse_chat_params(body, default_max_new=4096, default_temp=0.9, default_top_p=1.0, default_top_k=0)
    except BadRequest as e:
        return bad_request(str(e))
    max_new, temp, top_p, top_k, min_p, seed, tb, stop_strs = (
        p["max_new"], p["temp"], p["top_p"], p["top_k"], p["min_p"], p["seed"], p["thinking_budget"], p["stop_strs"]
    )
    try:
        prompt = render(body)
        img_urls = extract_image_urls(body["messages"])
    except (BadRequest, ValueError) as e:
        return bad_request(str(e))
    except jinja2.exceptions.TemplateError as e:
        return bad_request(f"chat template: {e}")
    vision_imgs = None
    if img_urls:
        if len(img_urls) > MAX_IMAGES:
            return bad_request(f"too many images ({len(img_urls)} > {MAX_IMAGES})")
        try:
            vision_imgs = await asyncio.to_thread(
                lambda: [preprocess_image(u, allow_local_media=args.allow_local_media) for u in img_urls]
            )
        except Exception as e:
            return bad_request(f"image preprocessing failed: {e}")
    raw_ids = await asyncio.to_thread(splicer.splice, prompt)
    try:
        await asyncio.to_thread(eng.ensure_awake)
    except Unavailable as e:
        return unavailable(str(e))
    if vision_imgs:
        ids = []
        img_i = 0
        for tid in raw_ids:
            if tid == IMAGE_PAD_ID and img_i < len(vision_imgs):
                im = vision_imgs[img_i]
                im["tok_off"] = len(ids)
                ids.extend([IMAGE_PAD_ID] * im["n_tokens"])
                img_i += 1
            else:
                ids.append(tid)
        if img_i != len(vision_imgs):
            return bad_request(f"image placeholder mismatch: template emitted {img_i}, request had {len(vision_imgs)}")
    else:
        ids = raw_ids
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
                         think_budget=budget, min_p=min_p, extra_stops=extra_stops, vision_imgs=vision_imgs)
    except Unavailable as e:
        cb({"error": str(e)})
        return unavailable(str(e))
    except struct.error as e:
        cb({"error": str(e)})
        return bad_request(f"request does not fit the engine protocol: {e}")
    sp = Splitter(tok, thinking, THINK_CLOSE, (HY_END, THINK_OPEN))
    return await run_chat_completion(
        request, body, eng, eid, q, sp,
        rid=rid, created=created, model_id=args.model_id, max_new=max_new,
        stop_strs=stop_strs, parse_tools_fn=lambda: parse_tools(sp.content, use_tools),
        use_tools=use_tools, tool_start=TOOL_START, tool_strip_re=CALLS_RE, tool_lookback=24,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
