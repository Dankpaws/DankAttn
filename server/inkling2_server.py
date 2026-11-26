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
    BadRequest, ClientGone, EngineProc, PayloadTooLarge, PrefixSplicer, Unavailable, Watch,
    add_security_args, api_error, bad_request, error_response, install_security_middleware,
    json_body, make_jinja_env, normalize_messages, num, payload_too_large,
    register_common_routes, safe_template_kwargs, unavailable,
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
ap.add_argument("--power", choices=("auto", "performance", "eco"), default=os.environ.get("INK_POWER", "auto"),
                help="power profile: auto (eco on battery/Low Power Mode, performance on AC), performance, or eco")
ap.add_argument("--no-eco-draft", action="store_true",
                help="keep speed-maximizing DSpark draft lengths even in eco/battery mode (INK_ECO_DRAFT=0)")
ap.add_argument("--idle-unload", type=float, default=float(os.environ.get("DANKATTN_IDLE_UNLOAD", "0")),
                help="unload engine after N seconds idle (saves Lane 0 to pcache, wakes on next request; 0: off)")
add_security_args(ap)
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
EFFORTS = ("off", "none", "minimal", "low", "medium", "high", "xhigh", "max")

_eng_env = dict(os.environ, INK_CAP=str(args.cap), INK_LANES=str(args.lanes), INK_POWER=args.power)
if args.no_eco_draft:
    _eng_env["INK_ECO_DRAFT"] = "0"

eng = EngineProc(
    [args.engine, args.pack, "serve"],
    env=_eng_env,
    default_stops=STOPS,
    think_end=END_MSG,
    think_force=[END_MSG],
    idle_unload_s=args.idle_unload,
)
splicer = PrefixSplicer(tok, max_held=16)


def render(body):
    kw = safe_template_kwargs(body)
    eff = body.get("reasoning_effort")
    if eff is not None and "reasoning_effort" not in kw:
        if eff not in EFFORTS:
            raise BadRequest(f"reasoning_effort must be one of {', '.join(EFFORTS)}")
        kw["reasoning_effort"] = eff
    et_kw = kw.pop("enable_thinking", None)
    et_top = body.get("enable_thinking", None)
    et = et_kw if et_kw is not None else et_top
    disable_thinking = (et is False) or (kw.get("reasoning_effort") == "off")
    if kw.get("reasoning_effort") == "off":
        kw["reasoning_effort"] = "none"
    elif et is False and "reasoning_effort" not in kw:
        kw["reasoning_effort"] = "none"
    rendered = template.render(messages=normalize_messages(body), tools=body.get("tools"), **kw)
    if disable_thinking:
        # Inkling's template sets effort=0 for "none" but the model still emits a short thinking
        # block unless we close thinking (with tools) or open <|content_text|> directly.
        rendered += "<|message_model|><|content_thinking|><|end_message|>" if body.get("tools") else "<|message_model|><|content_text|>"
    return rendered


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
install_security_middleware(app, allow_hosts=args.allow_host, api_key=args.api_key, max_body_mb=args.max_body_mb)
register_common_routes(app, eng, args.model_id, args.cap, lambda: args.lanes, render, tok)

_FPR_ALIAS_RE = re.compile(r":(\d+)x-(select|genselect|synth|fast)$")
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


_FAST_ECHO_RE = re.compile(
    r"^\s*(?:(?:Write\s+ONLY\s+Section\s+\d+[^\n]*|Start\s+directly\s+with[^\n]*|Do\s+not\s+write\s+any\s+other\s+sections\.?|"
    r"Here\s+(?:is|are)\s+Section\s+\d+[^\n]*|Full\s+response\s+outline[^\n]*)\n+)+",
    re.I,
)


def _strip_fast_echo(txt: str) -> str:
    return _FAST_ECHO_RE.sub("", txt).lstrip("\n")


def parse_fpr_opts(body):
    """Parse optional Fork-Prune-Resolve / Parallel-Fill settings from model alias (:4x-select|:4x-genselect|:4x-synth|:4x-fast) or extra_body."""
    model_name = body.get("model") or args.model_id
    fpr_lanes, fpr_mode = 0, None
    m = _FPR_ALIAS_RE.search(model_name)
    if m:
        fpr_lanes = int(m.group(1))
        fpr_mode = m.group(2)
    if body.get("dankattn_lanes") is not None:
        fpr_lanes = num(body, "dankattn_lanes", int, 0, 1, 16)
    if body.get("dankattn_mode") is not None:
        fpr_mode = str(body["dankattn_mode"])
        if fpr_mode not in ("select", "genselect", "synth", "fast"):
            raise BadRequest("dankattn_mode must be one of select, genselect, synth, fast")
    if fpr_lanes < 2 or not fpr_mode:
        return None
    fpr_lanes = min(fpr_lanes, args.lanes)
    if fpr_lanes < 2:
        return None
    default_q = max(2, fpr_lanes - 1) if fpr_lanes >= 4 else fpr_lanes
    quorum = num(body, "dankattn_quorum", int, default_q, 0, fpr_lanes)
    prune = num(body, "dankattn_prune", int, 1, 0, 2)
    return {"lanes": fpr_lanes, "mode": fpr_mode, "quorum": quorum, "prune": prune, "model": model_name}


def _parse_cand(tok_ids):
    p = InkParser()
    for t in tok_ids:
        p.push(t)
    p._close()
    return p


def _cand_summary(p, limit=1500):
    parts = []
    if p.content.strip():
        parts.append(p.content.strip()[:limit])
    if p.calls:
        parts.append("Tool calls: " + json.dumps([c["function"] for c in p.calls], ensure_ascii=False)[:limit])
    return "\n".join(parts) or "(empty / truncated)"


def _cand_sig(p):
    return (p.content.strip(), json.dumps([c["function"] for c in p.calls], sort_keys=True, ensure_ascii=False))


async def _collect_fpr_candidates(ids, max_new, temp, top_p, top_k, seed, budget, min_p, extra_stops, fpr, request,
                                  mode_code=1, fpr_suffixes=None):
    """Run C++ FPR in candidate/parallel-fill mode (mode_code=1 or 2): zero-copy Split-KV fork, batched DSpark.
    Rewinds Lane 0 to len(ids) on completion so any follow-up pass reuses 100% of the prompt prefix.
    """
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    opts = mode_code | ((0 if mode_code == 2 else fpr["prune"]) << 4)
    quorum = fpr["lanes"] if mode_code == 2 else fpr["quorum"]
    cb = eng.metered(lambda item: loop.call_soon_threadsafe(q.put_nowait, item))
    try:
        eid = eng.submit(ids, max_new, not args.no_spec, temp, top_p, top_k, seed, cb,
                         group=(fpr["lanes"], quorum, opts),
                         think_budget=budget, min_p=min_p, extra_stops=extra_stops, fpr_suffixes=fpr_suffixes)
    except Unavailable:
        cb({"error": "unavailable"})
        raise
    except struct.error as e:
        cb({"error": str(e)})
        raise BadRequest(f"request does not fit the engine protocol: {e}")
    watch = Watch(q, request)
    cands = []
    cur = None
    stats = None
    try:
        while True:
            item = await watch.get()
            if isinstance(item, dict):
                if "error" in item:
                    raise Unavailable(item["error"])
                stats = item
                break
            if isinstance(item, tuple) and item[0] == "cand_sep":
                cur = {"conf": item[1], "length": bool(item[2] & 1), "toks": []}
                cands.append(cur)
            elif isinstance(item, int) and cur is not None:
                cur["toks"].append(item)
    except (ClientGone, asyncio.CancelledError):
        eng.cancel(eid)
        raise
    return cands, stats


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
        fpr = parse_fpr_opts(body)
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
    try:
        await asyncio.to_thread(eng.ensure_awake)
    except Unavailable as e:
        return unavailable(str(e))
    extra_stops = []
    for s in stop_strs:
        sids = tok.encode(s, add_special_tokens=False).ids
        if len(sids) == 1:
            extra_stops.append(sids[0])
    max_stop_len = max((len(s) for s in stop_strs), default=0)
    budget = tb if tb is not None else args.thinking_budget
    rid, created = "chatcmpl-" + uuid.uuid4().hex[:16], int(time.time())
    resp_model = fpr["model"] if fpr else args.model_id

    # Lever 2: Parallel Section Fill (:4x-fast / :8x-fast) — fast JSON outline -> K parallel forked section fills -> stitch!
    if fpr and fpr["mode"] == "fast" and not body.get("tools") and max_new >= 512:
        t_fast0 = time.perf_counter()
        n_sec = fpr["lanes"]
        outline_suf = (
            f"<|end_message|><|message_user|><|content_text|>"
            f"Split the response above into {n_sec} independent, non-overlapping sections. "
            f"If the request is a greeting, simple question, or short/single-paragraph answer, return ONLY 1 section. "
            f"Keep each title <=5 words and each spec <=10 words. "
            f"Reply ONLY with compact JSON: {{\"sections\":[{{\"title\":\"...\",\"spec\":\"...\"}}]}}."
            f"<|end_message|><|message_model|><|content_text|>{{\"sections\":["
        )
        outline_ids = ids + tok.encode(outline_suf, add_special_tokens=False).ids
        loop = asyncio.get_running_loop()
        oq: asyncio.Queue = asyncio.Queue()
        ocb = eng.metered(lambda item: loop.call_soon_threadsafe(oq.put_nowait, item))
        try:
            oeid = eng.submit(outline_ids, 1024, not args.no_spec, 0.0, 1.0, 0, seed, ocb,
                              think_budget=0, min_p=min_p, extra_stops=[END_MSG] + extra_stops)
        except Unavailable as e:
            ocb({"error": str(e)})
            return unavailable(str(e))
        except struct.error as e:
            ocb({"error": str(e)})
            return bad_request(f"request does not fit the engine protocol: {e}")
        owatch = Watch(oq, request)
        otoks, st_out = [], {}
        try:
            while True:
                it = await owatch.get()
                if isinstance(it, dict):
                    if "error" in it:
                        return error_response(it["error"])
                    st_out = it
                    break
                if isinstance(it, int) and it not in (END_MSG, EOS):
                    otoks.append(it)
        except ClientGone:
            eng.cancel(oeid)
            return Response(status_code=499)
        except asyncio.CancelledError:
            eng.cancel(oeid)
            raise
        outline_ms = (time.perf_counter() - t_fast0) * 1000.0
        raw_json = '{"sections":[' + tok.decode(otoks, skip_special_tokens=True).strip()
        sections = []
        try:
            m_obj = re.search(r"\{.*\}", raw_json, re.S)
            obj = json.loads(m_obj.group(0) if m_obj else raw_json)
            if isinstance(obj, dict) and isinstance(obj.get("sections"), list):
                sections = [s for s in obj["sections"] if isinstance(s, dict) and s.get("title")][:n_sec]
        except Exception:
            sections = []
        if len(sections) < 2:
            for t_m, s_m in re.findall(r'"title"\s*:\s*"((?:[^"\\]|\\.)+)"\s*,\s*"spec"\s*:\s*"((?:[^"\\]|\\.)+)"', raw_json):
                sections.append({"title": t_m, "spec": s_m})
            sections = sections[:n_sec]

        if len(sections) >= 2:
            K = len(sections)
            outline_str = json.dumps({"sections": sections}, ensure_ascii=False)
            shared_ctx = (
                f"<|end_message|><|message_user|><|content_text|>"
                f"Full response outline (shared across parallel writers):\n{outline_str}\n"
            )
            shared_ids = ids + tok.encode(shared_ctx, add_special_tokens=False).ids
            suffixes = []
            for idx_s, sec in enumerate(sections):
                stxt = (
                    f"Write ONLY Section {idx_s+1} (\"{sec.get('title','')}\"): {sec.get('spec','')}\n"
                    f"Start directly with the section heading and complete content. Do not write any other sections."
                    f"<|end_message|><|message_model|><|content_text|>"
                )
                suffixes.append(tok.encode(stxt, add_special_tokens=False).ids)
            fpr_fill = dict(fpr, lanes=K, quorum=K)
            per_sec_max = max(256, max_new // K + 256)
            t_fill0 = time.perf_counter()
            if body.get("stream"):
                loop = asyncio.get_running_loop()
                fq: asyncio.Queue = asyncio.Queue()
                fcb = eng.metered(lambda item: loop.call_soon_threadsafe(fq.put_nowait, item))
                try:
                    feid = eng.submit(shared_ids, per_sec_max, not args.no_spec, temp, top_p, top_k, seed, fcb,
                                      group=(K, K, 2), think_budget=0, min_p=min_p,
                                      extra_stops=[END_MSG] + extra_stops, fpr_suffixes=suffixes)
                except Unavailable as e:
                    fcb({"error": str(e)})
                    return unavailable(str(e))
                except struct.error as e:
                    fcb({"error": str(e)})
                    return bad_request(f"request does not fit the engine protocol: {e}")

                async def gen_fast_live():
                    def ch(delta, finish=None, u=None):
                        d = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": resp_model,
                             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                        if u:
                            d["usage"] = u
                        return "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"
                    done_f = False
                    vis_toks = 0
                    sec_idx = -1
                    buf_toks, read_pos, pref_pos = [], 0, 0
                    sec_lead_done = False

                    def flush_sec_tail():
                        nonlocal sec_lead_done
                        if not buf_toks[read_pos:]:
                            return None
                        if not sec_lead_done:
                            sec_lead_done = True
                            cleaned = _strip_fast_echo(tok.decode(buf_toks, skip_special_tokens=True))
                            return ch({"content": cleaned}) if cleaned else None
                        tail = tok.decode(buf_toks[pref_pos:], skip_special_tokens=True)
                        prev = tok.decode(buf_toks[pref_pos:read_pos], skip_special_tokens=True)
                        return ch({"content": tail[len(prev):]}) if len(tail) > len(prev) else None

                    try:
                        yield ch({"role": "assistant", "content": ""})
                        while True:
                            it = await fq.get()
                            if isinstance(it, dict):
                                if "error" in it:
                                    yield "data: " + json.dumps({"error": api_error(it["error"])[1]}, ensure_ascii=False) + "\n\n"
                                    done_f = True
                                    break
                                msg = flush_sec_tail()
                                if msg:
                                    yield msg
                                fill_ms = (time.perf_counter() - t_fill0) * 1000.0
                                wall_ms = (time.perf_counter() - t_fast0) * 1000.0
                                usage = {
                                    "prompt_tokens": len(ids), "completion_tokens": vis_toks,
                                    "total_tokens": len(ids) + vis_toks,
                                    "completion_tokens_details": {"reasoning_tokens": 0},
                                    "dankattn": {
                                        "fpr_mode": "fast", "sections": K,
                                        "outline_ms": round(outline_ms, 1), "fill_ms": round(fill_ms, 1),
                                        "wall_ms": round(wall_ms, 1),
                                        "eff_tok_s": round(1000.0 * vis_toks / max(wall_ms, 1.0), 1),
                                        "fill_tok_s": round(1000.0 * vis_toks / max(fill_ms, 1.0), 1),
                                    },
                                }
                                yield ch({}, "stop", usage)
                                yield "data: [DONE]\n\n"
                                done_f = True
                                break
                            if isinstance(it, tuple) and it[0] == "cand_sep":
                                if sec_idx >= 0:
                                    msg = flush_sec_tail()
                                    if msg:
                                        yield msg
                                sec_idx += 1
                                if sec_idx > 0:
                                    yield ch({"content": "\n\n"})
                                buf_toks, read_pos, pref_pos = [], 0, 0
                                sec_lead_done = False
                            elif isinstance(it, int) and it not in (END_MSG, EOS):
                                vis_toks += 1
                                buf_toks.append(it)
                                if not sec_lead_done:
                                    lead = tok.decode(buf_toks, skip_special_tokens=True)
                                    if not lead.endswith("\ufffd") and ("\n" in lead or len(lead) >= 96):
                                        cleaned = _strip_fast_echo(lead)
                                        sec_lead_done = True
                                        read_pos = len(buf_toks)
                                        pref_pos = max(0, read_pos - 6)
                                        if cleaned:
                                            yield ch({"content": cleaned})
                                else:
                                    before = tok.decode(buf_toks[pref_pos:read_pos], skip_special_tokens=True)
                                    text = tok.decode(buf_toks[pref_pos:], skip_special_tokens=True)
                                    if not text.endswith("\ufffd") and len(text) > len(before):
                                        read_pos = len(buf_toks)
                                        pref_pos = max(0, read_pos - 6)
                                        yield ch({"content": text[len(before):]})
                    finally:
                        if not done_f:
                            eng.cancel(feid)
                return StreamingResponse(gen_fast_live(), media_type="text/event-stream")

            try:
                cands, st_fill = await _collect_fpr_candidates(
                    shared_ids, per_sec_max, temp, top_p, top_k, seed, 0, min_p, [END_MSG] + extra_stops,
                    fpr_fill, request, mode_code=2, fpr_suffixes=suffixes
                )
            except BadRequest as e:
                return bad_request(str(e))
            except Unavailable as e:
                return unavailable(str(e))
            except ClientGone:
                return Response(status_code=499)
            fill_ms = (time.perf_counter() - t_fill0) * 1000.0
            parts = []
            vis_toks = 0
            for c in cands:
                ctoks = [t for t in c["toks"] if t not in (END_MSG, EOS)]
                vis_toks += len(ctoks)
                txt = _strip_fast_echo(tok.decode(ctoks, skip_special_tokens=True)).strip()
                if txt:
                    parts.append(txt)
            final_text = "\n\n".join(parts)
            wall_ms = (time.perf_counter() - t_fast0) * 1000.0
            usage = {
                "prompt_tokens": len(ids), "completion_tokens": vis_toks,
                "total_tokens": len(ids) + vis_toks,
                "completion_tokens_details": {"reasoning_tokens": 0},
                "dankattn": {
                    "fpr_mode": "fast", "sections": K,
                    "outline_ms": round(outline_ms, 1), "fill_ms": round(fill_ms, 1),
                    "wall_ms": round(wall_ms, 1),
                    "eff_tok_s": round(1000.0 * vis_toks / max(wall_ms, 1.0), 1),
                    "fill_tok_s": round(1000.0 * vis_toks / max(fill_ms, 1.0), 1),
                },
            }
            return {"id": rid, "object": "chat.completion", "created": created, "model": resp_model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": final_text}, "finish_reason": "stop"}],
                    "usage": usage}
        fpr = None  # fallback to normal 1-lane decode when outline has <= 1 section

    # FPR genselect / synth: Stage 1 in C++ (Split-KV fork + DeepConf prune + quorum -> Lane 0 rewound to len(ids)),
    # Stage 2 either short-circuits on unanimous consensus or runs zero-prefill judge/synth on Lane 0!
    if fpr and fpr["mode"] in ("genselect", "synth"):
        t_fpr0 = time.perf_counter()
        try:
            cands, st1 = await _collect_fpr_candidates(
                ids, max_new, temp, top_p, top_k, seed, budget, min_p, extra_stops, fpr, request
            )
        except BadRequest as e:
            return bad_request(str(e))
        except Unavailable as e:
            return unavailable(str(e))
        except ClientGone:
            return Response(status_code=499)
        parsed = [_parse_cand(c["toks"]) for c in cands]
        valid_idx = [i for i, c in enumerate(cands) if not c["length"] and (parsed[i].content.strip() or parsed[i].calls)]
        if not valid_idx:
            valid_idx = list(range(len(cands)))

        # Unanimous short-circuit for genselect (0 ms judge call!)
        sigs = [_cand_sig(parsed[i]) for i in valid_idx]
        if fpr["mode"] == "genselect" and len(valid_idx) >= 1 and len(set(sigs)) == 1:
            win_i = max(valid_idx, key=lambda i: cands[i]["conf"])
            wp = parsed[win_i]
            wall_ms = (time.perf_counter() - t_fpr0) * 1000.0
            usage = {
                "prompt_tokens": st1["prompt"], "completion_tokens": len(cands[win_i]["toks"]),
                "total_tokens": st1["prompt"] + len(cands[win_i]["toks"]),
                "completion_tokens_details": {"reasoning_tokens": wp.n_reasoning},
                "dankattn": {"reused_prompt_tokens": st1["reused"], "prefill_ms": round(st1["prefill_ms"], 1),
                             "wall_ms": round(wall_ms, 1), "fpr_mode": "genselect_unanimous",
                             "fpr_lanes": fpr["lanes"], "fpr_cands": len(cands), "win_conf": round(cands[win_i]["conf"], 3)},
            }
            fin = "tool_calls" if wp.calls else ("length" if cands[win_i]["length"] else "stop")
            if body.get("stream"):
                async def gen_win():
                    def ch(delta, finish=None, u=None):
                        d = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": resp_model,
                             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                        if u:
                            d["usage"] = u
                        return "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"
                    yield ch({"role": "assistant", "content": ""})
                    if wp.reasoning:
                        yield ch({"reasoning_content": wp.reasoning})
                    if wp.content:
                        yield ch({"content": wp.content})
                    for idx_c, tc in enumerate(wp.calls):
                        yield ch({"tool_calls": [dict(tc, index=idx_c)]})
                    yield ch({}, fin, usage)
                    yield "data: [DONE]\n\n"
                return StreamingResponse(gen_win(), media_type="text/event-stream")
            msg = {"role": "assistant", "content": wp.content.strip() or None}
            if wp.reasoning:
                msg["reasoning_content"] = wp.reasoning
            if wp.calls:
                msg["tool_calls"] = wp.calls
            return {"id": rid, "object": "chat.completion", "created": created, "model": resp_model,
                    "choices": [{"index": 0, "message": msg, "finish_reason": fin}], "usage": usage}

        # Build zero-prefill suffix appended directly onto Lane 0's cached `ids`!
        cand_block = "\n\n".join(
            f"[Candidate {idx+1} | confidence={cands[i]['conf']:.2f}]\n{_cand_summary(parsed[i])}"
            for idx, i in enumerate(valid_idx)
        )
        if fpr["mode"] == "genselect":
            suffix_txt = (
                f"<|end_message|><|message_user|><|content_text|>"
                f"Review these {len(valid_idx)} independent candidate solutions to the request above. "
                f"Briefly verify which candidate is most accurate and complete, then output ONLY its number inside \\boxed{{N}} (1..{len(valid_idx)}).\n\n"
                f"{cand_block}<|end_message|><|message_model|>"
            )
            judge_ids = ids + tok.encode(suffix_txt, add_special_tokens=False).ids
            loop = asyncio.get_running_loop()
            jq: asyncio.Queue = asyncio.Queue()
            jcb = eng.metered(lambda item: loop.call_soon_threadsafe(jq.put_nowait, item))
            try:
                jeid = eng.submit(judge_ids, 512, not args.no_spec, 0.0, 1.0, 0, seed ^ 0x9E3779B9, jcb,
                                  think_budget=budget, min_p=min_p, extra_stops=extra_stops)
            except Unavailable as e:
                jcb({"error": str(e)})
                return unavailable(str(e))
            except struct.error as e:
                jcb({"error": str(e)})
                return bad_request(f"request does not fit the engine protocol: {e}")
            jps = InkParser()
            jwatch = Watch(jq, request)
            st2 = {}
            try:
                while True:
                    it = await jwatch.get()
                    if isinstance(it, dict):
                        if "error" in it:
                            return error_response(it["error"])
                        st2 = it
                        break
                    jps.push(it)
            except ClientGone:
                eng.cancel(jeid)
                return Response(status_code=499)
            except asyncio.CancelledError:
                eng.cancel(jeid)
                raise
            jps._close()
            bm = _BOXED_RE.findall(jps.content or jps.reasoning)
            pick = 0
            if bm:
                try:
                    pi = int(bm[-1].strip()) - 1
                    if 0 <= pi < len(valid_idx):
                        pick = pi
                except ValueError:
                    pass
            win_i = valid_idx[pick]
            wp = parsed[win_i]
            wall_ms = (time.perf_counter() - t_fpr0) * 1000.0
            usage = {
                "prompt_tokens": st1["prompt"], "completion_tokens": len(cands[win_i]["toks"]),
                "total_tokens": st1["prompt"] + len(cands[win_i]["toks"]),
                "completion_tokens_details": {"reasoning_tokens": wp.n_reasoning},
                "dankattn": {"reused_prompt_tokens": st1["reused"], "judge_reused": st2.get("reused", 0),
                             "prefill_ms": round(st1["prefill_ms"] + st2.get("prefill_ms", 0), 1),
                             "wall_ms": round(wall_ms, 1), "fpr_mode": "genselect",
                             "fpr_lanes": fpr["lanes"], "fpr_cands": len(cands), "fpr_pick": pick + 1,
                             "win_conf": round(cands[win_i]["conf"], 3)},
            }
            fin = "tool_calls" if wp.calls else ("length" if cands[win_i]["length"] else "stop")
            if body.get("stream"):
                async def gen_gs():
                    def ch(delta, finish=None, u=None):
                        d = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": resp_model,
                             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                        if u:
                            d["usage"] = u
                        return "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"
                    yield ch({"role": "assistant", "content": ""})
                    if wp.reasoning:
                        yield ch({"reasoning_content": wp.reasoning})
                    if wp.content:
                        yield ch({"content": wp.content})
                    for idx_c, tc in enumerate(wp.calls):
                        yield ch({"tool_calls": [dict(tc, index=idx_c)]})
                    yield ch({}, fin, usage)
                    yield "data: [DONE]\n\n"
                return StreamingResponse(gen_gs(), media_type="text/event-stream")
            msg = {"role": "assistant", "content": wp.content.strip() or None}
            if wp.reasoning:
                msg["reasoning_content"] = wp.reasoning
            if wp.calls:
                msg["tool_calls"] = wp.calls
            return {"id": rid, "object": "chat.completion", "created": created, "model": resp_model,
                    "choices": [{"index": 0, "message": msg, "finish_reason": fin}], "usage": usage}

        # fpr["mode"] == "synth": append candidate summaries onto Lane 0's cached `ids` and fall through to stream!
        suffix_txt = (
            f"<|end_message|><|message_user|><|content_text|>"
            f"Below are {len(valid_idx)} independent draft responses to the request above. "
            f"Cross-check them, resolve any discrepancies, and produce one clear, verified final response "
            f"(including any tool calls if appropriate).\n\n{cand_block}"
            f"<|end_message|><|message_model|>"
        )
        ids = ids + tok.encode(suffix_txt, add_special_tokens=False).ids
        fpr = None  # fall through to normal streaming decode on Lane 0 (reusing 100% of original prompt!)

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    held_ids = list(ids)

    def track(item):
        if isinstance(item, int):
            held_ids.append(item)
        elif isinstance(item, dict) and "error" not in item:
            splicer.hold(held_ids)
        loop.call_soon_threadsafe(q.put_nowait, item)

    cb = eng.metered(track)
    group_arg = (fpr["lanes"], fpr["quorum"], 0 | (fpr["prune"] << 4)) if (fpr and fpr["mode"] == "select") else None
    try:
        eid = eng.submit(ids, max_new, not args.no_spec, temp, top_p, top_k, seed, cb,
                         group=group_arg, think_budget=budget, min_p=min_p, extra_stops=extra_stops)
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
            d = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": resp_model,
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
    return {"id": rid, "object": "chat.completion", "created": created, "model": resp_model,
            "choices": [{"index": 0, "message": msg, "finish_reason": "stop" if stopped else finish_of(item)}],
            "usage": usage_of(item)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
