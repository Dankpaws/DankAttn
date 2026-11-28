"""Anthropic Messages API (/v1/messages & /v1/messages/count_tokens) adapter for DankAttn.

Registered automatically by `engine_frontend.register_common_routes` so every DankAttn server
(Hy4, Inkling2, ERNIE-5.5, Flash-Next) supports native Anthropic SDK / Claude Code clients
without any external proxy.
"""
from __future__ import annotations

import json
import uuid
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse


def _anthropic_err(status: int, err_type: str, msg: str):
    return JSONResponse({"type": "error", "error": {"type": err_type, "message": msg}}, status_code=status)


def _extract_system_text(sys_val) -> str:
    if isinstance(sys_val, str):
        return sys_val
    if isinstance(sys_val, list):
        parts = []
        for p in sys_val:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
        return "\n".join(parts)
    return ""


def anthropic_to_openai(body: dict, default_model: str) -> dict:
    """Translate an Anthropic /v1/messages request body into an OpenAI /v1/chat/completions body."""
    msgs = []
    sys_txt = _extract_system_text(body.get("system"))
    if sys_txt:
        msgs.append({"role": "system", "content": sys_txt})

    for m in body.get("messages") or []:
        role = m.get("role", "user")
        content = m.get("content")
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            msgs.append({"role": role, "content": ""})
            continue

        text_parts = []
        tool_calls = []
        tool_results = []
        for blk in content:
            if not isinstance(blk, dict):
                continue
            btype = blk.get("type")
            if btype == "text":
                text_parts.append(blk.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append({
                    "id": blk.get("id") or ("toolu_" + uuid.uuid4().hex[:12]),
                    "type": "function",
                    "function": {
                        "name": blk.get("name", ""),
                        "arguments": json.dumps(blk.get("input", {}), ensure_ascii=False),
                    },
                })
            elif btype == "tool_result":
                tr_content = blk.get("content", "")
                if isinstance(tr_content, list):
                    tr_content = "\n".join(
                        p.get("text", "") for p in tr_content if isinstance(p, dict) and p.get("type") == "text"
                    )
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": blk.get("tool_use_id", ""),
                    "content": str(tr_content),
                })
        if role == "assistant":
            am: dict = {"role": "assistant", "content": "".join(text_parts)}
            if tool_calls:
                am["tool_calls"] = tool_calls
            msgs.append(am)
        else:
            for tr in tool_results:
                msgs.append(tr)
            if text_parts or not tool_results:
                msgs.append({"role": role, "content": "".join(text_parts)})

    oai: dict = {
        "model": body.get("model") or default_model,
        "messages": msgs,
        "max_tokens": body.get("max_tokens", 4096),
        "stream": bool(body.get("stream", False)),
    }
    for k in ("temperature", "top_p", "top_k", "seed"):
        if k in body and body[k] is not None:
            oai[k] = body[k]
    if body.get("stop_sequences"):
        oai["stop"] = body["stop_sequences"]
    if body.get("tools"):
        oai["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for t in body["tools"]
            if isinstance(t, dict)
        ]
    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        if thinking.get("type") == "disabled":
            oai["enable_thinking"] = False
        elif thinking.get("type") == "enabled":
            oai["enable_thinking"] = True
            if isinstance(thinking.get("budget_tokens"), int):
                oai["thinking_budget"] = thinking["budget_tokens"]
    return oai


def _map_stop_reason(oai_finish: str | None) -> str:
    if oai_finish == "tool_calls":
        return "tool_use"
    if oai_finish == "length":
        return "max_tokens"
    return "end_turn"


def openai_to_anthropic(resp: dict, model_id: str) -> dict:
    """Convert a non-streaming OpenAI chat.completion dict into an Anthropic Message dict."""
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    blocks = []
    if msg.get("reasoning_content"):
        blocks.append({"type": "thinking", "thinking": msg["reasoning_content"]})
    if msg.get("content"):
        blocks.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments", "{}")
        try:
            inp = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except Exception:
            inp = {}
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or ("toolu_" + uuid.uuid4().hex[:12]),
            "name": fn.get("name", ""),
            "input": inp if isinstance(inp, dict) else {},
        })
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    usage = resp.get("usage") or {}
    return {
        "id": "msg_" + uuid.uuid4().hex[:20],
        "type": "message",
        "role": "assistant",
        "model": resp.get("model") or model_id,
        "content": blocks,
        "stop_reason": _map_stop_reason(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def register_anthropic_routes(app, model_id: str, render_fn, tok, json_body_fn, payload_too_large_fn, PayloadTooLargeCls):
    """Attach /v1/messages and /v1/messages/count_tokens to a DankAttn FastAPI app."""

    @app.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens(request: Request):
        try:
            body = await json_body_fn(request)
        except PayloadTooLargeCls as e:
            return payload_too_large_fn(str(e))
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            return _anthropic_err(400, "invalid_request_error", "messages must be a list")
        oai = anthropic_to_openai(body, model_id)
        try:
            ids = tok.encode(render_fn(oai), add_special_tokens=False).ids
        except Exception as e:
            return _anthropic_err(400, "invalid_request_error", str(e))
        return {"input_tokens": len(ids)}

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request):
        try:
            body = await json_body_fn(request)
        except PayloadTooLargeCls as e:
            return payload_too_large_fn(str(e))
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
            return _anthropic_err(400, "invalid_request_error", "messages must be a non-empty list")

        # Locate the app's own /v1/chat/completions endpoint handler
        chat_endpoint = None
        for route in app.routes:
            if getattr(route, "path", None) == "/v1/chat/completions" and "POST" in getattr(route, "methods", set()):
                chat_endpoint = route.endpoint
                break
        if chat_endpoint is None:
            return _anthropic_err(500, "api_error", "chat completions endpoint not registered")

        oai_body = anthropic_to_openai(body, model_id)
        oai_bytes = json.dumps(oai_body, ensure_ascii=False).encode("utf-8")
        sent = [False]

        async def _recv():
            if not sent[0]:
                sent[0] = True
                return {"type": "http.request", "body": oai_bytes, "more_body": False}
            return {"type": "http.disconnect"}

        sub_req = Request(request.scope, receive=_recv)
        res = await chat_endpoint(sub_req)

        if isinstance(res, JSONResponse):
            return res
        if isinstance(res, dict):
            return openai_to_anthropic(res, model_id)
        if not isinstance(res, StreamingResponse):
            return res

        msg_id = "msg_" + uuid.uuid4().hex[:20]

        async def _translate_sse():
            started = False
            blk_idx = -1
            cur_kind = None  # "thinking" | "text"
            stop_reason = "end_turn"
            out_tokens = 0
            in_tokens = 0

            async for raw in res.body_iterator:
                chunk_s = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
                for line in chunk_s.splitlines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        continue
                    try:
                        ev = json.loads(payload)
                    except Exception:
                        continue
                    if "error" in ev:
                        yield _sse("error", {"type": "error", "error": {"type": "api_error", "message": str(ev["error"])}})
                        return
                    if not started:
                        started = True
                        yield _sse("message_start", {
                            "type": "message_start",
                            "message": {
                                "id": msg_id, "type": "message", "role": "assistant",
                                "model": ev.get("model") or model_id, "content": [],
                                "stop_reason": None, "stop_sequence": None,
                                "usage": {"input_tokens": 0, "output_tokens": 0},
                            },
                        })
                    if ev.get("usage"):
                        in_tokens = ev["usage"].get("prompt_tokens", in_tokens)
                        out_tokens = ev["usage"].get("completion_tokens", out_tokens)
                    ch = (ev.get("choices") or [{}])[0]
                    delta = ch.get("delta") or {}
                    if delta.get("reasoning_content"):
                        if cur_kind != "thinking":
                            if cur_kind is not None:
                                yield _sse("content_block_stop", {"type": "content_block_stop", "index": blk_idx})
                            blk_idx += 1
                            cur_kind = "thinking"
                            yield _sse("content_block_start", {
                                "type": "content_block_start", "index": blk_idx,
                                "content_block": {"type": "thinking", "thinking": ""},
                            })
                        yield _sse("content_block_delta", {
                            "type": "content_block_delta", "index": blk_idx,
                            "delta": {"type": "thinking_delta", "thinking": delta["reasoning_content"]},
                        })
                    if delta.get("content"):
                        if cur_kind != "text":
                            if cur_kind is not None:
                                yield _sse("content_block_stop", {"type": "content_block_stop", "index": blk_idx})
                            blk_idx += 1
                            cur_kind = "text"
                            yield _sse("content_block_start", {
                                "type": "content_block_start", "index": blk_idx,
                                "content_block": {"type": "text", "text": ""},
                            })
                        yield _sse("content_block_delta", {
                            "type": "content_block_delta", "index": blk_idx,
                            "delta": {"type": "text_delta", "text": delta["content"]},
                        })
                    if delta.get("tool_calls"):
                        if cur_kind is not None:
                            yield _sse("content_block_stop", {"type": "content_block_stop", "index": blk_idx})
                            cur_kind = None
                        for tc in delta["tool_calls"]:
                            fn = tc.get("function") or {}
                            blk_idx += 1
                            yield _sse("content_block_start", {
                                "type": "content_block_start", "index": blk_idx,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": tc.get("id") or ("toolu_" + uuid.uuid4().hex[:12]),
                                    "name": fn.get("name", ""),
                                    "input": {},
                                },
                            })
                            yield _sse("content_block_delta", {
                                "type": "content_block_delta", "index": blk_idx,
                                "delta": {"type": "input_json_delta", "partial_json": fn.get("arguments", "{}")},
                            })
                            yield _sse("content_block_stop", {"type": "content_block_stop", "index": blk_idx})
                    if ch.get("finish_reason"):
                        stop_reason = _map_stop_reason(ch["finish_reason"])

            if cur_kind is not None:
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": blk_idx})
            yield _sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens},
            })
            yield _sse("message_stop", {"type": "message_stop"})

        return StreamingResponse(_translate_sse(), media_type="text/event-stream")
