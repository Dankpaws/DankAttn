# Agentic-coding style requests against an opus server: outputs that quote the prompt heavily
# (file rewrites, edits, tool calls) next to ones that do not (explanation, fresh code).
#   agentbench.py <port> <source file to use as context> [out.json]
# Prints tok/s per case from the server's own usage block; writes the texts for A/B diffing.
import json, os, sys, time, urllib.request

port, src = sys.argv[1], open(sys.argv[2]).read()
out_path = sys.argv[3] if len(sys.argv) > 3 else None
U = f"http://127.0.0.1:{port}/v1/chat/completions"
lines = src.split("\n")
part = "\n".join(lines[:120])

TOOLS = [{"type": "function", "function": {
    "name": "edit_file", "description": "Replace an exact string in a file with a new string",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}},
        "required": ["path", "old_string", "new_string"]}}}]

CASES = [
    ("rewrite_file", [{"role": "user", "content":
        "Here is server/opus_server.py:\n\n```python\n" + part + "\n```\n\n"
        "Rename the variable `lock` to `engine_lock` everywhere and print the complete updated file in one python code block. No commentary."}], None, 1400),
    ("edit_tool_call", [{"role": "user", "content":
        "Here is server/opus_server.py:\n\n```python\n" + part + "\n```\n\n"
        "Use the edit_file tool to make `run_engine` raise RuntimeError('engine exited early') instead of RuntimeError('engine exited'). "
        "Include the whole `while True:` loop of that function in old_string and new_string."}], TOOLS, 700),
    ("explain", [{"role": "user", "content":
        "Here is server/opus_server.py:\n\n```python\n" + part + "\n```\n\n"
        "Explain in prose, without quoting code, how streaming and tool-call parsing interact in this server."}], None, 500),
    ("fresh_code", [{"role": "user", "content":
        "Write a Python function that parses an ISO-8601 duration string such as P3DT4H5M into a timedelta, with tests. Code only."}], None, 500),
]

res = {}
for name, msgs, tools, max_tokens in CASES:
    body = {"messages": msgs, "max_tokens": max_tokens, "temperature": float(os.environ.get("TEMP", "0")),
            "seed": 20261109, "chat_template_kwargs": {"enable_thinking": False}}
    if tools:
        body["tools"] = tools
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        U, json.dumps(body).encode(), {"content-type": "application/json"}), timeout=900).read())
    wall = time.time() - t0
    u, m = r["usage"], r["choices"][0]["message"]
    o = u["opus"]
    gen = u["completion_tokens"]
    dec_s = (o["wall_ms"] - o["prefill_ms"]) / 1e3
    print(f"{name:15s} prompt {u['prompt_tokens']:5d} gen {gen:5d} rounds {o['spec_rounds']:4d} "
          f"tok/round {gen / max(1, o['spec_rounds']):5.2f}  decode {gen / dec_s:6.1f} tok/s  (request wall {wall:5.1f}s)")
    res[name] = {"content": m.get("content"), "tool_calls": [c["function"] for c in m.get("tool_calls") or []]}
if out_path:
    json.dump(res, open(out_path, "w"), indent=1, sort_keys=True)
