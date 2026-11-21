# Agentic edit-style requests against an opus server: tool calls that copy spans of a file the
# model has just been shown (the common shape of coding-agent output).
#   editbench.py <port> [out.json]      TEMP=0.7 for sampled runs
import json, os, sys, urllib.request

port = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else None
U = f"http://127.0.0.1:{port}/v1/chat/completions"
here = os.path.dirname(os.path.abspath(__file__))
srv = open(os.path.join(here, "..", "server", "opus_server.py")).read().split("\n")
eng = open(os.path.join(here, "..", "engine", "main.mm")).read().split("\n")

def fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": {k: {"type": "string"} for k in props}, "required": req}}}
TOOLS = [
    fn("edit_file", "Replace an exact string in a file with a new string", ["path", "old_string", "new_string"], ["path", "old_string", "new_string"]),
    fn("write_file", "Write a complete file", ["path", "content"], ["path", "content"]),
    fn("bash", "Run a shell command", ["command"], ["command"]),
]
def show(name, lines, a, b):
    return f"Here is {name} (lines {a + 1}-{b}):\n\n```\n" + "\n".join(lines[a:b]) + "\n```\n\n"

CASES = [
    ("small_edit", show("server/opus_server.py", srv, 0, 140) +
     "Use edit_file to change the default port from its current value to 9000. Keep old_string to the one line that changes.", 300),
    ("func_edit", show("server/opus_server.py", srv, 0, 140) +
     "Use edit_file to add a `timeout: float = 30.0` keyword parameter to the first function defined in this file and use it "
     "where sensible. old_string must be the complete original function and new_string the complete updated function.", 900),
    ("block_move", show("engine/main.mm", eng, 0, 110) +
     "Use edit_file to wrap the body of the `score` mode branch in an `@autoreleasepool { ... }` block. old_string must be the "
     "complete branch and new_string the complete updated branch.", 900),
    ("write_copy", show("server/opus_server.py", srv, 0, 90) +
     "Use write_file to create server/opus_server_min.py containing only the imports and the first two top-level definitions "
     "from this file, unchanged.", 900),
    ("bash_call", "Use the bash tool to find every Python file under the current directory modified in the last two days and "
     "count their lines, sorted by line count.", 200),
    ("explain_then_edit", show("engine/main.mm", eng, 0, 110) +
     "In two sentences say what the `spec` mode does, then use edit_file to rename the variable `steps` to `nsteps` in the "
     "`prof` mode branch (old_string/new_string: the lines that change, each with one line of context).", 600),
]
res, tot_tok, tot_s = {}, 0, 0.0
for name, prompt, max_tokens in CASES:
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "tools": TOOLS,
            "temperature": float(os.environ.get("TEMP", "0")), "seed": 20261110,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        U, json.dumps(body).encode(), {"content-type": "application/json"}), timeout=900).read())
    u, m = r["usage"], r["choices"][0]["message"]
    o = u["opus"]
    gen, dec_s = u["completion_tokens"], (o["wall_ms"] - o["prefill_ms"]) / 1e3
    tot_tok += gen; tot_s += dec_s
    calls = [c["function"]["name"] for c in m.get("tool_calls") or []]
    print(f"{name:18s} prompt {u['prompt_tokens']:5d} gen {gen:4d} rounds {o['spec_rounds']:4d} tok/round {gen / max(1, o['spec_rounds']):5.2f}  "
          f"decode {gen / dec_s:6.1f} tok/s  calls {calls}")
    res[name] = {"content": m.get("content"), "tool_calls": [c["function"] for c in m.get("tool_calls") or []]}
print(f"TOTAL              gen {tot_tok:4d} decode {tot_tok / tot_s:6.1f} tok/s")
if out_path:
    json.dump(res, open(out_path, "w"), indent=1, sort_keys=True)
