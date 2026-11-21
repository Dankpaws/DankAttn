# The prose, edit and agentic prompts of prosebench / editbench / agentbench sent to any
# OpenAI-compatible server (opus or oMLX), to compare drafters on identical requests.
#   draftbench.py <port> [model] [out.json]
# Prints tokens and wall tok/s per group; the drafter statistics come from each server's own log.
import importlib.util, io, json, os, sys, time, urllib.request, contextlib

port = sys.argv[1]
model = sys.argv[2] if len(sys.argv) > 2 else None
out_path = sys.argv[3] if len(sys.argv) > 3 else None
U = f"http://127.0.0.1:{port}/v1/chat/completions"
here = os.path.dirname(os.path.abspath(__file__))

def cases_of(name):
    """CASES (and TOOLS) of a sibling bench, read without running its requests."""
    src = open(os.path.join(here, name)).read()
    head = src[:src.index("\nres")]  # everything before the request loop
    g = {"__file__": os.path.join(here, name)}
    argv, sys.argv = sys.argv, [name, "0", os.path.join(here, "..", "server", "opus_server.py")]
    try:
        exec(compile(head, name, "exec"), g)
    finally:
        sys.argv = argv
    return g

groups = []
g = cases_of("prosebench.py")
groups.append(("prose", [(n, [{"role": "user", "content": p}], None, th, mt) for n, p, th, mt in g["CASES"]]))
g = cases_of("editbench.py")
groups.append(("edit", [(n, [{"role": "user", "content": p}], g["TOOLS"], False, mt) for n, p, mt in g["CASES"]]))
g = cases_of("agentbench.py")
groups.append(("agent", [(n, m, t, False, mt) for n, m, t, mt in g["CASES"]]))

res = {}
for gname, cases in groups:
    tok, secs = 0, 0.0
    for name, msgs, tools, think, max_tokens in cases:
        body = {"messages": msgs, "max_tokens": max_tokens, "temperature": float(os.environ.get("TEMP", "0")),
                "seed": 20261110, "chat_template_kwargs": {"enable_thinking": think}}
        if model: body["model"] = model
        if tools: body["tools"] = tools
        t0 = time.time()
        r = json.loads(urllib.request.urlopen(urllib.request.Request(
            U, json.dumps(body).encode(), {"content-type": "application/json"}), timeout=1800).read())
        dt = time.time() - t0
        u, m = r["usage"], r["choices"][0]["message"]
        tok += u["completion_tokens"]; secs += dt
        res[f"{gname}/{name}"] = {"content": m.get("content"), "reasoning": m.get("reasoning_content"),
                                  "tool_calls": [c["function"] for c in m.get("tool_calls") or []],
                                  "completion_tokens": u["completion_tokens"], "wall_s": round(dt, 2)}
        print(f"MARK {gname}/{name} tokens {u['completion_tokens']} wall {dt:.2f}s", flush=True)
    print(f"GROUP {gname}: {tok} tokens, {tok / secs:.1f} tok/s request wall (includes prefill)", flush=True)
if out_path:
    json.dump(res, open(out_path, "w"), indent=1, sort_keys=True)
