# Two conversations that share a system prompt, interleaved turn by turn (parallel sub-agents).
#   slotbench.py <port> [out.json]
# Prints per request: prompt tokens, tokens reused from cache, prefill ms, request wall seconds.
import json, os, sys, time, urllib.request

port = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else None
U = f"http://127.0.0.1:{port}/v1/chat/completions"
here = os.path.dirname(os.path.abspath(__file__))
rd = lambda p: open(os.path.join(here, "..", p)).read()
SYS = ("You are a coding agent working in a repository. Be concise.\n\nRepository notes:\n\n```\n" + rd("engine/model.h") + "\n```\n")
SIZE = int(os.environ.get("SIZE", "24000"))  # characters of file context per conversation
if SIZE <= 24000:
    ctx = {"A": rd("engine/step.inc")[:SIZE], "B": rd("engine/model.mm")[:SIZE]}
else:  # long contexts: disjoint halves of the whole source tree
    import glob
    corpus = "".join(f"\n// ==== {os.path.relpath(f, os.path.join(here, '..'))} ====\n" + open(f).read()
                     for pat in ("engine/*.inc", "engine/*.mm", "engine/kernels/*.metal", "bench/*.mm", "convert/*.py", "server/*.py", "tests/*")
                     for f in sorted(glob.glob(os.path.join(here, "..", pat))))
    ctx = {"A": corpus[:SIZE], "B": corpus[len(corpus) - SIZE:]}
convs = {
    "A": [{"role": "system", "content": SYS}, {"role": "user", "content": "Here are source files:\n\n```\n" + ctx["A"] + "\n```\n\nIn one sentence, what does encode_layer do?"}],
    "B": [{"role": "system", "content": SYS}, {"role": "user", "content": "Here are source files:\n\n```\n" + ctx["B"] + "\n```\n\nIn one sentence, what does pick_split do?"}],
}
follow = {"A": ["And what does encode_gdn do, in one sentence?", "Name the function that saves checkpoints."],
          "B": ["And what does hc_block do, in one sentence?", "Name the struct that holds matmul arguments."]}
res, tot_wall = {}, 0.0
for turn in range(3):
    for name in ("A", "B"):
        msgs = convs[name]
        if turn: msgs.append({"role": "user", "content": follow[name][turn - 1]})
        body = {"messages": msgs, "max_tokens": 80, "temperature": 0, "seed": 1, "chat_template_kwargs": {"enable_thinking": False}}
        t0 = time.time()
        r = json.loads(urllib.request.urlopen(urllib.request.Request(U, json.dumps(body).encode(), {"content-type": "application/json"}), timeout=900).read())
        wall = time.time() - t0
        tot_wall += wall
        u, c = r["usage"], r["choices"][0]["message"]["content"]
        o = u["opus"]
        print(f"{name}{turn + 1}: prompt {u['prompt_tokens']:6d} reused {o['reused_prompt_tokens']:6d} prefill {o['prefill_ms']:7.0f} ms  request {wall:5.2f} s")
        msgs.append({"role": "assistant", "content": c})
        res[f"{name}{turn + 1}"] = c
print(f"TOTAL request wall {tot_wall:.1f} s")
if out_path:
    json.dump(res, open(out_path, "w"), indent=1, sort_keys=True)
