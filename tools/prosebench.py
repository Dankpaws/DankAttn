# Prose-heavy requests against an opus server (explanations, essays, reasoning with thinking on).
#   prosebench.py <port> [out.json]      TEMP=0.7 for sampled runs
# Prints decode tok/s per case and the token-weighted total; writes texts for A/B diffing.
import json, os, sys, urllib.request

port = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else None
U = f"http://127.0.0.1:{port}/v1/chat/completions"
CASES = [
    ("history", "Explain the causes and consequences of the fall of the Western Roman Empire in about 300 words.", False, 450),
    ("howto", "How do I diagnose a Linux server that becomes unresponsive under memory pressure? Explain the steps in prose.", False, 450),
    ("science", "Why is the sky blue during the day but red at sunset? Give a careful physical explanation.", False, 400),
    ("review", "Write a balanced review of the trade-offs between microservices and a monolith for a ten-person startup.", False, 450),
    ("story", "Write a short story about a lighthouse keeper who discovers a message in a bottle.", False, 450),
    ("think_math", "A train leaves at 9:40 travelling 84 km/h; a second leaves the same station at 10:10 travelling 112 km/h on the same track. When does the second catch the first? Explain.", True, 500),
    ("think_design", "Design a rate limiter for an API gateway that must be fair across tenants. Think it through, then describe the design.", True, 600),
]
res, tot_tok, tot_s = {}, 0, 0.0
for name, prompt, think, max_tokens in CASES:
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": float(os.environ.get("TEMP", "0")), "seed": 20261110,
            "chat_template_kwargs": {"enable_thinking": think}}
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        U, json.dumps(body).encode(), {"content-type": "application/json"}), timeout=900).read())
    u, m = r["usage"], r["choices"][0]["message"]
    o = u["opus"]
    gen, dec_s = u["completion_tokens"], (o["wall_ms"] - o["prefill_ms"]) / 1e3
    tot_tok += gen; tot_s += dec_s
    print(f"{name:13s} gen {gen:4d} rounds {o['spec_rounds']:4d} tok/round {gen / max(1, o['spec_rounds']):5.2f}  decode {gen / dec_s:6.1f} tok/s")
    res[name] = [m.get("content"), m.get("reasoning_content")]
print(f"TOTAL         gen {tot_tok:4d} decode {tot_tok / tot_s:6.1f} tok/s")
if out_path:
    json.dump(res, open(out_path, "w"), indent=1, sort_keys=True)
