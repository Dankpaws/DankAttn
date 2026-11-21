"""Pool-mode serving check against a live engine started with OPUS_POOL=1.
Usage: python3 pool_check.py [base_url]. Each phase prints PASS/FAIL; engine-side placement is read from
the engine log by the caller (lines "opus: lane k req ..." vs "opus: slot k req ...")."""
import http.client, json, sys, threading, time, urllib.parse

U = urllib.parse.urlparse(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18340")
fails = 0


def check(label, ok, detail=""):
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail and not ok else ""), flush=True)


def post(messages, max_tokens=200, temperature=0.0, seed=1, thinking=False, effort=None, budget=None):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "top_p": 0.95, "top_k": 20,
            "seed": seed, "chat_template_kwargs": {"enable_thinking": thinking}}
    if effort:
        body["reasoning_effort"] = effort
    if budget is not None:
        body["thinking_budget"] = budget
    c = http.client.HTTPConnection(U.hostname, U.port, timeout=900)
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
    r = json.loads(c.getresponse().read())
    t1 = time.time()
    if "error" in r:
        return {"error": r["error"], "t0": t0, "t1": t1}
    m = r["choices"][0]["message"]
    return {"text": (m.get("reasoning_content") or "") + "\x00" + (m.get("content") or ""), "content": m.get("content") or "",
            "usage": r["usage"], "t0": t0, "t1": t1}


def together(calls):
    out = [None] * len(calls)
    ts = [threading.Thread(target=lambda i=i, c=c: out.__setitem__(i, c())) for i, c in enumerate(calls)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


def user(s):
    return [{"role": "user", "content": s}]


PROMPTS = [
    (user("Write a Python function that merges two sorted lists. Code only."), dict()),
    (user("Explain in four sentences why the sky is blue."), dict(temperature=1.0, seed=11)),
    (user("List ten prime numbers greater than 1000, comma separated."), dict(temperature=0.7, seed=12)),
    (user("Translate to French: The meeting moved to Thursday because the room was booked."), dict()),
    (user("Give three risks of running a database without backups. Short bullets."), dict(temperature=1.0, seed=13)),
    (user("What is 17 * 23? Think it through."), dict(thinking=True, effort="low", temperature=1.0, seed=14, max_tokens=400)),
    (user("Is 391 prime? Think step by step."), dict(thinking=True, budget=40, temperature=0.0, max_tokens=300)),
]

# A. alone (slot, speculation) vs six at once (lanes): identical replies, greedy and sampled
alone = [post(m, **kw) for m, kw in PROMPTS]
check("A: alone runs answered", all("text" in a for a in alone), [a.get("error") for a in alone])
t0 = time.time()
tog = together([lambda m=m, kw=kw: post(m, **kw) for m, kw in PROMPTS])
wall = time.time() - t0
same = [a.get("text") == b.get("text") for a, b in zip(alone, tog)]
check("A: seven together equal seven alone, byte for byte (incl. a spent thinking budget)", all(same), same)
toks = sum(b["usage"]["completion_tokens"] for b in tog if "usage" in b)
print(f"      together: {toks} tokens in {wall:.1f} s = {toks / wall:.0f} tok/s aggregate", flush=True)

# B. second turns of four conversations at once: each lands on the lane holding its conversation
conv = [PROMPTS[i][0] + [{"role": "assistant", "content": tog[i]["content"]},
                         {"role": "user", "content": "Now say the same thing in one sentence."}] for i in range(4)]
b2 = together([lambda m=m: post(m) for m in conv])
reused = [b["usage"]["opus"]["reused_prompt_tokens"] if "usage" in b else -1 for b in b2]
prompt = [b["usage"]["prompt_tokens"] if "usage" in b else -1 for b in b2]
check("B: turn 2 resumes its lane (reused >= 80% of the prompt)", all(r >= 0.8 * p > 0 for r, p in zip(reused, prompt)),
      list(zip(reused, prompt)))
b2a = [post(m) for m in conv]
check("B: turn 2 together equals turn 2 alone", all(x.get("text") == y.get("text") for x, y in zip(b2, b2a)))

# C. a recently used long conversation keeps the slot: a short newcomer goes to a lane, the long one resumes
long_doc = "\n".join(f"Line {i}: the quick brown fox jumps over the lazy dog, item {i * 7 % 101}." for i in range(260))
lc = user("Here is a document:\n" + long_doc + "\nHow many lines does it have? Answer with a number.")
c1 = post(lc, max_tokens=30)
check("C: long conversation answered", "text" in c1, c1.get("error"))
small = post(user("Say hello in Spanish."), max_tokens=20)
c2 = post(lc + [{"role": "assistant", "content": c1.get("content", "")}, {"role": "user", "content": "And the last line's item?"}],
          max_tokens=30)
u2 = c2.get("usage", {})
check("C: long conversation resumed on the slot after the newcomer",
      u2.get("opus", {}).get("reused_prompt_tokens", 0) >= 0.9 * u2.get("prompt_tokens", 1), u2)
print(f"      newcomer reused {small.get('usage', {}).get('opus', {}).get('reused_prompt_tokens')}", flush=True)

# D. a request arriving mid-decode joins the running step
X = (user("Count from 1 to 120, one number per line."), dict(max_tokens=500))
Y = (user("Write a haiku about rain."), dict(max_tokens=60))
res = {}
tx = threading.Thread(target=lambda: res.__setitem__("x", post(X[0], **X[1])))
tx.start()
time.sleep(1.5)
res["y"] = post(Y[0], **Y[1])
tx.join()
x, y = res["x"], res["y"]
check("D: late request finished while the first was still decoding", y["t1"] < x["t1"], (y["t1"] - x["t0"], x["t1"] - x["t0"]))
xa, ya = post(X[0], **X[1]), post(Y[0], **Y[1])
check("D: both equal their alone runs", x.get("text") == xa.get("text") and y.get("text") == ya.get("text"))

# F. siblings on one long system prompt (subagents): later ones fork from the first lane's checkpoint,
#    including while that lane is decoding; replies equal their alone runs
SYS = {"role": "system", "content": "You are an exploration agent. Rules:\n" + "\n".join(
    f"Rule {i}: when inspecting module {i}, note its imports, side effects and error paths." for i in range(220))}
F = [[SYS, {"role": "user", "content": f"In one sentence, what does rule {k} ask for?"}] for k in (3, 77, 150)]
fa = [post(m, max_tokens=60) for m in F]
res = {}
th = [threading.Thread(target=lambda i=i: res.__setitem__(i, post(F[i], max_tokens=60))) for i in range(3)]
th[0].start(); time.sleep(0.05); th[1].start(); time.sleep(2.0); th[2].start()
for t in th:
    t.join()
check("F: siblings equal their alone runs", all(res[i].get("text") == fa[i].get("text") for i in range(3)))
print("      sibling reused:", [res[i]["usage"]["opus"]["reused_prompt_tokens"] for i in range(3)],
      "of", [res[i]["usage"]["prompt_tokens"] for i in range(3)], flush=True)

# E. eight at once, greedy, thinking off: aggregate throughput
E = [user(f"Write a 300-word story about a lighthouse keeper named {n}.") for n in
     ["Ada", "Bo", "Cy", "Di", "Ed", "Fay", "Gus", "Hal"]]
t0 = time.time()
er = together([lambda m=m: post(m, max_tokens=400) for m in E])
wall = time.time() - t0
toks = sum(e["usage"]["completion_tokens"] for e in er if "usage" in e)
check("E: eight answered", all("usage" in e for e in er))
print(f"      eight together: {toks} tokens in {wall:.1f} s = {toks / wall:.0f} tok/s aggregate", flush=True)
t0 = time.time()
one = post(E[0], max_tokens=400)
print(f"      one alone: {one['usage']['completion_tokens']} tokens in {time.time() - t0:.1f} s", flush=True)

# G. aggregate throughput at 2, 4 and 6 concurrent requests (greedy, thinking off, 300 tokens)
for n in (2, 4, 6):
    msgs = [user(f"Write Python code for a {w} data structure with tests.") for w in
            ["queue", "stack", "trie", "heap", "ring buffer", "LRU cache"][:n]]
    t0 = time.time()
    rr = together([lambda m=m: post(m, max_tokens=300) for m in msgs])
    toks = sum(r["usage"]["completion_tokens"] for r in rr if "usage" in r)
    print(f"      G {n} together: {toks / (time.time() - t0):.0f} tok/s aggregate", flush=True)

print("ALL PASS" if not fails else f"{fails} FAILED")
