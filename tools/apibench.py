# Same requests against OpenAI-compatible engines: TTFT and decode tok/s from the stream.
# Usage: python3 tools/apibench.py <base_url> <model> <tokenizer.json> <docs_glob>
import json, sys, time, urllib.request, glob
from tokenizers import Tokenizer
url, model, tok_path, docs_glob = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
tok = Tokenizer.from_file(tok_path)
def doc(n):
    ids = []
    for f in sorted(glob.glob(docs_glob, recursive=True)):
        ids += tok.encode(open(f, errors='ignore').read()).ids
        if len(ids) > n: break
    return tok.decode(ids[:n])
cases = [("code", "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring. Then explain its complexity in two sentences.", 300),
         ("prose", "In three short bullet points, explain what a Mac Studio's unified memory means for running large language models.", 300),
         ("doc8k", doc(8000) + "\n\nSummarize the document above in five bullet points.", 200),
         ("doc30k", doc(30000) + "\n\nSummarize the document above in five bullet points.", 200)]
for name, q, mx in cases:
    body = {"model": model, "messages": [{"role": "user", "content": q}], "max_tokens": mx, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    t0 = time.time(); tf = None; n = 0; usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                if c.get("delta", {}).get("content") or c.get("delta", {}).get("reasoning_content"):
                    if tf is None: tf = time.time()
    t1 = time.time()
    ct = (usage or {}).get("completion_tokens", 0); pt = (usage or {}).get("prompt_tokens", 0)
    print(f"{name:7s} prompt {pt:6d}  TTFT {tf - t0:6.2f}s  gen {ct:4d} tok  decode {max(ct - 1, 1) / (t1 - tf):6.1f} tok/s", flush=True)
