# Token ids of a chat prompt, for the engine CLI tests (spec / ptest / prof).
#   mkids.py <pack> <out.bin> rewrite|toolcall <source file> [lines]
import json, os, struct, sys
import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

pack, out, kind, src = sys.argv[1:5]
nlines = int(sys.argv[5]) if len(sys.argv) > 5 else 120
tok = Tokenizer.from_file(os.path.join(pack, "tokenizer.json"))
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, **k: json.dumps(x, indent=indent, ensure_ascii=ensure_ascii)
template = env.from_string(open(os.path.join(pack, "chat_template.jinja")).read())
part = "\n".join(open(src).read().split("\n")[:nlines])
name = os.path.basename(src)
if kind == "rewrite":
    msg = (f"Here is {name}:\n\n```\n{part}\n```\n\nRename the variable `lock` to `engine_lock` everywhere and print the "
           "complete updated file in one code block. No commentary.")
else:
    msg = (f"Here is {name}:\n\n```\n{part}\n```\n\nQuote the first 60 lines of the file exactly, then explain them briefly.")
text = template.render(messages=[{"role": "user", "content": msg}], tools=None, add_generation_prompt=True, enable_thinking=False)
ids = tok.encode(text, add_special_tokens=False).ids
open(out, "wb").write(struct.pack(f"<{len(ids)}I", *ids))
print(len(ids), "tokens ->", out)
