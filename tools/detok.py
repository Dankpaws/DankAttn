# Print token ids (space-separated on stdin, or a uint32 file) as text.
import sys, numpy as np
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
ids = [int(x) for x in sys.stdin.read().split()] if len(sys.argv) < 3 else np.fromfile(sys.argv[2], dtype=np.uint32).tolist()
print(tok.decode(ids, skip_special_tokens=False))
