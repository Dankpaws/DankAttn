#!/usr/bin/env python3
"""Stand-in for `opus <pack> serve <cap>` speaking the front end's stdin/stdout protocol, for testing
opus_server.py's error handling without a GPU. Behaviour by request:
  prompt + 8 > cap        -> refused like the engine: "prompt of N tokens exceeds the context capacity of CAP"
  start_group flag        -> refused: "group 1: 1 of 2 members arrived"
  max_tokens == 7         -> two tokens, then an error record (engine failure mid-stream)
  max_tokens == 9         -> one token, then the process exits (engine crash: the front end restarts it)
  max_tokens == 3         -> three tokens, END with the length flag (reply cut at the capacity)
  otherwise               -> three tokens, then the END stats record
Sends the ready record first, as the engine does after its warm-up."""
import struct, sys

cap = int(sys.argv[3])
inp, out = sys.stdin.buffer, sys.stdout.buffer
TOKENS = [9707, 11, 1879]  # any ids; the front end only decodes them


def rd(fmt):
    n = struct.calcsize(fmt)
    b = inp.read(n)
    if len(b) < n:
        sys.exit(0)
    return struct.unpack(fmt, b)


def tok(rid, t):
    out.write(struct.pack("<II", rid, t))


def err(rid, msg):
    m = msg.encode()
    out.write(struct.pack("<III", rid, 0xFFFFFFFE, len(m)) + m)
    out.flush()


out.write(struct.pack("<II", 0, 0xFFFFFFFD))
out.flush()
while True:
    (magic,) = rd("<I")
    if magic == 0x5843504F:  # cancel
        rd("<I")
        continue
    rid, n = rd("<II")
    ids = rd(f"<{n}I") if n else ()
    max_new, flags, ns = rd("<III")
    if ns:
        rd(f"<{ns}I")
    rd("<ffIQ")
    if flags & 2:
        rd("<III")
    if flags & 4:
        _, _, nf = rd("<III")
        if nf:
            rd(f"<{nf}I")
    if n + 8 > cap:
        err(rid, f"prompt of {n} tokens exceeds the context capacity of {cap}")
    elif flags & 2:
        err(rid, "group 1: 1 of 2 members arrived")
    elif max_new == 9:
        tok(rid, TOKENS[0]); out.flush()
        sys.exit(3)
    elif max_new == 7:
        tok(rid, TOKENS[0]); tok(rid, TOKENS[1]); out.flush()
        err(rid, "engine exited")
    else:
        for t in TOKENS:
            tok(rid, t)
        out.write(struct.pack("<II6I2f", rid, 0xFFFFFFFF, 0, n, len(TOKENS), 1, 0, 1 if max_new == 3 else 0, 1.0, 2.0))
        out.flush()
