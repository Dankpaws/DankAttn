# PLE n-gram hash constants (no torch), shared by tools/ngram_index.py and ref/golden.py.
import math
MASK64 = (1 << 64) - 1
def splitmix64(v):
    v = (v + 0x9E3779B97F4A7C15) & MASK64
    v = ((v ^ (v >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    v = ((v ^ (v >> 27)) * 0x94D049BB133111EB) & MASK64
    return (v ^ (v >> 31)) & MASK64
def is_prime(v):
    if v < 2: return False
    if v % 2 == 0: return v == 2
    for d in range(3, math.isqrt(v) + 1, 2):
        if v % d == 0: return False
    return True
def nth_prime_after(start, count):
    p = start
    for _ in range(count):
        p += 1
        while not is_prime(p): p += 1
    return p
def ngram_meta(tc, ple_index):
    heads = (tc["ngram_size"] - 1) * tc["heads_per_ngram"]
    sizes, offs, tot = [], [], 0
    for h in range(heads):
        s = nth_prime_after(tc["ngram_vocab_size_base"] - 1, ple_index * heads + h + 1)
        sizes.append(s); offs.append(tot); tot += s
    mmax = ((1 << 63) - 1) // max(tc["vocab_size"], 1)
    half = max(1, mmax // 2)
    base = tc.get("seed", 1234) + 10007 * ple_index
    mult = [2 * (splitmix64((base + 0x9E3779B97F4A7C15 * (i + 1)) & MASK64) % half) + 1 for i in range(tc["ngram_size"])]
    return sizes, offs, mult
