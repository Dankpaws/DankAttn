"""Draft acceptance summary from an OPUS_CONFLOG=1 OPUS_TRACE=1 spec run's stderr."""
import sys, re, collections

t = collections.Counter(); g = collections.Counter()
rows = []; acc = []; K = []
pat = collections.Counter(); firstrej = collections.Counter(); streak = 0
for l in open(sys.argv[1], errors="replace"):
    if l.startswith("CONF "):
        f = l.split()
        i = int(f[1]); t[i] += 1; g[i] += int(f[3])
    elif l.startswith("TRACE round"):
        m = re.search(r"K (\d+) rows (\d+) accepted (\d+) chain (\S*)", l)
        if m:
            K.append(int(m[1])); rows.append(int(m[2])); a = int(m[3]); acc.append(a); ch = m[4]
            if a < len(ch): firstrej[ch[a]] += 1
            for c in ch[:a]: pat[c] += 1
    elif l.startswith("TRACE streak"):
        streak += 1
n = len(rows)
if not n:
    print("  no rounds parsed"); sys.exit(0)
full = sum(1 for a, r in zip(acc, rows) if a == r - 1)
print("  rounds %d (+%d streak chunks)  drafts/round %.2f  verify rows %.2f  accepted %.2f  all-accepted %.0f%%  none-accepted %.0f%%" % (
    n, streak, sum(K) / n, sum(rows) / n, sum(acc) / n, 100 * full / n, 100 * sum(1 for a in acc if a == 0) / n))
print("  acceptance of draft i when reached: " + " ".join("d%d %.0f%%(%d)" % (i, 100 * g[i] / t[i], t[i]) for i in sorted(t)))
print("  accepted by source (m=MTP, L=lookup, b=both):", dict(pat), " rejected token came from:", dict(firstrej))
hist = collections.Counter(acc)
print("  accepted-per-round histogram: " + " ".join("%d:%d" % (k, hist[k]) for k in sorted(hist)))
