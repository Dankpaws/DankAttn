"""Golden logits for N independent chunks of an ids file in one pass over the checkpoint (llama-perplexity style:
chunk k is ids[k*C:(k+1)*C], its own sequence from position 0). Token-wise ops (hyper-connections, norms, MLP,
MoE) run on all chunks' rows at once, so each layer's weights are read once; attention and Engram run per chunk.
Same ops as hy4_golden.Hy4.forward.

  python3 golden_chunks.py <ckpt> <ids.bin> <C> <N> <outdir>    writes outdir/g<k>.npz (logits [C][V], ids)
"""
import os, sys, time
import numpy as np
import torch
import hy4_golden as G


def main():
    ckpt, idsp, C, N, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    os.makedirs(out, exist_ok=True)
    ids = G.load_ids(idsp)
    seqs = [ids[k * C:(k + 1) * C] for k in range(N)]
    assert all(len(s) == C for s in seqs), "ids file too short"
    m = G.Hy4(ckpt)
    W = m.W
    pos = torch.arange(C)
    emb = W.get("model.embed_tokens.weight", None)
    hs = [emb[s].float()[:, None, :].expand(C, m.HC, m.H).contiguous() for s in seqs]
    t0 = time.time()
    for i in range(m.nL):
        if (i + 1) in m.engram_ids:
            hs = [m.engram(i, h, s) for h, s in zip(hs, seqs)]
        h = torch.cat(hs)
        res = h
        post, a = m.hc_pre(h, f"model.layers.{i}.hc_attn_layer.hc_pre.")
        a = G.rms(a, W.get(f"model.layers.{i}.input_layernorm.weight"), m.eps)
        a = torch.cat([m.attention(i, a[k * C:(k + 1) * C], pos) for k in range(N)])
        h = post[:, :, None] * a[:, None, :] + res
        res = h
        post, x = m.hc_pre(h, f"model.layers.{i}.hc_mlp_layer.hc_pre.")
        x = G.rms(x, W.get(f"model.layers.{i}.post_attention_layernorm.weight"), m.eps)
        x = m.dense_mlp(i, x) if m.mlp_types[i] == "dense" else m.moe(i, x)
        h = post[:, :, None] * x[:, None, :] + res
        hs = list(h.split(C))
        m.dump.clear()
        print(f"layer {i} {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
    fn = W.get("model.hc_head.hc_head_fn")
    head = W.get("lm_head.weight")
    for k, h in enumerate(hs):
        flat = h.reshape(C, -1)
        mixes = (flat @ fn.t()) * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + m.eps)
        pre = torch.sigmoid(mixes * W.get("model.hc_head.hc_head_scale") + W.get("model.hc_head.hc_head_base")) + m.hc_eps
        y = G.rms((pre[:, :, None] * h).sum(1), W.get("model.norm.weight"), m.eps)
        np.savez(f"{out}/g{k:02d}.npz", logits=G.lin(y, head, "lm_head").numpy(), ids=seqs[k].numpy())
    print("ok", N, "chunks", file=sys.stderr)


if __name__ == "__main__":
    main()
