"""BF16 golden reference for Hy4-flash (hy_v4_flash text tower), f32 on CPU, one layer at a time.

Follows transformers PR #50903 (modeling_hy_v4_flash.py @ c2ec850) op for op, text path only,
no cache: a prompt is one sequence. Every op runs in f32; weights are read from the BF16
checkpoint and widened. This is the fidelity oracle opus-hy4 packs are measured against and
the source of the per-channel activation statistics the converter weights its quantization by.

  python3 hy4_golden.py <ckpt> <ids.npy|ids.bin> <out.npz> [--stats stats.pt] [--keep N] [--dump]
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count())


class Stats:
    """sum x^2 per input channel of every linear (keys = pack tensor names); experts per expert."""

    def __init__(self):
        self.s, self.n = {}, {}

    def add(self, key, x):
        if STATS is None:
            return
        x = x.reshape(-1, x.shape[-1]).double()
        if key not in self.s:
            self.s[key] = torch.zeros(x.shape[-1], dtype=torch.float64)
            self.n[key] = 0
        self.s[key] += (x * x).sum(0)
        self.n[key] += x.shape[0]

    def add_expert(self, key, e, x, E):
        if STATS is None:
            return
        if key not in self.s:
            self.s[key] = torch.zeros(E, x.shape[-1], dtype=torch.float64)
            self.n[key] = torch.zeros(E, dtype=torch.int64)
        self.s[key][e] += (x.double() ** 2).sum(0)
        self.n[key][e] += x.shape[0]


STATS = None


class Weights:
    def __init__(self, root):
        self.root = root
        self.map = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
        self.handles = {}

    def _h(self, f):
        if f not in self.handles:
            self.handles[f] = safe_open(os.path.join(self.root, f), framework="pt")
        return self.handles[f]

    def has(self, name):
        return name in self.map

    def get(self, name, dtype=torch.float32):
        t = self._h(self.map[name]).get_tensor(name)
        return t if dtype is None else t.to(dtype)

    def rows(self, name, idx):
        """rows idx of a 2-D tensor (or experts idx of a 3-D one) without reading the rest"""
        sl = self._h(self.map[name]).get_slice(name)
        return torch.stack([sl[int(i)] for i in idx]).float()


def rms(x, w, eps):
    x = x.float()
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y if w is None else y * w


def lin(x, w, key=None):
    if key:
        STATS and STATS.add(key, x)
    return x @ w.t()


def rope_tables(pos, dim, base):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    f = pos.double()[:, None] * inv[None, :]
    cos = f.cos().float().repeat_interleave(2, -1)  # pair (2j, 2j+1) turns by inv[j]
    sin = f.sin().float().repeat_interleave(2, -1)
    return cos, sin


def rot_half(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x, cos, sin):
    # x [..., T, d] with cos/sin [T, d] broadcast over leading dims
    return x * cos + rot_half(x) * sin


class Hy4:
    def __init__(self, root):
        self.W = Weights(root)
        cfg = json.load(open(os.path.join(root, "config.json")))
        self.cfg = c = cfg["text_config"]
        self.H = c["hidden_size"]
        self.HC = c["hc_mult"]
        self.nL = c["num_hidden_layers"]
        self.eps = c["rms_norm_eps"]
        self.hc_eps = c["hc_eps"]
        self.hc_mag = c["hc_magnitude"]
        self.types = c["layer_types"]
        self.mlp_types = c["mlp_layer_types"]
        self.E = c["n_routed_experts"]
        self.topk = c["num_experts_per_tok"]
        self.scale = c["routed_scaling_factor"]
        self.limit = c["swiglu_limit"]
        self.kv_rank = c["kv_lora_rank"]
        self.rope = c["qk_rope_head_dim"]
        self.nope = c["qk_nope_head_dim"]
        self.vd = c["v_head_dim"]
        self.window = c["sliding_window"]
        self.ratio = c["compressed_indexer_compress_ratio"]
        self.gtopk = c["compressed_indexer_group_topk"]
        self.cand = c["compressed_indexer_candidate_topk"]
        self.ih = c["compressed_indexer_num_attention_heads"]
        self.ihd = c["compressed_indexer_head_dim"]
        self.engram_ids = c["engram_layer_ids"]  # 1-based layer ids
        self.eng_vocab = c["engram_layer_head_vocab_sizes"]
        self.eng_seed = c["engram_seed"]
        self.eng_n = c["engram_max_ngram_size"]
        self.eng_heads = c["engram_n_head_per_ngram"]
        self.eng_pad = c["engram_pad_id"]
        self.bases = {t: c["rope_parameters"][t]["rope_theta"] for t in c["rope_parameters"]}
        self.dump = {}

    # ---------------------------------------------------------------- engram
    def engram_hash(self, ids, lid, table):
        """hash ids [T, 16] for 1-based layer id lid (HF hash_ngrams + build_ngram_tokens)"""
        T = ids.shape[0]
        pad = self.eng_pad
        shifted = [ids]
        for o in range(1, self.eng_n):
            prev = torch.cat([torch.full((o,), pad, dtype=ids.dtype), ids[:-o]])[:T]
            shifted.append(prev)
        ng = torch.stack(shifted, -1)  # [T, n]
        tsz = table.shape[0]
        comp = table[ng.clamp_max(tsz - 1)].long()
        comp = torch.where(ng < tsz, comp, table[pad].long())
        # bounded by the compressed vocabulary (HF _init_weights: lookup_table.max() + 1), not the map length
        half = max(1, np.iinfo(np.int64).max // (int(table.max()) + 1) // 2)
        g = np.random.default_rng(self.eng_seed + 10007 * lid)
        mult = torch.tensor((g.integers(0, half, size=self.eng_n, dtype=np.int64) * 2 + 1).tolist(), dtype=torch.int64)
        mixed = comp * mult
        sizes = [s for order in self.eng_vocab[str(lid)] for s in order]
        out = []
        for n in range(2, self.eng_n + 1):
            mix = mixed[..., 0]
            for o in range(1, n):
                mix = torch.bitwise_xor(mix, mixed[..., o])
            m = torch.tensor(sizes[(n - 2) * self.eng_heads:(n - 1) * self.eng_heads], dtype=torch.int64)
            out.append(mix[:, None] % m)
        return torch.cat(out, -1), mult, sizes

    def engram(self, i, h, ids):
        lid = i + 1
        p = f"model.layers.{i}.engram_layer."
        W = self.W
        table = W.get(p + "lookup_table", None)
        hid, mult, sizes = self.engram_hash(ids, lid, table)
        offs = torch.tensor([0] + sizes[:-1], dtype=torch.int64).cumsum(0)
        flat = (hid + offs).reshape(-1)
        emb = W.rows(p + "multi_head_embedding.embedding.weight", flat.tolist()).reshape(hid.shape[0], hid.shape[1], -1)
        # fp8 round trip per vector, as the memory was trained to be read
        amax = emb.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
        emb = (emb / amax).clamp(-448, 448).to(torch.float8_e4m3fn).float() * amax
        feps = torch.finfo(torch.float32).eps  # nn.RMSNorm(eps=None) in f32
        nh = emb.shape[1]
        emb = torch.stack([rms(emb[:, j], W.get(p + f"lookup_per_head_norms.{j}.weight"), feps) for j in range(nh)], 1)
        e = emb.reshape(emb.shape[0], -1)  # [T, 1536]
        self.dump[f"L{i}.engram_rows"] = e
        out = torch.zeros_like(h)
        v = lin(e, W.get(p + "value_proj.weight"), f"L{i}.eng.kv") + W.get(p + "value_proj.bias")
        main = W.get(p + "unified_pre_gamma") * rms(v, None, feps)
        for s in range(self.HC):
            k = lin(e, W.get(p + f"key_projs.{s}.weight")) + W.get(p + f"key_projs.{s}.bias")
            k = rms(k, W.get(p + f"norm_key.{s}.weight"), feps)
            q = rms(h[:, s], W.get(p + f"norm_query.{s}.weight"), feps)
            g = (k * q).sum(-1) * self.H ** -0.5
            g = g.abs().clamp_min(1e-6).sqrt() * torch.where(g >= 0, 1.0, -1.0)
            g = torch.sigmoid(g)
            out[:, s] = g[:, None] * main + W.get(p + f"unified_per_hc_scales.{s}") * main
        mask = ids != self.eng_pad
        out = out * mask[:, None, None]
        return h + out

    # ---------------------------------------------------------------- hyper-connections
    def hc_pre(self, h, pfx):
        fn = self.W.get(pfx + "hc_fn")
        base = self.W.get(pfx + "hc_base")
        sc = self.W.get(pfx + "hc_scale")
        flat = h.reshape(h.shape[0], -1)
        mixes = (flat @ fn.t()) * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + self.eps)
        pre = torch.sigmoid(mixes[:, :self.HC] * sc[0] + base[:self.HC]) + self.hc_eps
        post = self.hc_mag * torch.sigmoid(mixes[:, self.HC:] * sc[1] + base[self.HC:]) + self.hc_eps
        return post, (pre[:, :, None] * h).sum(1)

    # ---------------------------------------------------------------- attention
    def attention(self, i, x, pos):
        W = self.W
        p = f"model.layers.{i}.self_attn."
        T = x.shape[0]
        sliding = self.types[i] == "sliding_attention"
        nh = W.get(p + "learnable_sink_param").shape[0]
        sinks = W.get(p + "learnable_sink_param")
        gate = lin(x, W.get(p + "linear_gate.weight"), f"L{i}.attn.gate").view(T, nh, self.vd)
        qa = lin(x, W.get(p + "q_a_proj.weight"), f"L{i}.attn.qa")
        q_resid = rms(qa, W.get(p + "q_a_layernorm.weight"), self.eps)
        q = lin(q_resid, W.get(p + "q_b_proj.weight"), f"L{i}.attn.qb").view(T, nh, self.nope + self.rope).transpose(0, 1)
        q_pass, q_rot = q[..., :self.nope], q[..., self.nope:]
        ckv = lin(x, W.get(p + "kv_a_proj_with_mqa.weight"), f"L{i}.attn.kva")
        kv_pass, k_rot = ckv[:, :self.kv_rank], ckv[:, self.kv_rank:]
        k_pass = rms(kv_pass, W.get(p + "kv_a_layernorm.weight"), self.eps)
        base = self.bases["sliding_attention" if sliding else "deepseek_sparse_attention"]
        cos, sin = rope_tables(pos, self.rope, base)
        q_rot = apply_rope(q_rot, cos, sin)
        k_rot = apply_rope(k_rot, cos, sin)
        self.dump[f"L{i}.latent"] = torch.cat([k_pass, k_rot], -1)
        kvb = lin(k_pass, W.get(p + "kv_b_proj.weight"), f"L{i}.attn.kvb").view(T, nh, self.nope + self.vd).transpose(0, 1)
        k_nope, v = kvb[..., :self.nope], kvb[..., self.nope:]
        qf = torch.cat([q_pass, q_rot], -1)
        kf = torch.cat([k_nope, k_rot[None].expand(nh, -1, -1)], -1)
        if sliding and T > 2 * self.window:
            # banded: each 512-query block sees its own keys and the previous block's (same result, O(T * W))
            W_ = self.window
            o = torch.empty(T, nh, self.vd)
            for b0 in range(0, T, W_):
                b1 = min(T, b0 + W_)
                k0 = max(0, b0 - W_)
                s = (qf[:, b0:b1] @ kf[:, k0:b1].transpose(-1, -2)) * (self.nope + self.rope) ** -0.5
                qi = torch.arange(b0, b1)[:, None]
                ki = torch.arange(k0, b1)[None, :]
                s = s.masked_fill(~((ki <= qi) & (ki > qi - W_)), float("-inf"))
                s = torch.cat([s, sinks[:, None, None].expand(nh, b1 - b0, 1)], -1)
                s = s - s.max(-1, keepdim=True).values
                pr = torch.softmax(s, -1)[..., :-1]
                o[b0:b1] = (pr @ v[:, k0:b1]).transpose(0, 1)
            o = o * torch.sigmoid(gate)
            o = lin(o.reshape(T, -1), W.get(p + "o_proj.weight"), f"L{i}.attn.o")
            return rms(o, W.get(p + "post_proj_norm.weight"), self.eps)
        qi = torch.arange(T)[:, None]
        ki = torch.arange(T)[None, :]
        allowed = ki <= qi
        if sliding:
            allowed = allowed & (ki > qi - self.window)
        elif T > self.cand:
            allowed = allowed & self.recall(i, x, q_resid, pos)
        if T > 4096:
            # query blocks of 1024 rows: the same result without a [heads, T, T] score tensor
            o = torch.empty(T, nh, self.vd)
            for b0 in range(0, T, 1024):
                b1 = min(T, b0 + 1024)
                s = (qf[:, b0:b1] @ kf[:, :b1].transpose(-1, -2)) * (self.nope + self.rope) ** -0.5
                s = s.masked_fill(~allowed[b0:b1, :b1], float("-inf"))
                s = torch.cat([s, sinks[:, None, None].expand(nh, b1 - b0, 1)], -1)
                s = s - s.max(-1, keepdim=True).values
                pr = torch.softmax(s, -1)[..., :-1]
                o[b0:b1] = (pr @ v[:, :b1]).transpose(0, 1)
            o = o * torch.sigmoid(gate)
            o = lin(o.reshape(T, -1), W.get(p + "o_proj.weight"), f"L{i}.attn.o")
            return rms(o, W.get(p + "post_proj_norm.weight"), self.eps)
        s = (qf @ kf.transpose(-1, -2)) * (self.nope + self.rope) ** -0.5
        s = s.masked_fill(~allowed, float("-inf"))
        s = torch.cat([s, sinks[:, None, None].expand(nh, T, 1)], -1)
        s = s - s.max(-1, keepdim=True).values
        pr = torch.softmax(s, -1)[..., :-1]
        o = (pr @ v).transpose(0, 1)  # [T, nh, vd]
        o = o * torch.sigmoid(gate)
        o = lin(o.reshape(T, -1), W.get(p + "o_proj.weight"), f"L{i}.attn.o")
        return rms(o, W.get(p + "post_proj_norm.weight"), self.eps)

    def recall(self, i, x, q_resid, pos):
        """[T, T] bool: keys each query may attend to by the compressed indexer (HF _mask_unrecalled_tokens)"""
        W = self.W
        p = f"model.layers.{i}.self_attn.compressed_indexer."
        T = x.shape[0]
        r = self.ratio
        tk = rms(lin(x, W.get(p + "token_key_projection.weight"), f"L{i}.idx.k"), W.get(p + "token_key_norm.weight"), self.eps)
        G = T // r
        g = tk[:G * r].view(G, r, self.ihd)
        vals = g @ W.get(p + "key_value_projection.weight").t()
        gts = g @ W.get(p + "key_gate_projection.weight").t() + W.get(p + "intra_group_ape")
        pooled = (vals * torch.softmax(gts, 1)).sum(1)
        gk = rms(pooled, W.get(p + "key_norm.weight"), self.eps)
        base = self.bases["deepseek_sparse_attention"]
        nope = self.ihd - self.rope
        gc, gsn = rope_tables(pos[:G * r:r], self.rope, base)
        gk = torch.cat([gk[:, :nope], apply_rope(gk[:, nope:], gc, gsn)], -1)
        q = lin(q_resid, W.get(p + "query_projection.weight"), f"L{i}.idx.q").view(T, self.ih, self.ihd)
        qc, qs = rope_tables(pos, self.rope, base)
        q = torch.cat([q[..., :nope], apply_rope(q[..., nope:], qc[:, None], qs[:, None])], -1)
        hw = lin(x, W.get(p + "head_weight_projection.weight"), f"L{i}.idx.w") * (self.ih ** -0.5 * self.ihd ** -0.5)
        sc = torch.relu(torch.einsum("thd,gd->thg", q, gk))
        logits = torch.einsum("th,thg->tg", hw, sc)
        qg = torch.arange(T) // r
        logits = logits.masked_fill(torch.arange(G)[None, :] >= qg[:, None], float("-inf"))
        kk = min(self.gtopk - 1, G)
        rec = torch.cat([logits.topk(kk, -1).indices, qg[:, None]], -1)
        self.dump[f"L{i}.recall"] = rec
        toks = (rec[..., None] * r + torch.arange(r)).flatten(-2).clamp(max=T - 1)
        allowed = torch.zeros(T, T, dtype=torch.bool)
        allowed.scatter_(1, toks, True)
        return allowed

    # ---------------------------------------------------------------- ffn
    def dense_mlp(self, i, x):
        W = self.W
        p = f"model.layers.{i}.mlp."
        g = lin(x, W.get(p + "gate_proj.weight"), f"L{i}.mlp.gu")
        u = x @ W.get(p + "up_proj.weight").t()
        a = F.silu(g) * u
        d = lin(a, W.get(p + "down_proj.weight"), f"L{i}.mlp.dn")
        return rms(d, W.get(p + "post_fc2_norm.weight"), self.eps)

    def moe(self, i, x):
        W = self.W
        p = f"model.layers.{i}.mlp."
        T = x.shape[0]
        logits = lin(x, W.get(p + "gate.weight"), f"L{i}.moe.router")
        scores = torch.sigmoid(logits)
        choice = scores + W.get(p + "gate.e_score_correction_bias")
        top = torch.topk(choice, self.topk, -1, sorted=True).indices
        w = scores.gather(1, top)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * self.scale
        self.dump[f"L{i}.topi"] = top
        lat = lin(x, W.get(p + "routed_expert_down_proj.weight"), f"L{i}.moe.ldn")
        out = torch.zeros_like(lat)
        hit = sorted(set(top.reshape(-1).tolist()))
        for c0 in range(0, len(hit), 64):  # 64 experts at a time bounds the f32 copies (~1.8 GB)
          chunk = hit[c0:c0 + 64]
          gu_all = W.rows(p + "experts.gate_up_proj", chunk)  # [n, 2I, 1536]
          dn_all = W.rows(p + "experts.down_proj", chunk)     # [n, 1536, I]
          I = gu_all.shape[1] // 2
          for j, e in enumerate(chunk):
            rows, slot = torch.where(top == e)
            xe = lat[rows]
            STATS and STATS.add_expert(f"L{i}.moe.gu", e, xe, self.E)
            gu = xe @ gu_all[j].t()
            g = gu[:, :I].clamp(max=self.limit)
            u = gu[:, I:].clamp(-self.limit, self.limit)
            a = F.silu(g) * u
            STATS and STATS.add_expert(f"L{i}.moe.dn", e, a, self.E)
            y = a @ dn_all[j].t()
            out.index_add_(0, rows, y * w[rows, slot, None])
        routed = lin(out, W.get(p + "routed_expert_up_proj.weight"), f"L{i}.moe.lup")
        sg = lin(x, W.get(p + "shared_experts.gate_proj.weight"), f"L{i}.moe.sgu")
        su = x @ W.get(p + "shared_experts.up_proj.weight").t()
        sa = F.silu(sg) * su
        sh = lin(sa, W.get(p + "shared_experts.down_proj.weight"), f"L{i}.moe.sdn")
        return rms(routed + sh, W.get(p + "combine_output_norm.weight"), self.eps)

    # ---------------------------------------------------------------- model
    def forward(self, ids, keep=None, layers=None):
        W = self.W
        T = ids.shape[0]
        pos = torch.arange(T)
        x = W.get("model.embed_tokens.weight", None)[ids].float()
        h = x[:, None, :].expand(T, self.HC, self.H).contiguous()
        t0 = time.time()
        for i in range(self.nL if layers is None else layers):
            if (i + 1) in self.engram_ids:
                h = self.engram(i, h, ids)
            res = h
            post, a = self.hc_pre(h, f"model.layers.{i}.hc_attn_layer.hc_pre.")
            a = rms(a, W.get(f"model.layers.{i}.input_layernorm.weight"), self.eps)
            a = self.attention(i, a, pos)
            h = post[:, :, None] * a[:, None, :] + res
            res = h
            post, m = self.hc_pre(h, f"model.layers.{i}.hc_mlp_layer.hc_pre.")
            m = rms(m, W.get(f"model.layers.{i}.post_attention_layernorm.weight"), self.eps)
            m = self.dense_mlp(i, m) if self.mlp_types[i] == "dense" else self.moe(i, m)
            h = post[:, :, None] * m[:, None, :] + res
            self.dump[f"L{i}.h"] = h[-1].clone()
            print(f"layer {i} {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
        fn = W.get("model.hc_head.hc_head_fn")
        flat = h.reshape(T, -1)
        mixes = (flat @ fn.t()) * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + self.eps)
        pre = torch.sigmoid(mixes * W.get("model.hc_head.hc_head_scale") + W.get("model.hc_head.hc_head_base")) + self.hc_eps
        y = (pre[:, :, None] * h).sum(1)
        y = rms(y, W.get("model.norm.weight"), self.eps)
        self.dump["h_final"] = y
        if keep:
            y = y[-keep:]
        return lin(y, W.get("lm_head.weight"), "lm_head")


def load_ids(p):
    if p.endswith(".npy"):
        return torch.from_numpy(np.load(p).astype(np.int64))
    return torch.from_numpy(np.fromfile(p, dtype=np.uint32).astype(np.int64))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("ids")
    ap.add_argument("out")
    ap.add_argument("--stats")
    ap.add_argument("--keep", type=int, default=0)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--dump", action="store_true")
    a = ap.parse_args()
    if a.stats:
        STATS = Stats()
    m = Hy4(a.ckpt)
    ids = load_ids(a.ids)
    lg = m.forward(ids, keep=a.keep or None, layers=a.layers)
    out = {"logits": lg.numpy(), "ids": ids.numpy()}
    if a.dump:
        out.update({k: v.numpy() for k, v in m.dump.items()})
    np.savez(a.out, **out)
    if a.stats:
        torch.save({"s": STATS.s, "n": STATS.n}, a.stats)
    print("ok", lg.shape, file=sys.stderr)
