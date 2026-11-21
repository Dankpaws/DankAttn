"""BF16 Qwen3.8-Flash-Next -> opus pack v0.

Every linear weight becomes register-direct int4/int8 codes with f16 group scales (see
convert/layout.py). Projections that share an input are concatenated into one matrix so
the engine issues one matmul per input. Precision plan v0 (pinned in the manifest):
  routed experts      W4 g128   (clip-searched RTN)
  everything dense    W8 g128 (g64 where K = 320)
  embed_tokens        bf16 rows
  small vectors       f32
The PLE n-gram table is not packed here (v0 reads it from SSD; see manifest "ngram").
"""
import json, math, os, sys, time
import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from convert.layout import pack_codes, pack_scales

torch.set_num_threads(os.cpu_count())
PAGE = 16384


CLIPS_V0 = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75)
CLIPS_V1 = tuple(1.0 - 0.025 * i for i in range(9))
CLIPS_V1_W8 = (1.0, 0.975, 0.95)
IMP = None  # importance stats: {key: (sum x^2, count)} from ref/calibrate.py


def importance(key, K, experts=1):
    """Per-input-channel weights E[x^2] for a tensor (experts: [E, K]); None if unknown."""
    if IMP is None or key not in IMP["s"]:
        return None
    s, n = IMP["s"][key], IMP["n"][key]
    if experts > 1:
        n = n.clamp_min(1).double()[:, None]
        m = (s / n).float()
        # experts with no calibration rows fall back to the layer's mean importance
        empty = (IMP["n"][key] == 0)
        layer = (s.sum(0) / IMP["n"][key].sum().clamp_min(1)).float()
        if empty.any():
            m[empty] = layer
        # few calibration rows per expert: blend with the layer-wide moments by row count
        wgt = (IMP["n"][key].float() / (IMP["n"][key].float() + 32.0))[:, None]
        m = wgt * m + (1 - wgt) * layer[None, :]
    else:
        m = (s / max(int(n), 1)).float()
    m = m / m.mean(-1, keepdim=True).clamp_min(1e-30)
    return m + 1e-3  # keep every channel slightly weighted


def hc_up_rows(H, HC):
    """Row order for hc.up: each 32-row fragment holds 8 channels x HC streams (s*8 + j),
    so the matmul epilogue can mix the streams of a channel without leaving the fragment."""
    order = []
    for f in range(H // 8):
        for st in range(HC):
            for j in range(8):
                order.append(st * H + 8 * f + j)
    return torch.tensor(order)


def gu_rows(I):
    """Row order for expert gate|up: fragment f holds gate rows 16f.. and up rows 16f..,
    so silu(gate) * up is computed inside the fragment."""
    order = []
    for f in range(I // 16):
        order += list(range(16 * f, 16 * f + 16)) + list(range(I + 16 * f, I + 16 * f + 16))
    return torch.tensor(order)


def tquant(w, bits, gs, clip=True, imp=None):
    """torch version of layout.quantize. w f32 [N, K] -> (int8 codes [N, K], f32 scales [N, G]).
    imp: optional per-input-channel importance [K] (or [N, K]); the clip search then
    minimises the importance-weighted squared error (diagonal-Hessian proxy)."""
    N, K = w.shape
    qmax, qmin = (7, -8) if bits == 4 else ((31, -32) if bits == 6 else (127, -128))
    g = w.reshape(N, K // gs, gs)
    amax = g.abs().amax(-1).clamp_min(1e-12)
    best = amax / qmax
    wi = None
    if imp is not None:
        wi = (imp if imp.dim() == 2 else imp[None, :].expand(N, K)).reshape(N, K // gs, gs)
    if clip:
        berr = None
        clips = (CLIPS_V1 if bits == 4 else CLIPS_V1_W8) if IMP is not None else (CLIPS_V0 if bits == 4 else (1.0,))
        for r in clips:
            s = (amax * r / qmax).half().float()
            q = torch.clamp(torch.round(g / s[..., None]), qmin, qmax)
            d2 = (q * s[..., None] - g) ** 2
            err = (d2 * wi).sum(-1) if wi is not None else d2.sum(-1)
            if berr is None:
                berr, best = err, s
            else:
                m = err < berr
                berr = torch.where(m, err, berr)
                best = torch.where(m, s, best)
    s16 = best.half().float().clamp_min(6e-8)
    q = torch.clamp(torch.round(g / s16[..., None]), qmin, qmax).to(torch.int8)
    return q.reshape(N, K), s16


class Writer:
    def __init__(self, out, max_file=8 << 30):
        self.out, self.max_file = out, max_file
        self.files, self.cur, self.size = [], None, 0
        self.tensors = {}
        os.makedirs(out, exist_ok=True)

    def _open(self):
        if self.cur:
            self.cur.close()
        name = f"pack-{len(self.files):02d}.bin"
        self.files.append(name)
        self.cur = open(os.path.join(self.out, name), "wb")
        self.size = 0

    def put(self, data: bytes):
        if self.cur is None or self.size + len(data) > self.max_file:
            self._open()
        pad = (-self.size) % PAGE
        if pad:
            self.cur.write(b"\0" * pad)
            self.size += pad
        off = self.size
        self.cur.write(data)
        self.size += len(data)
        return len(self.files) - 1, off

    def raw(self, name, arr, dtype="f32"):
        a = np.ascontiguousarray(arr.astype(np.float32) if dtype == "f32" else arr)
        f, o = self.put(a.tobytes())
        self.tensors[name] = dict(kind="raw", dtype=dtype, shape=list(a.shape), file=f, off=o, bytes=a.nbytes)

    def linear(self, name, w, bits, gs, n_pad=None, experts=1, rows=None, imp_key=None):
        """w: torch f32 [experts * N, K]. Codes then scales, one record."""
        EN, K = w.shape
        N = EN // experts
        if n_pad and n_pad > N:
            assert experts == 1
            w = torch.cat([w, torch.zeros(n_pad - N, K)], 0)
            N = n_pad
        q, s = tquant(w, bits, gs, imp=importance(imp_key or name, K))
        codes = pack_codes(q.numpy(), bits)
        sc = pack_scales(s.numpy())
        fc, oc = self.put(codes.tobytes())
        fs, os_ = self.put(sc.tobytes())
        self.tensors[name] = dict(kind="w", N=N, K=K, bits=bits, gs=gs, experts=experts,
                                  file=fc, off=oc, bytes=codes.nbytes, sfile=fs, soff=os_, sbytes=sc.nbytes,
                                  rows=rows or {})
        err = (q.float().reshape(EN if experts > 1 else N, K // gs, gs) * s[..., None]).reshape(-1, K) - w
        return float(err.pow(2).sum() / w.pow(2).sum().clamp_min(1e-30))

    def linear_experts(self, name, w3, bits, gs, chunk=64, imp_key=None):
        """w3: torch bf16 [E, N, K], quantized `chunk` experts at a time to bound memory.
        Codes of all experts are contiguous (expert e at e * N*K*bits/8), then scales."""
        E, N, K = w3.shape
        cb = N * K * bits // 8
        assert (chunk * cb) % PAGE == 0
        if self.cur is None or self.size + E * cb + PAGE > self.max_file:
            self._open()
        scs, num, den, first = [], 0.0, 0.0, None
        imp_all = importance(imp_key or name, K, experts=E)
        for e0 in range(0, E, chunk):
            w = w3[e0:e0 + chunk].float().reshape(-1, K)
            imp = None
            if imp_all is not None:
                ce = min(chunk, E - e0)
                imp = imp_all[e0:e0 + ce][:, None, :].expand(ce, N, K).reshape(-1, K)
            q, s = tquant(w, bits, gs, imp=imp)
            f, o = self.put(pack_codes(q.numpy(), bits).tobytes())
            first = (f, o) if first is None else first
            scs.append(pack_scales(s.numpy()))
            err = (q.float().reshape(-1, K // gs, gs) * s[..., None]).reshape(-1, K) - w
            num += float(err.pow(2).sum()); den += float(w.pow(2).sum())
        sc = np.concatenate(scs)
        fs, os_ = self.put(sc.tobytes())
        self.tensors[name] = dict(kind="w", N=N, K=K, bits=bits, gs=gs, experts=E, file=first[0], off=first[1],
                                  bytes=E * cb, sfile=fs, soff=os_, sbytes=sc.nbytes, rows={})
        return num / max(den, 1e-30)

    def close(self):
        if self.cur:
            self.cur.close()


def main(src, out):
    t0 = time.time()
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    cfg = json.load(open(os.path.join(src, "config.json")))
    tc = cfg["text_config"]
    handles = {}

    def T(name, dtype=torch.float32):
        f = idx[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(src, f), framework="pt")
        return handles[f].get_tensor(name).to(dtype)

    W = Writer(out)
    rel = {}
    H, HC, R = tc["hidden_size"], tc["hc_count"], tc["hc_lowrank"]
    pre = "model.language_model."

    def hc(dst, p):
        dn = T(p + "input_mix_weight_down.weight")
        rows = {"down": [0, R]}
        if (p + "block_inject_weight.weight") in idx:
            dn = torch.cat([dn, T(p + "block_inject_weight.weight")], 0)
            rows["inject"] = [R, R + HC]
        rel[dst + ".dn"] = W.linear(dst + ".dn", dn, 8, 128, n_pad=32 * math.ceil(dn.shape[0] / 32), rows=rows)
        rel[dst + ".up"] = W.linear(dst + ".up", T(p + "input_mix_weight_up.weight")[hc_up_rows(H, HC)], 8, 64)
        W.raw(dst + ".norm", T(p + "hc_norm.weight").numpy())

    if len(sys.argv) > 3:
        global IMP
        IMP = torch.load(sys.argv[3])
        print("importance stats:", len(IMP["s"]), "sets", flush=True)

    E = T(pre + "embed_tokens.weight", torch.bfloat16)
    W.raw("embed", E.view(torch.int16).numpy(), dtype="bf16")
    del E
    for L in range(tc["num_hidden_layers"]):
        p = f"{pre}layers.{L}."
        d = f"L{L}"
        hc(d + ".ahc", p + "attn_hyper_connection.")
        hc(d + ".mhc", p + "mlp_hyper_connection.")
        if tc["layer_types"][L] == "linear_attention":
            a = p + "linear_attn."
            ws = [T(a + "in_proj_qkv.weight"), T(a + "in_proj_z.weight"), T(a + "in_proj_b.weight"), T(a + "in_proj_a.weight")]
            n = [x.shape[0] for x in ws]
            rows = {"qkv": [0, n[0]], "z": [n[0], n[0] + n[1]], "b": [sum(n[:2]), sum(n[:3])], "a": [sum(n[:3]), sum(n)]}
            cat = torch.cat(ws, 0)
            rel[d + ".gdn.in"] = W.linear(d + ".gdn.in", cat, 8, 128, n_pad=32 * math.ceil(cat.shape[0] / 32), rows=rows)
            rel[d + ".gdn.out"] = W.linear(d + ".gdn.out", T(a + "out_proj.weight"), 8, 128)
            W.raw(d + ".gdn.conv", T(a + "conv1d.weight").reshape(-1, 4).numpy())
            W.raw(d + ".gdn.A_log", T(a + "A_log").numpy())
            W.raw(d + ".gdn.dt_bias", T(a + "dt_bias").numpy())
            W.raw(d + ".gdn.norm", T(a + "norm.weight").numpy())
        else:
            a = p + "self_attn."
            ws = [T(a + "q_proj.weight"), T(a + "k_proj.weight"), T(a + "v_proj.weight"), T(a + "indexer.index_qk_proj.weight")]
            n = [x.shape[0] for x in ws]
            rows = {"qg": [0, n[0]], "k": [n[0], sum(n[:2])], "v": [sum(n[:2]), sum(n[:3])], "idx": [sum(n[:3]), sum(n)]}
            cat = torch.cat(ws, 0)
            rel[d + ".attn.in"] = W.linear(d + ".attn.in", cat, 8, 128, n_pad=32 * math.ceil(cat.shape[0] / 32), rows=rows)
            rel[d + ".attn.o"] = W.linear(d + ".attn.o", T(a + "o_proj.weight"), 8, 128)
            for k in ["q_norm", "k_norm", "indexer.q_layernorm", "indexer.k_layernorm"]:
                W.raw(d + ".attn." + k, T(a + k + ".weight").numpy())
        m = p + "mlp."
        ws = [T(m + "gate.weight"), T(m + "shared_expert.gate_proj.weight"), T(m + "shared_expert.up_proj.weight"),
              T(m + "shared_expert_gate.weight")]
        n = [x.shape[0] for x in ws]
        cat = torch.cat(ws, 0)
        rows = {"router": [0, n[0]], "sgate_proj": [n[0], sum(n[:2])], "sup": [sum(n[:2]), sum(n[:3])], "sgate": [sum(n[:3]), sum(n)]}
        rel[d + ".moe.in"] = W.linear(d + ".moe.in", cat, 8, 128, n_pad=32 * math.ceil(cat.shape[0] / 32), rows=rows)
        rel[d + ".moe.sdown"] = W.linear(d + ".moe.sdown", T(m + "shared_expert.down_proj.weight"), 8, 128)
        egs = int(os.environ.get("OPUS_EXPERT_GS", "128"))
        rel[d + ".moe.gu"] = W.linear_experts(d + ".moe.gu", T(m + "experts.gate_up_proj", torch.bfloat16)[:, gu_rows(tc["moe_intermediate_size"])], 4, egs)
        rel[d + ".moe.dn"] = W.linear_experts(d + ".moe.dn", T(m + "experts.down_proj", torch.bfloat16), 4, egs)
        if (L + 1) in (tc.get("ple_layer_ids") or []):
            q = p + "ple."
            kv = torch.cat([T(q + "key_proj.weight"), T(q + "value_proj.weight")], 0)
            rel[d + ".ple.kv"] = W.linear(d + ".ple.kv", kv, 8, 128, rows={"key": [0, HC * H], "value": [HC * H, HC * H + H]})
            for k in ["norm_key", "norm_query", "norm_conv"]:
                W.raw(d + ".ple." + k, T(q + k + ".weight").numpy())
            W.raw(d + ".ple.conv", T(q + "conv1d.weight").reshape(-1, tc["ple_conv_kernel_size"]).numpy())
        print(f"layer {L:2d} done {time.time() - t0:6.0f}s  experts rel.err gu {rel[d + '.moe.gu']:.2e} dn {rel[d + '.moe.dn']:.2e}", flush=True)
    hc("mix", pre + "hyper_connection_mixer.")
    rel["lm_head"] = W.linear("lm_head", T("lm_head.weight"), 8, 128)
    W.close()
    man = dict(format="opus-pack-v0", source=dict(repo="Qwen/Qwen3.8-Flash-Next", revision="7ecef55a2e549814c62143fcf8b5be457695e3b9"),
               precision=os.environ.get("OPUS_PRECISION", "experts W4g128 clip-RTN; dense W8g128 (hc.up g64); embed bf16; vectors f32; activations bf16"),
               config=tc, files=W.files, tensors=W.tensors, rel_mse=rel,
               layout={"hc_up": "streams_interleaved_8", "moe_gu": "gate_up_interleaved_16"})
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print("done", time.time() - t0, "s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
