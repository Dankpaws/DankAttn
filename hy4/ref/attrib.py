"""Per-tensor-class fake-quantization attribution for Hy4-flash.

Runs hy4_golden.py's exact f32 forward pass while fake-quantizing one or more
tensor classes with convert/pack.py's tquant (importance-weighted clip search),
and compares the resulting logits against a saved BF16 golden .npz.

Usage:
  python3 attrib.py <ckpt> <golden.npz> --stats <calib.pt> --spec <class:bits:gs> [...]

Classes:
  experts    routed experts gate_up + down (key L*.moe.gu / L*.moe.dn)
  attn.gate  attention linear_gate (L*.attn.gate)
  attn.o     attention o_proj (L*.attn.o)
  attn.in    q_a + kv_a + idx_k + idx_w (L*.attn.qa/kva, L*.idx.k/w)
  attn.qb    q_b + idx_q (L*.attn.qb, L*.idx.q)
  attn.kvb   kv_b_proj (L*.attn.kvb)
  moe.io     router + ldn + lup + shared expert (L*.moe.router/ldn/lup/sgu/sdn)
  dense_mlp  layer 0 dense MLP (L0.mlp.gu/dn)
  eng.kv     Engram value/key projections (L*.eng.kv)
  lm_head    output head (lm_head)
  all_live   current live pack-g64 recipe (experts:4:64 + everything else W8:128)
"""
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "convert"))
sys.path.insert(0, os.path.join(ROOT, "hy4", "convert"))

import hy4_golden as G  # noqa: E402
import pack as P  # noqa: E402
import gptq as GQ  # noqa: E402

torch.set_grad_enabled(False)


def key_class(key):
    if not key:
        return None
    if key == "lm_head":
        return "lm_head"
    if key.endswith(".attn.gate"):
        return "attn.gate"
    if key.endswith(".attn.o"):
        return "attn.o"
    if key.endswith((".attn.qa", ".attn.kva", ".idx.k", ".idx.w")):
        return "attn.in"
    if key.endswith((".attn.qb", ".idx.q")):
        return "attn.qb"
    if key.endswith(".attn.kvb"):
        return "attn.kvb"
    if key.endswith((".moe.gu", ".moe.dn")):
        return "experts"
    if key.endswith((".moe.router", ".moe.ldn", ".moe.lup", ".moe.sgu", ".moe.sdn")):
        return "moe.io"
    if key.endswith((".mlp.gu", ".mlp.dn")):
        return "dense_mlp"
    if key.endswith(".eng.kv"):
        return "eng.kv"
    return None


def fake_quant(w, bits, gs, imp_key=None, expert_idx=None, n_experts=1):
    N, K = w.shape
    g = gs if K % gs == 0 else (64 if K % 64 == 0 else 32)
    imp = None
    if imp_key is not None and P.IMP is not None:
        if expert_idx is not None:
            all_imp = P.importance(imp_key, K, experts=n_experts)
            if all_imp is not None:
                imp = all_imp[expert_idx]
        else:
            imp = P.importance(imp_key, K, experts=1)
    q, s = P.tquant(w, bits, g, clip=True, imp=imp)
    return (q.float().reshape(N, K // g, g) * s[..., None]).reshape(N, K)


def lsm(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def parse_specs(specs):
    rules = {}
    for sp in specs:
        if sp == "all_live":
            for c in ("attn.gate", "attn.o", "attn.in", "attn.qb", "attn.kvb", "moe.io", "dense_mlp", "eng.kv", "lm_head"):
                rules[c] = (8, 128)
            rules["experts"] = (4, 64)
            continue
        cls, b, gs = sp.split(":")
        rules[cls] = (int(b), int(gs))
    return rules


def collect_gptq_calib(ckpt, calib_path, layers=None, need_dn=False):
    """Collect per-layer latent Hessians H_gu (and optional lat/top for sequential moe.dn GPTQ)."""
    ids = G.load_ids(calib_path)
    m = G.Hy4(ckpt)
    calib = {}
    orig_moe = m.moe

    def tap_moe(i, x):
        W = m.W
        p = f"model.layers.{i}.mlp."
        logits = G.lin(x, W.get(p + "gate.weight"))
        scores = torch.sigmoid(logits)
        choice = scores + W.get(p + "gate.e_score_correction_bias")
        top = torch.topk(choice, m.topk, -1, sorted=True).indices
        lat = G.lin(x, W.get(p + "routed_expert_down_proj.weight"))
        H_gu = (lat.double().T @ lat.double()) / max(lat.shape[0], 1)
        entry = {"H_gu": H_gu.float()}
        if need_dn:
            entry["lat"] = lat.half()
            entry["top"] = top.to(torch.int16)
        calib[i] = entry
        return orig_moe(i, x)

    m.moe = tap_moe
    t0 = time.time()
    m.forward(ids, keep=1, layers=layers)
    print(f"collected GPTQ calib from {calib_path} ({len(ids)} tok, {len(calib)} MoE layers) in {time.time() - t0:.1f}s",
          file=sys.stderr, flush=True)
    return calib


def run_attrib(ckpt, golden_path, stats_path, rules, max_tokens=0, layers=None, s0=0,
               gptq_calib=None, gptq_mode="both", gptq_damp=0.01, gptq_damp_dn=0.05):
    g_npz = np.load(golden_path)
    g_logits = g_npz["logits"]
    ids_np = g_npz["ids"]
    if max_tokens and max_tokens < len(ids_np):
        ids_np = ids_np[:max_tokens]
        g_logits = g_logits[:max_tokens]
    ids = torch.from_numpy(ids_np.astype(np.int64))

    if stats_path:
        P.IMP = torch.load(stats_path, map_location="cpu")

    calib = None
    if gptq_calib:
        calib = collect_gptq_calib(ckpt, gptq_calib, layers=layers, need_dn=(gptq_mode == "both"))

    orig_lin = G.lin

    def patched_lin(x, w, key=None):
        cls = key_class(key)
        if cls in rules:
            bits, gs = rules[cls]
            w = fake_quant(w, bits, gs, imp_key=key)
        return orig_lin(x, w, key)

    G.lin = patched_lin

    m = G.Hy4(ckpt)

    # Also patch routed experts and up_proj (unkeyed in hy4_golden) if requested
    exp_rule = rules.get("experts")
    if exp_rule is not None:
        ebits, egs = exp_rule

        def patched_moe(i, x):
            W = m.W
            p = f"model.layers.{i}.mlp."
            logits = G.lin(x, W.get(p + "gate.weight"), f"L{i}.moe.router")
            scores = torch.sigmoid(logits)
            choice = scores + W.get(p + "gate.e_score_correction_bias")
            top = torch.topk(choice, m.topk, -1, sorted=True).indices
            w = scores.gather(1, top)
            w = w / (w.sum(-1, keepdim=True) + 1e-20) * m.scale
            lat = G.lin(x, W.get(p + "routed_expert_down_proj.weight"), f"L{i}.moe.ldn")
            out = torch.zeros_like(lat)
            hit = sorted(set(top.reshape(-1).tolist()))
            imp_gu = P.importance(f"L{i}.moe.gu", lat.shape[-1], experts=m.E) if P.IMP is not None else None
            imp_dn = P.importance(f"L{i}.moe.dn", m.cfg["moe_intermediate_size"], experts=m.E) if P.IMP is not None else None
            Hinv_gu = hdiag_gu = cal_lat = cal_top = None
            if calib is not None and i in calib:
                Hinv_gu, hdiag_gu = GQ.prepare_hinv(calib[i]["H_gu"], damp=gptq_damp)
                if gptq_mode == "both":
                    cal_lat = calib[i]["lat"].float()
                    cal_top = calib[i]["top"].long()
            for c0 in range(0, len(hit), 64):
                chunk = hit[c0:c0 + 64]
                gu_all = W.rows(p + "experts.gate_up_proj", chunk)
                dn_all = W.rows(p + "experts.down_proj", chunk)
                I = gu_all.shape[1] // 2
                for j, e in enumerate(chunk):
                    rows, slot = torch.where(top == e)
                    xe = lat[rows]
                    if Hinv_gu is not None:
                        qgu, sgu = GQ.gptq_quant(gu_all[j], bits=ebits, gs=egs, clip=True, Hinv=Hinv_gu, hdiag=hdiag_gu)
                    else:
                        qgu, sgu = P.tquant(gu_all[j], ebits, egs, clip=True, imp=None if imp_gu is None else imp_gu[e])
                    wgu = (qgu.float().reshape(2 * I, -1, egs) * sgu[..., None]).reshape(2 * I, -1)
                    gu = xe @ wgu.t()
                    g = gu[:, :I].clamp(max=m.limit)
                    u = gu[:, I:].clamp(-m.limit, m.limit)
                    a = F.silu(g) * u
                    use_gq_dn = False
                    if cal_lat is not None:
                        crows, _ = torch.where(cal_top == e)
                        if crows.numel() >= 8:
                            cgu = cal_lat[crows] @ wgu.t()
                            ca = F.silu(cgu[:, :I].clamp(max=m.limit)) * cgu[:, I:].clamp(-m.limit, m.limit)
                            H_dn = (ca.double().T @ ca.double()) / crows.numel()
                            qdn, sdn = GQ.gptq_quant(dn_all[j], H=H_dn, bits=ebits, gs=egs, damp=gptq_damp_dn, clip=True)
                            use_gq_dn = True
                    if not use_gq_dn:
                        qdn, sdn = P.tquant(dn_all[j], ebits, egs, clip=True, imp=None if imp_dn is None else imp_dn[e])
                    wdn = (qdn.float().reshape(lat.shape[-1], -1, egs) * sdn[..., None]).reshape(lat.shape[-1], -1)
                    y = a @ wdn.t()
                    out.index_add_(0, rows, y * w[rows, slot, None])
            routed = G.lin(out, W.get(p + "routed_expert_up_proj.weight"), f"L{i}.moe.lup")
            sg = G.lin(x, W.get(p + "shared_experts.gate_proj.weight"), f"L{i}.moe.sgu")
            su = G.lin(x, W.get(p + "shared_experts.up_proj.weight"), f"L{i}.moe.sgu")
            sa = F.silu(sg) * su
            sh = G.lin(sa, W.get(p + "shared_experts.down_proj.weight"), f"L{i}.moe.sdn")
            return G.rms(routed + sh, W.get(p + "combine_output_norm.weight"), m.eps)

        m.moe = patched_moe

    t0 = time.time()
    lg = m.forward(ids, layers=layers).numpy()
    dt = time.time() - t0
    G.lin = orig_lin

    Gf = g_logits[s0:].astype(np.float64)
    Ef = lg[s0:].astype(np.float64)
    id_s = ids_np[s0:]
    T = Gf.shape[0]
    lg_g = lsm(Gf)
    lg_e = lsm(Ef)
    kl = (np.exp(lg_g) * (lg_g - lg_e)).sum(-1)
    top1 = (Gf.argmax(-1) == Ef.argmax(-1)).mean()
    nll_g = -lg_g[np.arange(T - 1), id_s[1:]].mean()
    nll_e = -lg_e[np.arange(T - 1), id_s[1:]].mean()
    tag = f"GPTQ({gptq_mode},d={gptq_damp}/{gptq_damp_dn})" if gptq_calib else "RTN"
    print(f"RESULT {tag} {rules} ({T} tok, {dt:.1f}s): KL {kl.mean():.5f} (med {np.median(kl):.5f}, max {kl.max():.4f})  "
          f"top-1 {100*top1:.2f}%  NLL {nll_e:.4f} (golden {nll_g:.4f})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("golden")
    ap.add_argument("--stats", default=None)
    ap.add_argument("--spec", nargs="+", required=True, help="class:bits:gs or all_live")
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--from", dest="s0", type=int, default=0)
    ap.add_argument("--gptq-calib", default=None, help="calibration ids.bin for GPTQ Hessians")
    ap.add_argument("--gptq-mode", default="both", choices=["gu", "both"])
    ap.add_argument("--gptq-damp", type=float, default=0.01)
    ap.add_argument("--gptq-damp-dn", type=float, default=0.05)
    a = ap.parse_args()
    run_attrib(a.ckpt, a.golden, a.stats, parse_specs(a.spec), a.max_tokens, a.layers, a.s0,
               a.gptq_calib, a.gptq_mode, a.gptq_damp, a.gptq_damp_dn)
