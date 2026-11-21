"""Collect per-layer MoE latent Hessians (H_gu) and optional (lat, top) for GPTQ packing.

Usage:
  python3 collect_gptq.py <bf16 ckpt> <calib_ids.bin> <out.pt> [--layers N] [--chunk 4096] [--gu-only]

Output .pt dict:
  {
    "source": calib_ids.bin,
    "tokens": T,
    "layers": {
      L: {"H_gu": f32 [1536, 1536], "lat": f16 [T, 1536], "top": i16 [T, topk]},
      ...
    }
  }
"""
import argparse
import os
import sys
import time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ref"))
import hy4_golden as G  # noqa: E402

torch.set_grad_enabled(False)


def collect_chunked(ckpt, ids_path, layers=None, chunk=4096, need_dn=True):
    ids = G.load_ids(ids_path)
    T = int(ids.shape[0])
    m = G.Hy4(ckpt)
    acc_H = {}
    acc_n = {}
    acc_lat = {}
    acc_top = {}
    orig_moe = m.moe

    def tap_moe(i, x):
        W = m.W
        p = f"model.layers.{i}.mlp."
        logits = G.lin(x, W.get(p + "gate.weight"))
        scores = torch.sigmoid(logits)
        choice = scores + W.get(p + "gate.e_score_correction_bias")
        top = torch.topk(choice, m.topk, -1, sorted=True).indices
        lat = G.lin(x, W.get(p + "routed_expert_down_proj.weight"))
        H = lat.double().T @ lat.double()
        if i not in acc_H:
            acc_H[i] = H
            acc_n[i] = lat.shape[0]
            if need_dn:
                acc_lat[i] = [lat.half()]
                acc_top[i] = [top.to(torch.int16)]
        else:
            acc_H[i] += H
            acc_n[i] += lat.shape[0]
            if need_dn:
                acc_lat[i].append(lat.half())
                acc_top[i].append(top.to(torch.int16))
        return orig_moe(i, x)

    m.moe = tap_moe
    t0 = time.time()
    n_chunks = (T + chunk - 1) // chunk if chunk > 0 else 1
    for ci in range(n_chunks):
        s0 = ci * chunk if chunk > 0 else 0
        s1 = min(T, s0 + chunk) if chunk > 0 else T
        print(f"=== chunk {ci + 1}/{n_chunks} [{s0}:{s1}] ({s1 - s0} tok) ===", file=sys.stderr, flush=True)
        m.forward(ids[s0:s1], keep=1, layers=layers)
    calib = {}
    for i in sorted(acc_H):
        entry = {"H_gu": (acc_H[i] / max(acc_n[i], 1)).float()}
        if need_dn:
            entry["lat"] = torch.cat(acc_lat[i], 0)
            entry["top"] = torch.cat(acc_top[i], 0)
        calib[i] = entry
    print(f"collected GPTQ calib from {ids_path} ({T} tok, {n_chunks} chunks, {len(calib)} MoE layers) in {time.time() - t0:.1f}s",
          file=sys.stderr, flush=True)
    return T, calib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("ids")
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=4096, help="max tokens per forward chunk (0 = whole file)")
    ap.add_argument("--gu-only", action="store_true", help="only store H_gu (smaller file, skips sequential moe.dn GPTQ)")
    a = ap.parse_args()
    T, calib = collect_chunked(a.ckpt, a.ids, layers=a.layers, chunk=a.chunk, need_dn=not a.gu_only)
    torch.save({"source": a.ids, "tokens": T, "layers": calib}, a.out)
    mb = os.path.getsize(a.out) / 1e6
    print(f"saved {a.out} ({mb:.1f} MB, {T} tok, {len(calib)} MoE layers)", flush=True)


if __name__ == "__main__":
    main()
