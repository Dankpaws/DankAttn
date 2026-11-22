"""Pack Inkling-2-Mini bundled DSpark MTP drafter (mtp.safetensors) as a standalone overlay
(pack-mtp.bin + mtp_manifest.json) inside an existing DankAttn inkling2 pack directory.

Precomputes:
  - fuse_alpha = softmax(fuse_weight, dim=-1) [4, 4] f32
  - markov_w1 as raw BF16 [V, 256] for O(1) token embedding lookup
  - folded Markov head: fold_w = lm_head @ hidden_out.weight [V_pad, 256] W8 (~51 MB)
                        fold_b = lm_head @ hidden_out.bias   [V_pad] f32 (~800 KB)

Usage:
  python3 inkling2/convert/pack_mtp.py /path/to/Inkling-2-Mini /path/to/pack-inkling2-mini-w8
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "convert"))
import pack as P  # noqa: E402

torch.set_grad_enabled(False)


def pad_rows_32(w: torch.Tensor) -> torch.Tensor:
    rem = w.shape[0] % 32
    if rem == 0:
        return w
    return torch.cat([w, torch.zeros(32 - rem, w.shape[1], dtype=w.dtype)], dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="Inkling-2-Mini checkpoint dir (with mtp.safetensors & model safetensors)")
    ap.add_argument("out", help="DankAttn pack dir to write pack-mtp.bin + mtp_manifest.json into")
    ap.add_argument("--mlp-bits", type=int, default=6, choices=[6, 8])
    args = ap.parse_args()

    t0 = time.time()
    cfg = json.load(open(os.path.join(args.src, "config.json")))
    tc = cfg["text_config"]
    mc = cfg["mtp_config"]
    v_valid = tc["unpadded_vocab_size"]
    v_pad = (v_valid + 31) & ~31

    # Load unembed from main checkpoint for exact Markov head fold
    idx = json.load(open(os.path.join(args.src, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(args.src, idx["model.llm.unembed.weight"]), framework="pt", device="cpu") as fm:
        lm_w = pad_rows_32(fm.get_tensor("model.llm.unembed.weight")[:v_valid].float())

    # Custom writer targeting pack-mtp-00.bin
    W = P.Writer(args.out, max_file=8 << 30)
    def _open_mtp():
        if W.cur:
            W.cur.close()
        name = f"pack-mtp-{len(W.files):02d}.bin"
        W.files.append(name)
        W.cur = open(os.path.join(W.out, name), "wb")
        W.size = 0
    W._open = _open_mtp

    def raw_f32(name, t): W.raw(name, t.float().numpy(), dtype="f32")
    def raw_bf16(name, t): W.raw(name, t.to(torch.bfloat16).view(torch.uint16).numpy(), dtype="bf16")

    mtp_path = os.path.join(args.src, "mtp.safetensors")
    with safe_open(mtp_path, framework="pt", device="cpu") as f:
        fw = f.get_tensor("fuse_weight").float()
        fuse_alpha = torch.softmax(fw, dim=-1)
        raw_f32("mtp.fuse_alpha", fuse_alpha)
        raw_bf16("mtp.mask_embed", f.get_tensor("mask_embed"))
        raw_f32("mtp.norm", f.get_tensor("norm.weight"))

        for l in range(mc["num_hidden_layers"]):
            p = f"layers.{l}"
            dp = f"mtp.L{l}"
            raw_f32(f"{dp}.dflare_norm", f.get_tensor(f"{p}.dflare_norm.weight"))
            raw_f32(f"{dp}.attn_norm", f.get_tensor(f"{p}.input_layernorm.weight"))
            raw_f32(f"{dp}.mlp_norm", f.get_tensor(f"{p}.post_attention_layernorm.weight"))
            raw_f32(f"{dp}.q_norm", f.get_tensor(f"{p}.self_attn.q_norm.weight"))
            raw_f32(f"{dp}.k_norm", f.get_tensor(f"{p}.self_attn.k_norm.weight"))
            raw_f32(f"{dp}.k_norm_ctx", f.get_tensor(f"{p}.self_attn.k_norm_ctx.weight"))

            qkv = torch.cat([
                f.get_tensor(f"{p}.self_attn.q_proj.weight").float(),
                f.get_tensor(f"{p}.self_attn.k_proj.weight").float(),
                f.get_tensor(f"{p}.self_attn.v_proj.weight").float(),
            ], dim=0)
            W.linear(f"{dp}.attn.qkv", qkv, 8, 128)

            kv_ctx = torch.cat([
                f.get_tensor(f"{p}.self_attn.k_proj_ctx.weight").float(),
                f.get_tensor(f"{p}.self_attn.v_proj_ctx.weight").float(),
            ], dim=0)
            W.linear(f"{dp}.attn.kv_ctx", kv_ctx, 8, 128)
            W.linear(f"{dp}.attn.o", f.get_tensor(f"{p}.self_attn.o_proj.weight").float(), 8, 128)

            g = f.get_tensor(f"{p}.mlp.gate_proj.weight").float()
            u = f.get_tensor(f"{p}.mlp.up_proj.weight").float()
            gu = torch.cat([g, u], dim=0)[P.gu_rows(g.shape[0])]
            W.linear(f"{dp}.mlp.gu", gu, args.mlp_bits, 128)
            W.linear(f"{dp}.mlp.dn", f.get_tensor(f"{p}.mlp.down_proj.weight").float(), args.mlp_bits, 128)

        # Markov RNN head + folded lm_head projection
        raw_f32("mtp.mh.dflare_norm", f.get_tensor("markov_head.dflare_norm.weight"))
        raw_bf16("mtp.mh.markov_w1", f.get_tensor("markov_head.markov_w1.weight"))
        raw_bf16("mtp.mh.rnn_proj.w", f.get_tensor("markov_head.rnn_proj.weight"))
        raw_f32("mtp.mh.rnn_proj.b", f.get_tensor("markov_head.rnn_proj.bias"))

        ho_w = f.get_tensor("markov_head.hidden_out.weight").double()  # [2304, 256]
        ho_b = f.get_tensor("markov_head.hidden_out.bias").double()    # [2304]
        print("Computing folded Markov LM head [200064, 256]...", flush=True)
        fold_w = torch.empty((v_pad, ho_w.shape[1]), dtype=torch.float32)
        fold_b = torch.empty((v_pad,), dtype=torch.float32)
        for s0 in range(0, v_pad, 8192):
            s1 = min(s0 + 8192, v_pad)
            hd_chunk = lm_w[s0:s1].double()
            fold_w[s0:s1] = (hd_chunk @ ho_w).float()
            fold_b[s0:s1] = (hd_chunk @ ho_b).float()
        W.linear("mtp.mh.fold_w", fold_w, 8, 128)
        raw_f32("mtp.mh.fold_b", fold_b)

    if W.cur:
        W.cur.close()

    man = {
        "format": "dankattn-inkling2-mtp-v1",
        "mtp_config": mc,
        "vocab_valid": v_valid,
        "vocab_pad": v_pad,
        "files": W.files,
        "tensors": W.tensors,
    }
    with open(os.path.join(args.out, "mtp_manifest.json"), "w") as f:
        json.dump(man, f, indent=2)

    mb = sum(os.path.getsize(os.path.join(args.out, s)) for s in W.files) / 1e6
    print(f"Packed DSpark overlay: {mb:.1f} MB ({W.files}) in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
