"""Pack incoai/Inkling-2-Mini-DFlash2 as a standalone overlay (pack-df2-00.bin + dflash2_manifest.json)
inside an existing DankAttn inkling2 pack directory (coexists with DSpark mtp_manifest.json).

Usage:
  python3 inkling2/convert/pack_dflash2.py /path/to/Inkling-2-Mini-DFlash2 /path/to/pack-inkling2-mini-w8
"""
import argparse
import json
import os
import sys
import time
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "convert"))
import pack as P  # noqa: E402

torch.set_grad_enabled(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="Inkling-2-Mini-DFlash2 checkpoint dir (config.json + model.safetensors)")
    ap.add_argument("out", help="DankAttn pack dir to write pack-df2-00.bin + dflash2_manifest.json into")
    ap.add_argument("--mlp-bits", type=int, default=6, choices=[6, 8])
    args = ap.parse_args()

    t0 = time.time()
    cfg = json.load(open(os.path.join(args.src, "config.json")))
    dc = cfg["dflash_config"]
    main_man = json.load(open(os.path.join(args.out, "manifest.json")))
    v_valid = int(main_man["text_config"]["unpadded_vocab_size"])
    v_pad = (v_valid + 31) & ~31

    W = P.Writer(args.out, max_file=8 << 30)

    def _open_df2():
        if W.cur:
            W.cur.close()
        name = f"pack-df2-{len(W.files):02d}.bin"
        W.files.append(name)
        W.cur = open(os.path.join(W.out, name), "wb")
        W.size = 0

    W._open = _open_df2

    def raw_f32(name, t):
        W.raw(name, t.float().contiguous().numpy(), dtype="f32")

    def raw_bf16(name, t):
        W.raw(name, t.to(torch.bfloat16).contiguous().view(torch.uint16).numpy(), dtype="bf16")

    st_path = os.path.join(args.src, "model.safetensors")
    with safe_open(st_path, framework="pt", device="cpu") as f:
        W.linear("df2.fc", f.get_tensor("fc.weight").float(), 8, 128)
        raw_f32("df2.hidden_norm", f.get_tensor("hidden_norm.weight"))
        raw_f32("df2.norm", f.get_tensor("norm.weight"))

        for l in range(cfg["num_hidden_layers"]):
            p = f"layers.{l}"
            dp = f"df2.L{l}"
            raw_f32(f"{dp}.attn_norm", f.get_tensor(f"{p}.input_layernorm.weight"))
            raw_f32(f"{dp}.mlp_norm", f.get_tensor(f"{p}.post_attention_layernorm.weight"))
            raw_f32(f"{dp}.q_norm", f.get_tensor(f"{p}.self_attn.q_norm.weight"))
            raw_f32(f"{dp}.k_norm", f.get_tensor(f"{p}.self_attn.k_norm.weight"))

            q = f.get_tensor(f"{p}.self_attn.q_proj.weight").float()
            k = f.get_tensor(f"{p}.self_attn.k_proj.weight").float()
            v = f.get_tensor(f"{p}.self_attn.v_proj.weight").float()
            W.linear(f"{dp}.attn.qkv", torch.cat([q, k, v], dim=0), 8, 128)
            W.linear(f"{dp}.attn.kv", torch.cat([k, v], dim=0), 8, 128)
            W.linear(f"{dp}.attn.o", f.get_tensor(f"{p}.self_attn.o_proj.weight").float(), 8, 128)

            raw_f32(f"{dp}.attn_conv.base", f.get_tensor(f"{p}.attention_conv.base_kernel"))
            W.linear(f"{dp}.attn_conv.kproj", f.get_tensor(f"{p}.attention_conv.kernel_projection.weight").float(), 8, 128)

            g = f.get_tensor(f"{p}.mlp.gate_proj.weight").float()
            u = f.get_tensor(f"{p}.mlp.up_proj.weight").float()
            gu = torch.cat([g, u], dim=0)[P.gu_rows(g.shape[0])]
            W.linear(f"{dp}.mlp.gu", gu, args.mlp_bits, 128)
            W.linear(f"{dp}.mlp.dn", f.get_tensor(f"{p}.mlp.down_proj.weight").float(), args.mlp_bits, 128)

            raw_f32(f"{dp}.mlp_conv.base", f.get_tensor(f"{p}.mlp_conv.base_kernel"))
            W.linear(f"{dp}.mlp_conv.kproj", f.get_tensor(f"{p}.mlp_conv.kernel_projection.weight").float(), 8, 128)

        W.linear("df2.sel.hproj", f.get_tensor("candidate_selector.hidden_projection.weight").float(), 8, 128)
        raw_bf16("df2.sel.pred_cb", f.get_tensor("candidate_selector.predecessor_codebook")[:v_valid])
        raw_bf16("df2.sel.succ_cb", f.get_tensor("candidate_selector.successor_codebook")[:v_valid])

    if W.cur:
        W.cur.close()

    man = {
        "format": "dankattn-inkling2-dflash2-v1",
        "dflash2_config": {
            "num_hidden_layers": cfg["num_hidden_layers"],
            "hidden_size": cfg["hidden_size"],
            "intermediate_size": cfg["intermediate_size"],
            "num_attention_heads": cfg["num_attention_heads"],
            "num_key_value_heads": cfg["num_key_value_heads"],
            "head_dim": cfg["head_dim"],
            "sliding_window": cfg["sliding_window"],
            "rope_theta": cfg["rope_parameters"]["rope_theta"],
            "rms_norm_eps": cfg["rms_norm_eps"],
            "block_size": dc["block_size"],
            "mask_token_id": dc["mask_token_id"],
            "conv_kernel_size": dc["conv_kernel_size"],
            "conv_group_size": dc["conv_group_size"],
            "selector_rank": dc["selector_rank"],
            "selector_top_k": dc["selector_top_k"],
            "final_logit_softcapping": dc["final_logit_softcapping"],
            "selector_softcap_mode": dc.get("selector_softcap_mode", "after_correction"),
            "target_layer_ids": dc["target_layer_ids"],
        },
        "vocab_valid": v_valid,
        "vocab_pad": v_pad,
        "files": W.files,
        "tensors": W.tensors,
    }
    with open(os.path.join(args.out, "dflash2_manifest.json"), "w") as f:
        json.dump(man, f, indent=2)

    mb = sum(os.path.getsize(os.path.join(args.out, s)) for s in W.files) / 1e6
    print(f"Packed DFlash2 overlay: {mb:.1f} MB ({W.files}) in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
