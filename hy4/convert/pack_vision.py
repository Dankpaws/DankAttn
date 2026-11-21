"""Append Hy4-flash's vision tower (vit.*) to an opus-hy4 pack (or write a standalone vision pack).

  python3 pack_vision.py <bf16 ckpt> <pack>

Env:
  HY4_VIT_BITS (default 8)
  HY4_VIT_GS   (default 128)
  HY4_DEV      (default cpu; set cuda for faster quantization)

Layout (all W = register-direct mm2/mmp weight, R = raw):
  V.patch.w       W8g128  N=1152, K=768   (Conv2d [1152, 3, 16, 16] flattened to [1152, 768])
  V.patch.b       R f32   [1152]
  V.pos_embed     R bf16  [16384, 1152]   (rows 1..16384 of position_embedding, 128x128 grid; row 0 unused CLS)
  V{i}.ln1.w/b    R f32   [1152]          (i = 0..26)
  V{i}.attn.qkv   W8g128  N=3456, K=1152  rows q [0,1152] | k [1152,2304] | v [2304,3456]
  V{i}.attn.qkv.b R f32   [3456]
  V{i}.attn.o     W8g128  N=1152, K=1152
  V{i}.attn.o.b   R f32   [1152]
  V{i}.ln2.w/b    R f32   [1152]
  V{i}.mlp.fc1    W8g128  N=4352, K=1152  (4304 zero-padded to 4352 = 34*128; GELU(0)=0)
  V{i}.mlp.fc1.b  R f32   [4352]          (zero-padded at 4304..4351)
  V{i}.mlp.fc2    W8g128  N=1152, K=4352  (K zero-padded 4304 -> 4352)
  V{i}.mlp.fc2.b  R f32   [1152]
  V.merge.norm    R f32   [1152]          (vit.perceive.norm.weight, RMSNorm)
  V.merge.fc1     W8g128  N=4608, K=4608  (Conv2d [4608, 1152, 2, 2] -> permute(0,2,3,1).reshape(4608,4608))
  V.merge.fc1.b   R f32   [4608]
  V.merge.fc2     W8g128  N=3072, K=4608  (Conv2d [3072, 4608, 1, 1] -> reshape(3072,4608))
  V.merge.fc2.b   R f32   [3072]
  V.merge.newline R f32   [3072]          (vit.perceive.image_newline)
"""
import json, math, os, sys, time
import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "convert"))
import pack as P  # noqa: E402

torch.set_grad_enabled(False)
DEV = os.environ.get("HY4_DEV", "cpu")
_tq_cpu = P.tquant


def tquant_dev(w, bits, gs, clip=True, imp=None):
    q, s = _tq_cpu(w.to(DEV), bits, gs, clip, None if imp is None else imp.to(DEV))
    return q.cpu(), s.cpu()


if DEV != "cpu":
    P.tquant = tquant_dev


def main(src, out):
    t0 = time.time()
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    cfg = json.load(open(os.path.join(src, "config.json")))
    vc = cfg["vision_config"]
    os.makedirs(out, exist_ok=True)
    man_p = os.path.join(out, "manifest.json")
    if os.path.exists(man_p):
        man = json.load(open(man_p))
        if "vision" in man:
            print("pack_vision: manifest already has 'vision'; overwriting vision entries (note: old pack file stays on disk)", flush=True)
    else:
        man = dict(format="opus-hy4-vision-v0", files=[], tensors={})

    handles = {}

    def T(name, dtype=torch.float32):
        f = idx[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(src, f), framework="pt")
        return handles[f].get_tensor(name).to(dtype)

    W = P.Writer(out)
    W.files = list(man["files"])  # continue pack-NN.bin numbering
    rel = {}
    vbits = int(os.environ.get("HY4_VIT_BITS", "8"))
    vgs = int(os.environ.get("HY4_VIT_GS", "128"))
    H = vc["hidden_size"]            # 1152
    I = vc["intermediate_size"]      # 4304
    I_pad = vgs * math.ceil(I / vgs) # 4352 (divisible by 128 and 32)
    nl = vc["num_hidden_layers"]     # 27
    nh = vc["num_attention_heads"]   # 16
    ps = vc["patch_size"]            # 16
    merge = vc["spatial_merge_size"] # 2
    edge = vc["max_image_size"] // ps  # 128
    out_h = vc["out_hidden_size"]    # 3072
    print(f"pack_vision: {nl} layers, H={H}, I={I}->{I_pad}, W{vbits}g{vgs}, dev={DEV}", flush=True)

    def lin(name, w, bits=vbits, gs=vgs, rows=None):
        assert w.shape[0] % 32 == 0 and w.shape[1] % gs == 0, f"{name}: bad shape {tuple(w.shape)}"
        rel[name] = W.linear(name, w.contiguous(), bits, gs, rows=rows)

    # 1. Patch embedding & learned 2D position embedding (skip unused CLS row 0)
    pw = T("vit.embeddings.patch_embedding.weight")  # [1152, 3, 16, 16]
    lin("V.patch.w", pw.reshape(H, -1))              # [1152, 768]
    W.raw("V.patch.b", T("vit.embeddings.patch_embedding.bias").numpy())
    pos_bf16 = T("vit.embeddings.position_embedding.weight", torch.bfloat16)[1:1 + edge * edge].contiguous()
    W.raw("V.pos_embed", pos_bf16.view(torch.uint16).numpy(), dtype="bf16")

    # 2. 27 ViT transformer blocks
    for i in range(nl):
        p = f"vit.layers.{i}."
        d = f"V{i}"
        W.raw(d + ".ln1.w", T(p + "input_layernorm.weight").numpy())
        W.raw(d + ".ln1.b", T(p + "input_layernorm.bias").numpy())
        W.raw(d + ".ln2.w", T(p + "post_attention_layernorm.weight").numpy())
        W.raw(d + ".ln2.b", T(p + "post_attention_layernorm.bias").numpy())

        a = p + "self_attn."
        qkv_w = torch.cat([T(a + "q_proj.weight"), T(a + "k_proj.weight"), T(a + "v_proj.weight")], 0)
        qkv_b = torch.cat([T(a + "q_proj.bias"), T(a + "k_proj.bias"), T(a + "v_proj.bias")], 0)
        lin(d + ".attn.qkv", qkv_w, rows={"q": [0, H], "k": [H, 2 * H], "v": [2 * H, 3 * H]})
        W.raw(d + ".attn.qkv.b", qkv_b.numpy())
        lin(d + ".attn.o", T(a + "o_proj.weight"))
        W.raw(d + ".attn.o.b", T(a + "o_proj.bias").numpy())

        fc1_w = T(p + "mlp.dense_h_to_4h.weight")  # [4304, 1152]
        fc1_b = T(p + "mlp.dense_h_to_4h.bias")    # [4304]
        fc2_w = T(p + "mlp.dense_4h_to_h.weight")  # [1152, 4304]
        fc2_b = T(p + "mlp.dense_4h_to_h.bias")    # [1152]
        if I_pad > I:
            fc1_w = torch.cat([fc1_w, torch.zeros(I_pad - I, H)], 0)
            fc1_b = torch.cat([fc1_b, torch.zeros(I_pad - I)], 0)
            fc2_w = torch.cat([fc2_w, torch.zeros(H, I_pad - I)], 1)
        lin(d + ".mlp.fc1", fc1_w)
        W.raw(d + ".mlp.fc1.b", fc1_b.numpy())
        lin(d + ".mlp.fc2", fc2_w)
        W.raw(d + ".mlp.fc2.b", fc2_b.numpy())
        if (i + 1) % 9 == 0 or i == nl - 1:
            print(f"  V{i} done ({time.time() - t0:.1f}s)", flush=True)

    # 3. Patch merger (vit.perceive.*)
    # Conv2d [4608, 1152, 2, 2] applied to [1152, 2, 2] spatial block == Linear [4608, 4608]
    # when input patches are gathered in (dy, dx, c) order.
    W.raw("V.merge.norm", T("vit.perceive.norm.weight").numpy())
    m0 = T("vit.perceive.proj.0.weight")  # [4608, 1152, 2, 2]
    m0_lin = m0.permute(0, 2, 3, 1).reshape(H * merge * merge, H * merge * merge)
    lin("V.merge.fc1", m0_lin)
    W.raw("V.merge.fc1.b", T("vit.perceive.proj.0.bias").numpy())
    m2 = T("vit.perceive.proj.2.weight").reshape(out_h, H * merge * merge)
    lin("V.merge.fc2", m2)
    W.raw("V.merge.fc2.b", T("vit.perceive.proj.2.bias").numpy())
    W.raw("V.merge.newline", T("vit.perceive.image_newline").numpy())

    W.close()
    new_files = W.files[len(man["files"]):]
    added_bytes = sum(os.path.getsize(os.path.join(out, f)) for f in new_files)
    man["files"] = W.files
    man["tensors"].update(W.tensors)
    man.setdefault("rel_mse", {}).update(rel)
    man["vision"] = dict(
        layers=nl,
        hidden_size=H,
        intermediate_size=I,
        intermediate_padded=I_pad,
        num_heads=nh,
        head_dim=H // nh,
        patch_size=ps,
        spatial_merge_size=merge,
        position_edge=edge,
        out_hidden_size=out_h,
        eps=float(vc.get("rms_norm_eps", 1e-5)),
        bits=vbits,
        gs=vgs,
        image_start_token_id=int(cfg["image_start_token_id"]),
        image_end_token_id=int(cfg["image_end_token_id"]),
        image_token_id=int(cfg["image_token_id"]),
        image_newline_token_id=int(cfg["image_newline_token_id"]),
        video_start_token_id=int(cfg["video_start_token_id"]),
        video_end_token_id=int(cfg["video_end_token_id"]),
        video_token_id=int(cfg["video_token_id"]),
    )
    json.dump(man, open(man_p, "w"), indent=1)
    max_mse = max(rel.values())
    mean_mse = sum(rel.values()) / len(rel)
    print(f"done in {time.time() - t0:.1f}s: wrote {new_files} ({added_bytes / 1e6:.2f} MB), "
          f"W{vbits}g{vgs} rel_mse mean={mean_mse:.3e} max={max_mse:.3e}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
