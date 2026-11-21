"""Reference BF16 vs W8g128 vision tower verification + golden binary dump for Metal ViT.

  python3 hy4/ref/vision_golden.py <bf16 ckpt> <vision pack> [out_golden.bin]

Loads the exact ViT weights from <bf16 ckpt> (BF16) and compares against the dequantized
W8g128 weights unpacked from <vision pack> on a synthetic test image (default 256x256 ->
16x16 patches -> 8x(8+1) = 72 merged visual tokens). Also writes a binary golden file
containing (grid_thw, pixel_values bf16, ref_bf16 f32, ref_w8 f32) for C++/Metal validation.
"""
import json, math, os, struct, sys
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "convert"))
from layout import RK, RN  # noqa: E402

torch.set_grad_enabled(False)


def unpack_w(buf, t):
    """Unpack a register-direct W8/W6/W4 weight from pack bytes back to float32 [N, K]."""
    N, K, bits, gs = t["N"], t["K"], t["bits"], t["gs"]
    NF, KS = N // 32, K // 32
    raw = np.frombuffer(buf[t["file"]], dtype=np.uint8, count=t["bytes"], offset=t["off"])
    sc = np.frombuffer(buf[t["sfile"]], dtype=np.float16, count=t["sbytes"] // 2, offset=t["soff"])
    s = sc.reshape(NF, K // gs, 32).transpose(0, 2, 1).reshape(N, K // gs).astype(np.float32)
    assert bits == 8, f"only W8 unpack implemented here (got {bits})"
    e = raw.view(np.int8).reshape(NF, KS, 32, 32)
    q4 = np.empty((NF, KS, 32, 32), dtype=np.int8)
    q4[:, :, RN, RK] = e
    q = q4.transpose(0, 2, 1, 3).reshape(N, K)
    return torch.from_numpy((q.reshape(N, K // gs, gs).astype(np.float32) * s[..., None]).reshape(N, K))


def load_raw(buf, t):
    if t["dtype"] == "f32":
        a = np.frombuffer(buf[t["file"]], dtype=np.float32, count=t["bytes"] // 4, offset=t["off"]).copy()
        return torch.from_numpy(a).reshape(t["shape"])
    if t["dtype"] == "bf16":
        a = np.frombuffer(buf[t["file"]], dtype=np.uint16, count=t["bytes"] // 2, offset=t["off"]).copy()
        return torch.from_numpy(a).view(torch.bfloat16).reshape(t["shape"]).float()
    raise ValueError(t["dtype"])


def interpolate_pos(pos_grid_128, h, w):
    """Exact match to HYV4FlashVisionPatchEmbed.interpolate_pos_encoding."""
    edge = 128
    # pos_grid_128: [16384, 1152] float32 -> [1, 1152, 128, 128]
    p = pos_grid_128.reshape(1, edge, edge, -1).permute(0, 3, 1, 2).float()
    scale_h = (int(h) + 0.1) / edge
    scale_w = (int(w) + 0.1) / edge
    out = F.interpolate(p, scale_factor=(scale_h, scale_w), mode="bilinear", align_corners=False)
    return out.permute(0, 2, 3, 1).reshape(h * w, -1)


def run_vit_bf16(src, pixel_values, grid_thw):
    """Exact PyTorch reference matching modeling_hy_v4_flash.py."""
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    cfg = json.load(open(os.path.join(src, "config.json")))["vision_config"]
    handles = {}

    def T(name):
        f = idx[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(src, f), framework="pt")
        return handles[f].get_tensor(name).float()

    H, nh, hd = cfg["hidden_size"], cfg["num_attention_heads"], cfg["hidden_size"] // cfg["num_attention_heads"]
    eps = float(cfg.get("rms_norm_eps", 1e-5))
    t, h, w = grid_thw
    # Patch embed: pixel_values [N, 3*16*16]
    pw = T("vit.embeddings.patch_embedding.weight").reshape(H, -1)
    pb = T("vit.embeddings.patch_embedding.bias")
    x = F.linear(pixel_values.float(), pw, pb)
    pos_all = T("vit.embeddings.position_embedding.weight")[1:]  # [16384, 1152]
    x = x + interpolate_pos(pos_all, h, w).repeat(t, 1)

    for i in range(cfg["num_hidden_layers"]):
        p = f"vit.layers.{i}."
        r = x
        xn = F.layer_norm(x, (H,), T(p + "input_layernorm.weight"), T(p + "input_layernorm.bias"), eps)
        a = p + "self_attn."
        q = F.linear(xn, T(a + "q_proj.weight"), T(a + "q_proj.bias")).view(-1, nh, hd).transpose(0, 1)
        k = F.linear(xn, T(a + "k_proj.weight"), T(a + "k_proj.bias")).view(-1, nh, hd).transpose(0, 1)
        v = F.linear(xn, T(a + "v_proj.weight"), T(a + "v_proj.bias")).view(-1, nh, hd).transpose(0, 1)
        attn = F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), is_causal=False)[0]
        attn = attn.transpose(0, 1).contiguous().view(-1, H)
        x = r + F.linear(attn, T(a + "o_proj.weight"), T(a + "o_proj.bias"))

        r = x
        xn = F.layer_norm(x, (H,), T(p + "post_attention_layernorm.weight"), T(p + "post_attention_layernorm.bias"), eps)
        m = F.gelu(F.linear(xn, T(p + "mlp.dense_h_to_4h.weight"), T(p + "mlp.dense_h_to_4h.bias")))
        x = r + F.linear(m, T(p + "mlp.dense_4h_to_h.weight"), T(p + "mlp.dense_4h_to_h.bias"))

    # Patch merger
    rms_w = T("vit.perceive.norm.weight")
    xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * rms_w
    xn = xn.reshape(t, h, w, H).permute(0, 3, 1, 2)  # [t, 1152, h, w]
    m0_w = T("vit.perceive.proj.0.weight")
    m0_b = T("vit.perceive.proj.0.bias")
    m2_w = T("vit.perceive.proj.2.weight")
    m2_b = T("vit.perceive.proj.2.bias")
    y = F.conv2d(xn, m0_w, m0_b, stride=2)
    y = F.gelu(y)
    y = F.conv2d(y, m2_w, m2_b)  # [t, 3072, h/2, w/2]
    nl = T("vit.perceive.image_newline").reshape(1, -1, 1, 1).expand(t, -1, h // 2, 1)
    y = torch.cat([y, nl], dim=-1)  # [t, 3072, h/2, w/2 + 1]
    return y.reshape(t, y.shape[1], -1).permute(0, 2, 1).reshape(-1, y.shape[1])


def run_vit_pack(pack_dir, pixel_values, grid_thw):
    """Run using unpacked W8g128 weights from our DankAttn vision pack."""
    man = json.load(open(os.path.join(pack_dir, "manifest.json")))
    vc = man["vision"]
    ts = man["tensors"]
    bufs = [open(os.path.join(pack_dir, f), "rb").read() for f in man["files"]]
    H, nh, hd = vc["hidden_size"], vc["num_heads"], vc["head_dim"]
    eps = vc["eps"]
    t, h, w = grid_thw

    x = F.linear(pixel_values.bfloat16().float(), unpack_w(bufs, ts["V.patch.w"]), load_raw(bufs, ts["V.patch.b"]))
    pos_grid = load_raw(bufs, ts["V.pos_embed"])
    x = x + interpolate_pos(pos_grid, h, w).repeat(t, 1)

    for i in range(vc["layers"]):
        d = f"V{i}"
        r = x
        xn = F.layer_norm(x, (H,), load_raw(bufs, ts[d + ".ln1.w"]), load_raw(bufs, ts[d + ".ln1.b"]), eps)
        qkv = F.linear(xn.bfloat16().float(), unpack_w(bufs, ts[d + ".attn.qkv"]), load_raw(bufs, ts[d + ".attn.qkv.b"]))
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(-1, nh, hd).transpose(0, 1)
        k = k.view(-1, nh, hd).transpose(0, 1)
        v = v.view(-1, nh, hd).transpose(0, 1)
        attn = F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), is_causal=False)[0]
        attn = attn.transpose(0, 1).contiguous().view(-1, H)
        x = r + F.linear(attn.bfloat16().float(), unpack_w(bufs, ts[d + ".attn.o"]), load_raw(bufs, ts[d + ".attn.o.b"]))

        r = x
        xn = F.layer_norm(x, (H,), load_raw(bufs, ts[d + ".ln2.w"]), load_raw(bufs, ts[d + ".ln2.b"]), eps)
        m = F.gelu(F.linear(xn.bfloat16().float(), unpack_w(bufs, ts[d + ".mlp.fc1"]), load_raw(bufs, ts[d + ".mlp.fc1.b"])))
        x = r + F.linear(m.bfloat16().float(), unpack_w(bufs, ts[d + ".mlp.fc2"]), load_raw(bufs, ts[d + ".mlp.fc2.b"]))

    # Patch merger using linear gather (dy, dx, c)
    xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * load_raw(bufs, ts["V.merge.norm"])
    xn = xn.reshape(t, h // 2, 2, w // 2, 2, H).permute(0, 1, 3, 2, 4, 5).reshape(t * (h // 2) * (w // 2), 4 * H)
    y = F.gelu(F.linear(xn.bfloat16().float(), unpack_w(bufs, ts["V.merge.fc1"]), load_raw(bufs, ts["V.merge.fc1.b"])))
    y = F.linear(y.bfloat16().float(), unpack_w(bufs, ts["V.merge.fc2"]), load_raw(bufs, ts["V.merge.fc2.b"]))
    y = y.reshape(t, h // 2, w // 2, -1)
    nl = load_raw(bufs, ts["V.merge.newline"]).reshape(1, 1, 1, -1).expand(t, h // 2, 1, -1)
    y = torch.cat([y, nl], dim=2).reshape(-1, y.shape[-1])
    return y


def main():
    src, pack_dir = sys.argv[1], sys.argv[2]
    out_bin = sys.argv[3] if len(sys.argv) > 3 else None
    torch.manual_seed(42)
    t, h, w = 1, 16, 16  # 256x256 image -> 256 patches -> 8*(8+1) = 72 merged tokens
    pixel_values = torch.randn(t * h * w, 3 * 16 * 16, dtype=torch.bfloat16)

    y_ref = run_vit_bf16(src, pixel_values, (t, h, w))
    y_w8 = run_vit_pack(pack_dir, pixel_values, (t, h, w))
    diff = (y_ref - y_w8).abs()
    cos = F.cosine_similarity(y_ref.reshape(-1), y_w8.reshape(-1), dim=0).item()
    rel_l2 = (y_ref - y_w8).norm().item() / y_ref.norm().item()
    print(f"ViT output shape: {tuple(y_ref.shape)}")
    print(f"BF16 vs W8g128: cos={cos:.6f}, rel_L2={rel_l2:.5f}, max_abs={diff.max().item():.5f}, mean_abs={diff.mean().item():.5f}")

    if out_bin:
        with open(out_bin, "wb") as f:
            f.write(struct.pack("<4I", t, h, w, y_ref.shape[0]))
            f.write(pixel_values.view(torch.uint16).numpy().tobytes())
            f.write(y_ref.float().numpy().tobytes())
            f.write(y_w8.float().numpy().tobytes())
        print(f"Wrote golden dump to {out_bin} ({os.path.getsize(out_bin)} bytes)")


if __name__ == "__main__":
    main()
