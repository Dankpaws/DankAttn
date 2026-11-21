"""A small random qwen4_exp checkpoint (HF names, f32 safetensors) whose weights the
converter reproduces exactly: every group of every linear weight is int codes times an
f16 scale, with a +-qmax code in each group (so the requantized scale is the same).
Golden (ref/golden.py) on this checkpoint is then exactly the model the engine runs,
up to activation rounding.
"""
import json, os, shutil, sys
import numpy as np
import torch
from safetensors.torch import save_file

src, out = sys.argv[1], sys.argv[2]
nlayers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
os.makedirs(out, exist_ok=True)
cfg = json.load(open(os.path.join(src, "config.json")))
tc = cfg["text_config"]
tc["num_hidden_layers"] = nlayers
tc["layer_types"] = ["linear_attention"] * (nlayers - 1) + ["full_attention"]
tc["num_experts"] = 16
tc["ple_layer_ids"] = []
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"]:
    if os.path.exists(os.path.join(src, f)):
        shutil.copy(os.path.join(src, f), out)

rng = np.random.default_rng(0)
H, HC, R, E, I = tc["hidden_size"], tc["hc_count"], tc["hc_lowrank"], tc["num_experts"], tc["moe_intermediate_size"]


def qweight(N, K, bits, gs, std, pow2=False):
    qmax = 7 if bits == 4 else 127
    q = rng.integers(-qmax, qmax + 1, size=(N, K // gs, gs))
    pos = rng.integers(0, gs, size=(N, K // gs))
    sign = np.where(rng.random((N, K // gs)) < 0.5, -1, 1)
    np.put_along_axis(q, pos[..., None], (sign * qmax)[..., None], -1)
    s = (std * np.sqrt(3) / qmax * (0.5 + rng.random((N, K // gs)))).astype(np.float16).astype(np.float32)
    if pow2:  # bf16-exact products (experts are loaded as bf16 by the converter)
        s = np.exp2(np.round(np.log2(s))).astype(np.float32)
    return torch.from_numpy((q * s[..., None]).reshape(N, K).astype(np.float32))


def vec(n, scale=0.1):
    return torch.from_numpy((rng.standard_normal(n) * scale).astype(np.float32))


T = {}
pre = "model.language_model."
T[pre + "embed_tokens.weight"] = torch.from_numpy(rng.standard_normal((tc["vocab_size"], H)).astype(np.float32)).bfloat16().float()


def hc(p, combine=True):
    T[p + "hc_norm.weight"] = vec(H * HC)
    T[p + "input_mix_weight_down.weight"] = qweight(R, H * HC, 8, 128, 0.02)
    T[p + "input_mix_weight_up.weight"] = qweight(H * HC, R, 8, 64, 0.05)
    if combine:
        T[p + "block_inject_weight.weight"] = qweight(HC, H * HC, 8, 128, 0.02)


nk, nv, dk, dv = tc["linear_num_key_heads"], tc["linear_num_value_heads"], tc["linear_key_head_dim"], tc["linear_value_head_dim"]
nh, nkv, hd = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
inh, ihd = tc["indexer_n_heads"], tc["indexer_head_dim"]
for L in range(nlayers):
    p = f"{pre}layers.{L}."
    hc(p + "attn_hyper_connection.")
    hc(p + "mlp_hyper_connection.")
    if tc["layer_types"][L] == "linear_attention":
        a = p + "linear_attn."
        T[a + "in_proj_qkv.weight"] = qweight(2 * nk * dk + nv * dv, H, 8, 128, 0.03)
        T[a + "in_proj_z.weight"] = qweight(nv * dv, H, 8, 128, 0.03)
        T[a + "in_proj_b.weight"] = qweight(nv, H, 8, 128, 0.03)
        T[a + "in_proj_a.weight"] = qweight(nv, H, 8, 128, 0.03)
        T[a + "out_proj.weight"] = qweight(H, nv * dv, 8, 128, 0.02)
        T[a + "conv1d.weight"] = torch.from_numpy((rng.standard_normal((2 * nk * dk + nv * dv, 1, 4)) * 0.3).astype(np.float32))
        T[a + "A_log"] = torch.from_numpy(np.log(rng.uniform(1, 16, nv)).astype(np.float32))
        T[a + "dt_bias"] = torch.ones(nv)
        T[a + "norm.weight"] = 1 + vec(dv)
    else:
        a = p + "self_attn."
        T[a + "q_proj.weight"] = qweight(nh * hd * 2, H, 8, 128, 0.03)
        T[a + "k_proj.weight"] = qweight(nkv * hd, H, 8, 128, 0.03)
        T[a + "v_proj.weight"] = qweight(nkv * hd, H, 8, 128, 0.03)
        T[a + "o_proj.weight"] = qweight(H, nh * hd, 8, 128, 0.02)
        T[a + "indexer.index_qk_proj.weight"] = qweight((inh + 1) * ihd, H, 8, 128, 0.03)
        for k, d in [("q_norm", hd), ("k_norm", hd), ("indexer.q_layernorm", ihd), ("indexer.k_layernorm", ihd)]:
            T[a + k + ".weight"] = vec(d)
    m = p + "mlp."
    T[m + "gate.weight"] = qweight(E, H, 8, 128, 0.05)
    T[m + "shared_expert.gate_proj.weight"] = qweight(I, H, 8, 128, 0.03)
    T[m + "shared_expert.up_proj.weight"] = qweight(I, H, 8, 128, 0.03)
    T[m + "shared_expert.down_proj.weight"] = qweight(H, I, 8, 128, 0.03)
    T[m + "shared_expert_gate.weight"] = qweight(1, H, 8, 128, 0.03)
    T[m + "experts.gate_up_proj"] = torch.stack([qweight(2 * I, H, 4, 128, 0.03, True) for _ in range(E)])
    T[m + "experts.down_proj"] = torch.stack([qweight(H, I, 4, 128, 0.03, True) for _ in range(E)])
hc(pre + "hyper_connection_mixer.", combine=False)
# MTP head: one attention layer + MoE, input fusion, own mixer
mp = "mtp.layers.0."
hc(mp + "attn_hyper_connection.")
hc(mp + "mlp_hyper_connection.")
a = mp + "self_attn."
T[a + "q_proj.weight"] = qweight(nh * hd * 2, H, 8, 128, 0.03)
T[a + "k_proj.weight"] = qweight(nkv * hd, H, 8, 128, 0.03)
T[a + "v_proj.weight"] = qweight(nkv * hd, H, 8, 128, 0.03)
T[a + "o_proj.weight"] = qweight(H, nh * hd, 8, 128, 0.02)
T[a + "indexer.index_qk_proj.weight"] = qweight((inh + 1) * ihd, H, 8, 128, 0.03)
for k, d in [("q_norm", hd), ("k_norm", hd), ("indexer.q_layernorm", ihd), ("indexer.k_layernorm", ihd)]:
    T[a + k + ".weight"] = vec(d)
m = mp + "mlp."
T[m + "gate.weight"] = qweight(E, H, 8, 128, 0.05)
T[m + "shared_expert.gate_proj.weight"] = qweight(I, H, 8, 128, 0.03)
T[m + "shared_expert.up_proj.weight"] = qweight(I, H, 8, 128, 0.03)
T[m + "shared_expert.down_proj.weight"] = qweight(H, I, 8, 128, 0.03)
T[m + "shared_expert_gate.weight"] = qweight(1, H, 8, 128, 0.03)
T[m + "experts.gate_up_proj"] = torch.stack([qweight(2 * I, H, 4, 128, 0.03, True) for _ in range(E)])
T[m + "experts.down_proj"] = torch.stack([qweight(H, I, 4, 128, 0.03, True) for _ in range(E)])
hc("mtp.hyper_connection_mixer.", combine=False)
T["mtp.fc_embedding.weight"] = qweight(H, H, 8, 128, 0.03)
T["mtp.fc_hidden.weight"] = qweight(H, H, 8, 128, 0.03)
T["mtp.pre_fc_norm_embedding.weight"] = vec(H)
T["mtp.pre_fc_norm_hidden.weight"] = vec(H * HC)
T["lm_head.weight"] = qweight(tc["vocab_size"], H, 8, 128, 0.02)
save_file({k: v.contiguous() for k, v in T.items()}, os.path.join(out, "model.safetensors"))
json.dump({"weight_map": {k: "model.safetensors" for k in T}}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
print("wrote", len(T), "tensors to", out)
