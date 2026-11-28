// ---------------------------------------------------------------- step encoding
// (included by model.mm)

void Model::set_rope(uint32_t p0, uint32_t M, int slot) {
  rope = rope_slot[slot];
  rope_blk = rope_blk_slot[slot];
  std::vector<uint32_t> ps(M), pb(M);
  for (uint32_t r = 0; r < M; r++) {
    ps[r] = p0 + r;
    pb[r] = ((p0 + r) / ratio) * ratio;
  }
  rope_rows(rope_theta, rot, ps, (float*)rope.contents);
  std::vector<float> tmp(2 * M * rot);
  rope_rows(rope_theta, rot, pb, tmp.data());
  float* rb = (float*)rope_blk.contents;
  for (uint32_t r = 0; r < M; r++) {
    memcpy(rb + r * 2 * rot, tmp.data() + r * rot, rot * 4);
    memcpy(rb + r * 2 * rot + rot, tmp.data() + (M + r) * rot, rot * 4);
  }
}

void Model::copy(Rec& rc, id<MTLBuffer> src, uint32_t src0, id<MTLBuffer> dst, uint32_t dst0, uint32_t rows, uint32_t n) {
  struct { uint32_t n, src0, dst0, rows; } a{n, src0, dst0, rows};
  copy_enc(rc.op("copy"), src, src0, dst, dst0, rows, n);
}

void Model::copy_enc(id<MTLComputeCommandEncoder> e, id<MTLBuffer> src, uint32_t src0, id<MTLBuffer> dst,
                     uint32_t dst0, uint32_t rows, uint32_t n) {
  struct { uint32_t n, src0, dst0, rows; } b{n, src0, dst0, rows};
  [e setComputePipelineState:k_copy];
  [e setBuffer:src offset:0 atIndex:0];
  [e setBuffer:dst offset:0 atIndex:1];
  [e setBytes:&b length:sizeof b atIndex:2];
  [e dispatchThreads:MTLSizeMake(n, rows, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// One decoder layer: [PLE] -> attention/GDN -> MoE. h holds the residual streams; the MoE
// output is left pending in moe_o (injected by whichever hc_pre runs next).
void Model::encode_layer(Rec& rc, Layer& l, uint32_t M, uint32_t p0, int mode, bool& pending) {
  id<MTLComputeCommandEncoder> e;
  const size_t li = size_t(&l - L.data());  // only meaningful for trunk layers (joint steps)
  if (l.ple && ple_wait) {
    if (rc.e) [rc.e endEncoding];
    rc.e = nil;
    [rc.cb encodeWaitForEvent:ple_event value:ple_wait];
  }
  if (l.ple) {
    if (pending) {  // the previous MoE output must land in h before PLE reads it
      HcArgs ha{M, H, HC, 1, prev_dn_N, R, topk};
      e = rc.op("hc_pre");
      [e setComputePipelineState:k_hc_pre];
      [e setBuffer:h offset:0 atIndex:0];
      [e setBuffer:moe_o offset:0 atIndex:1];
      [e setBuffer:ydn offset:0 atIndex:2];
      [e setBuffer:l.ahc_norm.b offset:l.ahc_norm.off atIndex:3];
      [e setBuffer:hn offset:0 atIndex:4];
      [e setBuffer:hnf offset:0 atIndex:5];
      [e setBytes:&ha length:sizeof ha atIndex:6];
      [e setBuffer:yd offset:0 atIndex:7];
      [e setBuffer:ysh offset:0 atIndex:8];
      [e setBuffer:topi offset:0 atIndex:9];
      [e setBuffer:topw offset:0 atIndex:10];
      [e setBuffer:slot_of offset:0 atIndex:11];
      [e setBuffer:sgate offset:0 atIndex:12];
      [e dispatchThreadgroups:MTLSizeMake(M, HC, 1 + pf_ride(e, M, HC, hct, 0)) threadsPerThreadgroup:MTLSizeMake(hct, 1, 1)];
      pending = false;
    }
    mm(rc, "mm.ple.kv", l.ple_kv, ple_e, yin, M);
    PleArgs pa{M, 1, l.ple_kv.N, H, HC, ple_par, 9, 3, 4, uint32_t(mode)};
    e = rc.op("ple_gate");
    [e setComputePipelineState:k_ple_gate];
    [e setBuffer:yin offset:0 atIndex:0];
    [e setBuffer:h offset:0 atIndex:1];
    [e setBuffer:l.ple_nk.b offset:l.ple_nk.off atIndex:2];
    [e setBuffer:l.ple_nq.b offset:l.ple_nq.off atIndex:3];
    [e setBuffer:l.ple_nc.b offset:l.ple_nc.off atIndex:4];
    [e setBuffer:ple_gv offset:0 atIndex:5];
    [e setBuffer:ple_gvn offset:0 atIndex:6];
    [e setBytes:&pa length:sizeof pa atIndex:7];
    [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    if (!jl.empty()) encode_ple_lanes(rc, l, li, mode); else {
    e = rc.op("ple_conv");
    [e setComputePipelineState:k_ple_conv];
    [e setBuffer:h offset:0 atIndex:0];
    [e setBuffer:ple_gv offset:0 atIndex:1];
    [e setBuffer:ple_gvn offset:0 atIndex:2];
    [e setBuffer:l.plehist offset:0 atIndex:3];
    [e setBuffer:l.ple_conv.b offset:l.ple_conv.off atIndex:4];
    [e setBytes:&pa length:sizeof pa atIndex:5];
    [e dispatchThreads:MTLSizeMake(H * HC, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
  }
  // ride-along prefetch of mm.in's weights: the previous layer's route and comb read the head,
  // this layer's first hc_pre continues from there
  if (M <= MAXM && pf_hc) pf_set(l.in, &l != L.data() ? pf_route + pf_comb : 0);
  pf_take = pf_hc;
  hc_block(rc, l.ahc_dn, l.ahc_up, l.ahc_norm, M, pending ? 1 : 0, moe_o);
  pf = {};
  pending = false;
  if (!l.attn) {
    id<MTLBuffer> gi = M <= MAXM ? l.gin : yin;
    mm(rc, "mm.in", l.in, xf, gi, M);
    if (!jl.empty() && lane_batch) encode_gdn_lanes_b(rc, l, li, mode); else if (!jl.empty()) encode_gdn_lanes(rc, l, li, mode); else {
    GdnArgs ga{M, 1, l.in.N, nk, nv, dk, dv, conv_par, uint32_t(mode)};
    encode_gdn(rc, l, gi, ga, M);
    }
  } else {
    mm(rc, "mm.in", l.in, xf, yin, M);
    if (!jl.empty() && lane_batch) encode_attn_lanes_b(rc, l, li); else if (!jl.empty()) encode_attn_lanes(rc, l, li); else {
    AttnArgs aa{M, 1, l.in.N, nh, nkv, hd, inh, ihd, rot, p0, ratio, cap, budget, 0, kv8};
    e = rc.op("attn_prep");
    [e setComputePipelineState:k_attn_prep];
    [e setBuffer:yin offset:0 atIndex:0];
    [e setBuffer:rope offset:0 atIndex:1];
    [e setBuffer:l.q_norm.b offset:l.q_norm.off atIndex:2];
    [e setBuffer:l.k_norm.b offset:l.k_norm.off atIndex:3];
    [e setBuffer:l.iq_norm.b offset:l.iq_norm.off atIndex:4];
    [e setBuffer:l.ik_norm.b offset:l.ik_norm.off atIndex:5];
    [e setBuffer:q offset:0 atIndex:6];
    [e setBuffer:gate offset:0 atIndex:7];
    [e setBuffer:l.Kc offset:0 atIndex:8];
    [e setBuffer:l.Vc offset:0 atIndex:9];
    [e setBuffer:iq offset:0 atIndex:10];
    [e setBuffer:l.rawk offset:0 atIndex:11];
    [e setBuffer:l.bk offset:0 atIndex:12];
    [e setBuffer:rope_blk offset:0 atIndex:13];
    [e setBytes:&aa length:sizeof aa atIndex:14];
    [e dispatchThreadgroups:MTLSizeMake(M, nh + nkv + inh, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    bool completes = false, sparse = false;
    for (uint32_t r = 0; r < M; r++) {
      completes |= (p0 + r + 1) % ratio == 0;
      sparse |= (p0 + r + 1) / ratio > budget / ratio;
    }
    if (completes) {
    e = rc.op("attn_blocks");
    [e setComputePipelineState:k_attn_blocks];
    [e setBuffer:l.rawk offset:0 atIndex:0];
    [e setBuffer:l.ik_norm.b offset:l.ik_norm.off atIndex:1];
    [e setBuffer:l.bk offset:0 atIndex:2];
    [e setBuffer:rope_blk offset:0 atIndex:3];
    [e setBytes:&aa length:sizeof aa atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(ihd, 1, 1)];
    }
    if (sparse) {
    e = rc.op("qsa_sc");
    [e setComputePipelineState:k_qsa_scores];
    [e setBuffer:iq offset:0 atIndex:0];
    [e setBuffer:l.bk offset:0 atIndex:1];
    [e setBuffer:scores offset:0 atIndex:2];
    [e setBytes:&aa length:sizeof aa atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(((p0 + M) / ratio + 127) / 128, (M + 16 / inh - 1) / (16 / inh), 1)
         threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    e = rc.op("qsa");
    [e setComputePipelineState:k_qsa];
    [e setBuffer:iq offset:0 atIndex:0];
    [e setBuffer:l.bk offset:0 atIndex:1];
    [e setBuffer:scores offset:0 atIndex:2];
    [e setBuffer:sel offset:0 atIndex:3];
    [e setBuffer:nsel offset:0 atIndex:4];
    [e setBytes:&aa length:sizeof aa atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    }
    e = rc.op("attn_rows");
    [e setComputePipelineState:k_attn_rows];
    [e setBuffer:q offset:0 atIndex:0];
    [e setBuffer:gate offset:0 atIndex:1];
    [e setBuffer:l.Kc offset:0 atIndex:2];
    [e setBuffer:l.Vc offset:0 atIndex:3];
    [e setBuffer:sel offset:0 atIndex:4];
    [e setBuffer:nsel offset:0 atIndex:5];
    [e setBuffer:attn_part offset:0 atIndex:6];
    [e setBytes:&aa length:sizeof aa atIndex:7];
    [e dispatchThreadgroups:MTLSizeMake(M, nkv, ASPLIT) threadsPerThreadgroup:MTLSizeMake(attn_mma ? 128 : 32 * (nh / nkv), 1, 1)];
    e = rc.op("attn_merge");
    [e setComputePipelineState:k_attn_merge];
    [e setBuffer:attn_part offset:0 atIndex:0];
    [e setBuffer:gate offset:0 atIndex:1];
    [e setBuffer:nsel offset:0 atIndex:2];
    [e setBuffer:of offset:0 atIndex:3];
    [e setBytes:&aa length:sizeof aa atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake(M, nh, 1) threadsPerThreadgroup:MTLSizeMake(hd, 1, 1)];
    }
  }
  mm(rc, "mm.out", l.out, of, yo, M);
  hc_block(rc, l.mhc_dn, l.mhc_up, l.mhc_norm, M, 1, yo);
  mm(rc, "mm.moe.in", l.moe_in, xf, yin, M);
  RouteArgs ra{M, 1, l.moe_in.N, E, I, topk};
  if (M <= MAXM) {
    e = rc.op("route");
    [e setComputePipelineState:k_route];
    [e setBuffer:yin offset:0 atIndex:0];
    [e setBuffer:topi offset:0 atIndex:1];
    [e setBuffer:topw offset:0 atIndex:2];
    [e setBuffer:shf offset:0 atIndex:3];
    [e setBuffer:sgate offset:0 atIndex:4];
    [e setBytes:&ra length:sizeof ra atIndex:5];
    [e setBuffer:slots offset:0 atIndex:6];
    [e setBuffer:nslots offset:0 atIndex:7];
    [e setBuffer:slot_of offset:0 atIndex:8];
    [e setBuffer:rmask offset:0 atIndex:9];
    Layer* nx = &l + 1 < L.data() + L.size() ? &l + 1 : nullptr;  // next layer's mm.in weights
    if (nx) pf_set(nx->in);
    const uint32_t rt = 32 * std::min<uint32_t>(std::max<uint32_t>(M, 8), 32);  // simdgroups loop over rows past 32
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1 + pf_ride(e, 1, 1, rt, pf_route)) threadsPerThreadgroup:MTLSizeMake(rt, 1, 1)];
    const uint32_t SLm = M * topk;
    // shared expert down and routed gate/up are independent: no barrier between them
    if (dump_nslots) {  // btest: this layer's distinct experts
      const uint32_t li = uint32_t(&l - L.data());
      e = rc.op("copy_u32");
      [e setComputePipelineState:k_copy_u32];
      [e setBuffer:nslots offset:0 atIndex:0];
      [e setBuffer:dump_nslots offset:0 atIndex:1];
      [e setBytes:&li length:4 atIndex:2];
      [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    }
    mm(rc, "mm.moe.sdown", l.moe_sdown, shf, ysh, M);
    rmask_on = rmask_ok;  // this route's masks cover the gate|up and down that follow
    if (fused_layout) {
      mm(rc, "mm.moe.gu", l.moe_gu, xf, ygu, M, SLm, 0, M * l.moe_gu.N, nil, 0, 1.0f, false, 3, af, nil, nil, I);
    } else {
    mm(rc, "mm.moe.gu", l.moe_gu, xf, ygu, M, SLm, 0, M * l.moe_gu.N, nil, 0, 1.0f, false);
    ActArgs aa2{M, 1, l.moe_gu.N, I};
    e = rc.op("act");
    [e setComputePipelineState:k_act];
    [e setBuffer:ygu offset:0 atIndex:0];
    [e setBuffer:af offset:0 atIndex:1];
    [e setBuffer:nslots offset:0 atIndex:2];
    [e setBytes:&aa2 length:sizeof aa2 atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(M, SLm, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    mm(rc, "mm.moe.dn", l.moe_dn, af, yd, M, SLm, 16 * tiles_for(M) * I / 8, M * l.moe_dn.N);  // af: 16 * TILES rows per slot
    rmask_on = false;
    CombArgs ca{M, 1, H, topk, 1};
    e = rc.op("comb");
    [e setComputePipelineState:k_comb];
    [e setBuffer:yd offset:0 atIndex:0];
    [e setBuffer:ysh offset:0 atIndex:1];
    [e setBuffer:topi offset:0 atIndex:2];
    [e setBuffer:topw offset:0 atIndex:3];
    [e setBuffer:slot_of offset:0 atIndex:4];
    [e setBuffer:sgate offset:0 atIndex:5];
    [e setBuffer:moe_o offset:0 atIndex:6];
    [e setBytes:&ca length:sizeof ca atIndex:7];
    [e dispatchThreadgroups:MTLSizeMake((H + 255) / 256, M, 1 + pf_ride(e, (H + 255) / 256, M, 256, pf_comb)) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    pf = {};
  } else {
    // prefill: each expert's rows gathered into 16-row tiles; same per-row arithmetic
    e = rc.op("route");
    [e setComputePipelineState:k_route_rows];
    [e setBuffer:yin offset:0 atIndex:0];
    [e setBuffer:topi offset:0 atIndex:1];
    [e setBuffer:topw offset:0 atIndex:2];
    [e setBuffer:shf offset:0 atIndex:3];
    [e setBuffer:sgate offset:0 atIndex:4];
    [e setBytes:&ra length:sizeof ra atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake((M + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    struct { uint32_t M, topk, E, maxtiles; } ta{M, topk, E, maxtiles};
    e = rc.op("plan");
    [e setComputePipelineState:k_plan_count];
    [e setBuffer:topi offset:0 atIndex:0];
    [e setBuffer:plan_cnt offset:0 atIndex:1];
    [e setBytes:&ta length:sizeof ta atIndex:2];
    [e dispatchThreads:MTLSizeMake(M * topk, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    e = rc.op("plan");
    [e setComputePipelineState:k_plan_prefix];
    [e setBuffer:plan_cnt offset:0 atIndex:0];
    [e setBuffer:plan_first offset:0 atIndex:1];
    [e setBuffer:tile_exp offset:0 atIndex:2];
    [e setBuffer:ntiles offset:0 atIndex:3];
    [e setBuffer:tile_rows offset:0 atIndex:4];
    [e setBytes:&ta length:sizeof ta atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
    e = rc.op("plan");
    [e setComputePipelineState:k_plan_scatter];
    [e setBuffer:topi offset:0 atIndex:0];
    [e setBuffer:plan_cnt offset:0 atIndex:1];
    [e setBuffer:plan_first offset:0 atIndex:2];
    [e setBuffer:tile_rows offset:0 atIndex:3];
    [e setBuffer:assign_pos offset:0 atIndex:4];
    [e setBytes:&ta length:sizeof ta atIndex:5];
    [e dispatchThreads:MTLSizeMake(M * topk, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    // the fill counters must be zero again for the next layer
    e = rc.op("plan");
    [e setComputePipelineState:k_zero];
    [e setBuffer:plan_cnt offset:0 atIndex:0];
    [e dispatchThreads:MTLSizeMake(E, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    const uint32_t T = M * topk / 16 + E < maxtiles ? M * topk / 16 + E : maxtiles;
    // grouped matmuls over tiles: slot = tile, expert = tile_exp[tile], 16 rows each
    std::swap(slots, tile_exp);
    std::swap(nslots, ntiles);
    if (fused_layout) {
      gather_mode = true;  // the tile rows are read from xf directly
      mm(rc, "mm.moe.gu", l.moe_gu, xf, ygu, 16, T, 16 * H / 8, 16 * l.moe_gu.N, nil, 0, 1.0f, true, 3, af, nil, nil, I);
      gather_mode = false;
    } else {
    gather_mode = true;
    mm(rc, "mm.moe.gu", l.moe_gu, xf, ygu, 16, T, 16 * H / 8, 16 * l.moe_gu.N);
    gather_mode = false;
    ActArgs aa2{16, 1, l.moe_gu.N, I};
    e = rc.op("act");
    [e setComputePipelineState:k_act];
    [e setBuffer:ygu offset:0 atIndex:0];
    [e setBuffer:af offset:0 atIndex:1];
    [e setBuffer:nslots offset:0 atIndex:2];
    [e setBytes:&aa2 length:sizeof aa2 atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(16, T, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    mm(rc, "mm.moe.dn", l.moe_dn, af, yd, 16, T, 16 * I / 8, 16 * l.moe_dn.N);
    std::swap(slots, tile_exp);
    std::swap(nslots, ntiles);
    mm(rc, "mm.moe.sdown", l.moe_sdown, shf, ysh, M);
    CombArgs ca{M, 1, H, topk, 1};
    e = rc.op("comb");
    [e setComputePipelineState:k_comb_tiles];
    [e setBuffer:yd offset:0 atIndex:0];
    [e setBuffer:ysh offset:0 atIndex:1];
    [e setBuffer:topi offset:0 atIndex:2];
    [e setBuffer:topw offset:0 atIndex:3];
    [e setBuffer:assign_pos offset:0 atIndex:4];
    [e setBuffer:sgate offset:0 atIndex:5];
    [e setBuffer:moe_o offset:0 atIndex:6];
    [e setBytes:&ca length:sizeof ca atIndex:7];
    [e dispatchThreadgroups:MTLSizeMake((H + 255) / 256, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }
  pending = true;
}

// GatedDeltaNet in three passes (pre / scan / post); post is skipped for commits.
void Model::encode_gdn(Rec& rc, Layer& l, id<MTLBuffer> gi, const GdnArgs& ga, uint32_t M) {
  id<MTLComputeCommandEncoder> e;
  if (M <= MAXM) {  // few rows: one kernel (same arithmetic as the split passes)
    e = rc.op("gdn");
    [e setComputePipelineState:k_gdn];
    [e setBuffer:gi offset:0 atIndex:0];
    [e setBuffer:l.convh offset:0 atIndex:1];
    [e setBuffer:l.conv.b offset:l.conv.off atIndex:2];
    [e setBuffer:l.S offset:0 atIndex:3];
    [e setBuffer:l.A_log.b offset:l.A_log.off atIndex:4];
    [e setBuffer:l.dt_bias.b offset:l.dt_bias.off atIndex:5];
    [e setBuffer:l.gnorm.b offset:l.gnorm.off atIndex:6];
    [e setBuffer:of offset:0 atIndex:7];
    [e setBytes:&ga length:sizeof ga atIndex:8];
    pf_set(l.out);  // mm.out follows
    [e dispatchThreadgroups:MTLSizeMake(nv, 1, 1 + pf_ride(e, nv, 1, 1024, pf_gdn)) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    pf = {};
    return;
  }
  e = rc.op("gdn.pre");
  [e setComputePipelineState:k_gdn_pre];
  [e setBuffer:gi offset:0 atIndex:0];
  [e setBuffer:l.convh offset:0 atIndex:1];
  [e setBuffer:l.conv.b offset:l.conv.off atIndex:2];
  [e setBuffer:l.A_log.b offset:l.A_log.off atIndex:3];
  [e setBuffer:l.dt_bias.b offset:l.dt_bias.off atIndex:4];
  [e setBuffer:g_qn offset:0 atIndex:5];
  [e setBuffer:g_kn offset:0 atIndex:6];
  [e setBuffer:g_vv offset:0 atIndex:7];
  [e setBuffer:g_bd offset:0 atIndex:8];
  [e setBytes:&ga length:sizeof ga atIndex:9];
  [e dispatchThreadgroups:MTLSizeMake(M, nv, 1) threadsPerThreadgroup:MTLSizeMake(dv, 1, 1)];
  e = rc.op("gdn.scan");
  [e setComputePipelineState:k_gdn_scan];
  [e setBuffer:gi offset:0 atIndex:0];
  [e setBuffer:l.convh offset:0 atIndex:1];
  [e setBuffer:l.S offset:0 atIndex:2];
  [e setBuffer:g_qn offset:0 atIndex:3];
  [e setBuffer:g_kn offset:0 atIndex:4];
  [e setBuffer:g_vv offset:0 atIndex:5];
  [e setBuffer:g_bd offset:0 atIndex:6];
  [e setBuffer:g_o offset:0 atIndex:7];
  [e setBytes:&ga length:sizeof ga atIndex:8];
  [e dispatchThreadgroups:MTLSizeMake(nv, dv / 16, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  if (ga.mode == 2) return;
  e = rc.op("gdn.post");
  [e setComputePipelineState:k_gdn_post];
  [e setBuffer:gi offset:0 atIndex:0];
  [e setBuffer:g_o offset:0 atIndex:1];
  [e setBuffer:l.gnorm.b offset:l.gnorm.off atIndex:2];
  [e setBuffer:of offset:0 atIndex:3];
  [e setBytes:&ga length:sizeof ga atIndex:4];
  [e dispatchThreadgroups:MTLSizeMake(M, nv, 1) threadsPerThreadgroup:MTLSizeMake(dv, 1, 1)];
}

// mixer read-out of h (after injecting the pending MoE output) -> lm_head -> argmax
void Model::encode_head(Rec& rc, const Lin& dn, const Lin& up, const Raw& norm, uint32_t M, bool pending, uint32_t p0,
                        const Lin* head) {
  hc_block(rc, dn, up, norm, M, pending ? 1 : 0, moe_o);
  mm(rc, "mm.lm.head", head ? *head : lm_head, xf, logits, M);
  ArgArgs ga{M, 1, head ? head->N : lm_head.N};
  id<MTLComputeCommandEncoder> e = rc.op("argmax");
  [e setComputePipelineState:k_argmax];
  [e setBuffer:logits offset:0 atIndex:0];
  [e setBuffer:amax_i offset:0 atIndex:1];
  [e setBuffer:amax_v offset:0 atIndex:2];
  [e setBuffer:amax_m offset:0 atIndex:4];
  [e setBytes:&ga length:sizeof ga atIndex:3];
  [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
  const bool mapped = head && head == &mtp_head && has_sub_head;
  const bool per_row = !row_samp.empty();
  if (per_row || (!lane_rows && temp > 0.0f)) {  // sampled tokens overwrite the argmax (row r predicts position p0 + r + 1)
    struct { uint32_t M, N; float temp, top_p; uint32_t top_k, seed_lo, seed_hi, pos0, mapped, per_row; } sa{
        M, head ? head->N : lm_head.N, temp, top_p, top_k, uint32_t(sample_seed), uint32_t(sample_seed >> 32), p0 + 1,
        mapped ? 1u : 0u, per_row ? 1u : 0u};
    if (per_row && row_samp.size() < M) mtl::die("encode_head: row_samp has fewer rows than the step");
    e = rc.op("sample");
    [e setComputePipelineState:k_sample];
    [e setBuffer:logits offset:0 atIndex:0];
    [e setBuffer:amax_i offset:0 atIndex:1];
    [e setBytes:&sa length:sizeof sa atIndex:2];
    [e setBuffer:(mapped ? sub_ids.b : logits) offset:(mapped ? sub_ids.off : 0) atIndex:3];
    if (per_row) [e setBytes:row_samp.data() length:row_samp.size() * sizeof(RowSamp) atIndex:4];
    else [e setBytes:&sa length:sizeof sa atIndex:4];  // unread placeholder
    [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
  }
  if (mapped) {
    e = rc.op("map");
    [e setComputePipelineState:k_map_ids];
    [e setBuffer:amax_i offset:0 atIndex:0];
    [e setBuffer:sub_ids.b offset:sub_ids.off atIndex:1];
    [e setBytes:&M length:4 atIndex:2];
    [e dispatchThreads:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  }
}

void Model::begin(Rec& rc, std::map<std::string, double>* profile) {
  enc_t0 = std::chrono::steady_clock::now();
  rc.cb = [C.q commandBuffer];
  rc.prof = profile != nullptr;
  rc.conc = getenv("OPUS_CONC") != nullptr;
  if (rc.prof) rc.ext_pool = &prof_pool;
  static const uint32_t chunk0 = getenv("OPUS_CHUNK0") ? atoi(getenv("OPUS_CHUNK0")) : 4;
  static const uint32_t chunk = getenv("OPUS_CHUNK") ? atoi(getenv("OPUS_CHUNK")) : 256;
  if (!rc.prof) { rc.chunk0 = chunk0; rc.chunk = chunk; }
}

double Model::finish(Rec& rc, std::map<std::string, double>* profile) {
  if (rc.e) [rc.e endEncoding];
  t_encode += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - enc_t0).count();
  MTLTimestamp c0 = 0, g0 = 0, c1 = 0, g1 = 0;
  if (rc.prof) [C.dev sampleTimestamps:&c0 gpuTimestamp:&g0];
  mtl::last_use().store(mtl::now_s());
  [rc.cb commit];
  static const bool spin = getenv("OPUS_SPIN") != nullptr;
  if (spin) {  // poll instead of sleeping on the completion: avoids the scheduler wake-up latency
    while (rc.cb.status < MTLCommandBufferStatusCompleted) {
#if defined(__arm64__)
      __asm__ __volatile__("yield");
#endif
    }
  }
  [rc.cb waitUntilCompleted];
  if (rc.cb.error) mtl::die("command buffer", rc.cb.error);
  for (id<MTLCommandBuffer> c : rc.sent) {  // committed before rc.cb on the same queue: already done
    [c waitUntilCompleted];
    if (c.error) mtl::die("command buffer", c.error);
  }
  if (rc.prof) {
    [C.dev sampleTimestamps:&c1 gpuTimestamp:&g1];
    const double ns_per_tick = g1 > g0 ? double(c1 - c0) / double(g1 - g0) : 1.0;
    for (size_t b = 0; b * PROF_OPS < rc.names.size(); b++) {
      const size_t n = std::min(PROF_OPS, rc.names.size() - b * PROF_OPS);
      NSData* d = [prof_pool[b] resolveCounterRange:NSMakeRange(0, 2 * n)];
      const MTLCounterResultTimestamp* ts = (const MTLCounterResultTimestamp*)d.bytes;
      for (size_t i = 0; i < n; i++)
        (*profile)[rc.names[b * PROF_OPS + i]] += double(ts[2 * i + 1].timestamp - ts[2 * i].timestamp) * ns_per_tick * 1e-6;
    }
  }
  if (!rc.sent.empty()) return (rc.cb.GPUEndTime - rc.sent[0].GPUStartTime) * 1e3;  // first start to last end
  return mtl::gpu_seconds(rc.cb) * 1e3;
}

// Trunk forward over M tokens at positions pos.. . mode 0 commits every row; mode 1
// (verify) leaves recurrent state, conv/PLE history and pos untouched until commit(j).
std::vector<uint32_t> Model::forward(const std::vector<uint32_t>& toks, int mode, double* gpu_ms,
                                     std::vector<float>* logits_out, std::map<std::string, double>* profile) {
  const uint32_t M = uint32_t(toks.size());
  if (M == 0 || M > MP || (mode != 0 && M > MAXM)) mtl::die("forward: 1..512 rows (verify: 16)");
  if (want_all && M > SROWS) mtl::die("forward: per-row logits for at most SROWS rows");
  if (pos + M > cap) mtl::die("forward: KV capacity exceeded");
  memcpy(ids.contents, toks.data(), M * 4);
  const uint32_t p0 = pos;
  nsel = nsel_trunk;
  auto th0 = std::chrono::steady_clock::now();
  set_rope(p0, M);
  pending_toks = toks;
  if (pre_frag_M == M && !pre_frag.empty()) {
    // rows fetched in the background while the previous chunk was on the GPU (see feed)
    memcpy(ple_e.contents, pre_frag.data(), pre_frag.size() * 2);
  } else {
    std::vector<uint32_t> hist = history;
    hist.insert(hist.end(), toks.begin(), toks.end());
    ngram->rows_frag(hist, M, (uint16_t*)ple_e.contents);
  }
  pre_frag_M = 0;
  for (uint32_t r = 0; r < M; r++) ((uint32_t*)nsel.contents)[r] = 0xFFFFFFFFu;
  t_host += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - th0).count();

  Rec rc;
  begin(rc, profile);
  const uint32_t MH = encode_trunk(rc, M, mode, p0);
  const double ms = finish(rc, profile);
  if (gpu_ms) *gpu_ms = ms;
  if (mode == 0) {
    pos += M;
    conv_par ^= 1;
    ple_par ^= 1;
    history.insert(history.end(), toks.begin(), toks.end());
  }
  if (logits_out) {
    logits_out->resize(size_t(MH) * vocab);
    memcpy(logits_out->data(), logits.contents, size_t(MH) * vocab * 4);
  }
  std::vector<uint32_t> out(MH);
  memcpy(out.data(), amax_i.contents, MH * 4);
  return out;
}

// Embedding -> layers -> head for M rows at p0 (ids, rope, n-gram rows already in place).
// Returns the number of logit rows (1 for prefill chunks).
uint32_t Model::encode_trunk(Rec& rc, uint32_t M, int mode, uint32_t p0) {
  EmbArgs ea{M, H, HC};
  id<MTLComputeCommandEncoder> e = rc.op("embed");
  [e setComputePipelineState:k_embed];
  [e setBuffer:ids offset:0 atIndex:0];
  [e setBuffer:embed.b offset:embed.off atIndex:1];
  [e setBuffer:h offset:0 atIndex:2];
  [e setBytes:&ea length:sizeof ea atIndex:3];
  [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
  bool pending = false;
  for (uint32_t li = 0; li < nL; li++) {
    encode_layer(rc, L[li], M, p0, mode, pending);
    if (dump_h) {  // dtest: layer li's rows, before the next layer injects the pending MoE output
      copy(rc, h, 0, dump_h, li * M, M, H * HC);
      copy(rc, moe_o, 0, dump_moe, li * M, M, H);
    }
  }
  uint32_t MH = M;
  if (M > MAXM) {
    // prefill: only the last row needs logits. Land the MoE output in h, keep every row's
    // final streams for the MTP head, then run the mixer and lm_head on the last row.
    HcArgs ha{M, H, HC, 1, prev_dn_N, R, topk};
    e = rc.op("hc_pre");
    [e setComputePipelineState:k_hc_pre];
    [e setBuffer:h offset:0 atIndex:0];
    [e setBuffer:moe_o offset:0 atIndex:1];
    [e setBuffer:ydn offset:0 atIndex:2];
    [e setBuffer:mix_norm.b offset:mix_norm.off atIndex:3];
    [e setBuffer:hn offset:0 atIndex:4];
    [e setBuffer:hnf offset:0 atIndex:5];
    [e setBytes:&ha length:sizeof ha atIndex:6];
    [e dispatchThreadgroups:MTLSizeMake(M, HC, 1 + pf_ride(e, M, HC, hct, 0)) threadsPerThreadgroup:MTLSizeMake(hct, 1, 1)];
    copy(rc, h, 0, h_tgt, 0, M, H * HC);
    pending = false;
    if (!want_all) {  // streak chunks keep every row: the head then runs exactly as it does per row
      copy(rc, h_tgt, M - 1, h, 0, 1, H * HC);
      MH = 1;
    }
  }
  encode_head(rc, mix_dn, mix_up, mix_norm, MH, pending, (M > MAXM && MH == 1) ? p0 + M - 1 : p0);
  if (M <= MAXM) copy(rc, h, 0, h_tgt, 0, M, H * HC);  // final hidden streams, for the MTP head
  return MH;
}

// After a verify forward: make the first j rows permanent (bit-identical to having fed them
// with mode 0), by replaying their recurrent updates from the stored layer inputs.
void Model::encode_commit(Rec& rc, uint32_t j) {
  id<MTLComputeCommandEncoder> e;
  for (auto& l : L) {
    if (l.ple) {
      PleArgs pa{j, 1, l.ple_kv.N, H, HC, ple_par, 9, 3, 4, 2};
      e = rc.op("ple_conv");
      [e setComputePipelineState:k_ple_conv];
      [e setBuffer:h offset:0 atIndex:0];
      [e setBuffer:ple_gv offset:0 atIndex:1];
      [e setBuffer:ple_gvn offset:0 atIndex:2];
      [e setBuffer:l.plehist offset:0 atIndex:3];
      [e setBuffer:l.ple_conv.b offset:l.ple_conv.off atIndex:4];
      [e setBytes:&pa length:sizeof pa atIndex:5];
      [e dispatchThreads:MTLSizeMake(H * HC, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    if (l.attn) continue;
    GdnArgs ga{j, 1, l.in.N, nk, nv, dk, dv, conv_par, 2};
    encode_gdn(rc, l, l.gin, ga, j);
  }
  conv_par ^= 1;
  ple_par ^= 1;
  pos += j;
  history.insert(history.end(), pending_toks.begin(), pending_toks.begin() + j);
}

// ---------------------------------------------------------------- lanes
void Model::encode_ple_lanes(Rec& rc, Layer& l, size_t li, int mode) {
  const size_t W = size_t(H) * HC;
  for (auto& ln : jl) {
    Layer& ll = &l == &mtp ? ln.m->mtp : ln.m->L[li];  // the MTP layer is not in L
    PleArgs pa{ln.M, 1, l.ple_kv.N, H, HC, ln.m->ple_par, 9, 3, 4, uint32_t(mode)};
    id<MTLComputeCommandEncoder> e = rc.op("ple_conv");
    [e setComputePipelineState:k_ple_conv];
    [e setBuffer:h offset:ln.row0 * W * 4 atIndex:0];
    [e setBuffer:ple_gv offset:ln.row0 * W * 4 atIndex:1];
    [e setBuffer:ple_gvn offset:ln.row0 * W * 4 atIndex:2];
    [e setBuffer:ll.plehist offset:0 atIndex:3];
    [e setBuffer:l.ple_conv.b offset:l.ple_conv.off atIndex:4];
    [e setBytes:&pa length:sizeof pa atIndex:5];
    [e dispatchThreads:MTLSizeMake(H * HC, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }
}

void Model::encode_gdn_lanes(Rec& rc, Layer& l, size_t li, int mode) {
  for (auto& ln : jl) {
    Layer& ll = &l == &mtp ? ln.m->mtp : ln.m->L[li];  // the MTP layer is not in L
    GdnArgs ga{ln.M, 1, l.in.N, nk, nv, dk, dv, ln.m->conv_par, uint32_t(mode), ln.row0};
    id<MTLComputeCommandEncoder> e = rc.op("gdn");
    [e setComputePipelineState:k_gdn];
    [e setBuffer:l.gin offset:size_t(ln.row0) * l.in.N * 4 atIndex:0];
    [e setBuffer:ll.convh offset:0 atIndex:1];
    [e setBuffer:l.conv.b offset:l.conv.off atIndex:2];
    [e setBuffer:ll.S offset:0 atIndex:3];
    [e setBuffer:l.A_log.b offset:l.A_log.off atIndex:4];
    [e setBuffer:l.dt_bias.b offset:l.dt_bias.off atIndex:5];
    [e setBuffer:l.gnorm.b offset:l.gnorm.off atIndex:6];
    [e setBuffer:of offset:0 atIndex:7];
    [e setBytes:&ga length:sizeof ga atIndex:8];
    pf = {};
    [e dispatchThreadgroups:MTLSizeMake(nv, 1, 1 + pf_ride(e, nv, 1, 1024, 0)) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
  }
}

void Model::encode_attn_lanes(Rec& rc, Layer& l, size_t li) {
  for (auto& ln : jl) {
    Layer& ll = &l == &mtp ? ln.m->mtp : ln.m->L[li];  // the MTP layer is not in L
    const uint32_t M = ln.M, p0 = ln.p0;
    const size_t r0 = ln.row0;
    const size_t o_y = r0 * l.in.N * 4, o_q = r0 * nh * hd * 4, o_iq = r0 * inh * ihd * 4,
                 o_sc = r0 * (cap / ratio) * 4, o_sel = r0 * (budget / ratio) * 4, o_n = r0 * 4,
                 o_part = r0 * nh * ASPLIT * (hd + 2) * 4;
    id<MTLBuffer> rp = rope_slot[ln.slot], rb = rope_blk_slot[ln.slot];
    AttnArgs aa{M, 1, l.in.N, nh, nkv, hd, inh, ihd, rot, p0, ratio, cap, budget, ln.row0, kv8};
    id<MTLComputeCommandEncoder> e = rc.op("attn_prep");
    [e setComputePipelineState:k_attn_prep];
    [e setBuffer:yin offset:o_y atIndex:0];
    [e setBuffer:rp offset:0 atIndex:1];
    [e setBuffer:l.q_norm.b offset:l.q_norm.off atIndex:2];
    [e setBuffer:l.k_norm.b offset:l.k_norm.off atIndex:3];
    [e setBuffer:l.iq_norm.b offset:l.iq_norm.off atIndex:4];
    [e setBuffer:l.ik_norm.b offset:l.ik_norm.off atIndex:5];
    [e setBuffer:q offset:o_q atIndex:6];
    [e setBuffer:gate offset:o_q atIndex:7];
    [e setBuffer:ll.Kc offset:0 atIndex:8];
    [e setBuffer:ll.Vc offset:0 atIndex:9];
    [e setBuffer:iq offset:o_iq atIndex:10];
    [e setBuffer:ll.rawk offset:0 atIndex:11];
    [e setBuffer:ll.bk offset:0 atIndex:12];
    [e setBuffer:rb offset:0 atIndex:13];
    [e setBytes:&aa length:sizeof aa atIndex:14];
    [e dispatchThreadgroups:MTLSizeMake(M, nh + nkv + inh, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    bool completes = false, sparse = false;
    for (uint32_t r = 0; r < M; r++) {
      completes |= (p0 + r + 1) % ratio == 0;
      sparse |= (p0 + r + 1) / ratio > budget / ratio;
    }
    if (completes) {
      e = rc.op("attn_blocks");
      [e setComputePipelineState:k_attn_blocks];
      [e setBuffer:ll.rawk offset:0 atIndex:0];
      [e setBuffer:l.ik_norm.b offset:l.ik_norm.off atIndex:1];
      [e setBuffer:ll.bk offset:0 atIndex:2];
      [e setBuffer:rb offset:0 atIndex:3];
      [e setBytes:&aa length:sizeof aa atIndex:4];
      [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(ihd, 1, 1)];
    }
    if (sparse) {
      e = rc.op("qsa_sc");
      [e setComputePipelineState:k_qsa_scores];
      [e setBuffer:iq offset:o_iq atIndex:0];
      [e setBuffer:ll.bk offset:0 atIndex:1];
      [e setBuffer:scores offset:o_sc atIndex:2];
      [e setBytes:&aa length:sizeof aa atIndex:3];
      [e dispatchThreadgroups:MTLSizeMake(((p0 + M) / ratio + 127) / 128, (M + 16 / inh - 1) / (16 / inh), 1)
           threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      e = rc.op("qsa");
      [e setComputePipelineState:k_qsa];
      [e setBuffer:iq offset:o_iq atIndex:0];
      [e setBuffer:ll.bk offset:0 atIndex:1];
      [e setBuffer:scores offset:o_sc atIndex:2];
      [e setBuffer:sel offset:o_sel atIndex:3];
      [e setBuffer:nsel offset:o_n atIndex:4];
      [e setBytes:&aa length:sizeof aa atIndex:5];
      [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    }
    e = rc.op("attn_rows");
    [e setComputePipelineState:k_attn_rows];
    [e setBuffer:q offset:o_q atIndex:0];
    [e setBuffer:gate offset:o_q atIndex:1];
    [e setBuffer:ll.Kc offset:0 atIndex:2];
    [e setBuffer:ll.Vc offset:0 atIndex:3];
    [e setBuffer:sel offset:o_sel atIndex:4];
    [e setBuffer:nsel offset:o_n atIndex:5];
    [e setBuffer:attn_part offset:o_part atIndex:6];
    [e setBytes:&aa length:sizeof aa atIndex:7];
    [e dispatchThreadgroups:MTLSizeMake(M, nkv, ASPLIT) threadsPerThreadgroup:MTLSizeMake(attn_mma ? 128 : 32 * (nh / nkv), 1, 1)];
    e = rc.op("attn_merge");
    [e setComputePipelineState:k_attn_merge];
    [e setBuffer:attn_part offset:o_part atIndex:0];
    [e setBuffer:gate offset:o_q atIndex:1];
    [e setBuffer:nsel offset:o_n atIndex:2];
    [e setBuffer:of offset:0 atIndex:3];
    [e setBytes:&aa length:sizeof aa atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake(M, nh, 1) threadsPerThreadgroup:MTLSizeMake(hd, 1, 1)];
  }
}

// ---- lane-batched stateful ops (one dispatch per layer for every lane; see model.h lane_batch)
struct LaneAttnH { uint64_t Kc, Vc, rawk, bk; uint32_t p0, row0, pad0, pad1; };
struct LaneGdnH { uint64_t S, convh; uint32_t M, row0, conv_par, pad; };

void Model::lanes_tables() {
  std::vector<uint32_t> ps, pb;
  uint8_t* rl = (uint8_t*)rowlane.contents;
  for (size_t k = 0; k < jl.size(); k++)
    for (uint32_t r = 0; r < jl[k].M; r++) {
      ps.push_back(jl[k].p0 + r);
      pb.push_back(((jl[k].p0 + r) / ratio) * ratio);
      rl[jl[k].row0 + r] = uint8_t(k);
    }
  const uint32_t M = uint32_t(ps.size());
  rope_rows(rope_theta, rot, ps, (float*)rope_lanes.contents);
  std::vector<float> tmp(2 * M * rot);
  rope_rows(rope_theta, rot, pb, tmp.data());
  float* rb = (float*)rope_blk_lanes.contents;
  for (uint32_t r = 0; r < M; r++) {
    memcpy(rb + r * 2 * rot, tmp.data() + r * rot, rot * 4);
    memcpy(rb + r * 2 * rot + rot, tmp.data() + (M + r) * rot, rot * 4);
  }
}

void Model::encode_gdn_lanes_b(Rec& rc, Layer& l, size_t li, int mode) {
  LaneGdnH* tab = (LaneGdnH*)lane_gdn_tab.contents + li * NSLOT;
  std::vector<id<MTLResource>> res;
  for (size_t k = 0; k < jl.size(); k++) {
    Layer& ll = jl[k].m->L[li];
    tab[k] = {ll.S.gpuAddress, ll.convh.gpuAddress, jl[k].M, jl[k].row0, jl[k].m->conv_par, 0};
    res.push_back(ll.S);
    res.push_back(ll.convh);
  }
  GdnArgs ga{0, 1, l.in.N, nk, nv, dk, dv, 0, uint32_t(mode), 0};
  id<MTLComputeCommandEncoder> e = rc.op("gdn");
  [e setComputePipelineState:k_gdn_lanes];
  [e setBuffer:l.gin offset:0 atIndex:0];
  [e setBuffer:lane_gdn_tab offset:li * NSLOT * sizeof(LaneGdnH) atIndex:1];
  [e setBuffer:l.conv.b offset:l.conv.off atIndex:2];
  [e setBuffer:l.A_log.b offset:l.A_log.off atIndex:4];
  [e setBuffer:l.dt_bias.b offset:l.dt_bias.off atIndex:5];
  [e setBuffer:l.gnorm.b offset:l.gnorm.off atIndex:6];
  [e setBuffer:of offset:0 atIndex:7];
  [e setBytes:&ga length:sizeof ga atIndex:8];
  [e useResources:res.data() count:res.size() usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  [e dispatchThreadgroups:MTLSizeMake(nv, 1, jl.size()) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
}

void Model::encode_attn_lanes_b(Rec& rc, Layer& l, size_t li) {
  const bool is_mtp = &l == &mtp;
  const size_t slot = is_mtp ? nL : li;
  LaneAttnH* tab = (LaneAttnH*)lane_attn_tab.contents + slot * NSLOT;
  const size_t toff = slot * NSLOT * sizeof(LaneAttnH);
  std::vector<id<MTLResource>> res;
  uint32_t M = 0;
  bool completes = false;
  for (size_t k = 0; k < jl.size(); k++) {
    Layer& ll = is_mtp ? jl[k].m->mtp : jl[k].m->L[li];
    tab[k] = {ll.Kc.gpuAddress, ll.Vc.gpuAddress, ll.rawk.gpuAddress, ll.bk.gpuAddress, jl[k].p0, jl[k].row0, 0, 0};
    for (id<MTLBuffer> b : {ll.Kc, ll.Vc, ll.rawk, ll.bk}) res.push_back(b);
    for (uint32_t r = 0; r < jl[k].M; r++) completes |= (jl[k].p0 + r + 1) % ratio == 0;
    M += jl[k].M;
  }
  Layer& l0 = is_mtp ? jl[0].m->mtp : jl[0].m->L[li];  // any lane's buffers fill the unused bindings
  AttnArgs aa{M, 1, l.in.N, nh, nkv, hd, inh, ihd, rot, 0, ratio, cap, budget, 0, kv8};
  auto lanes_bind = [&](id<MTLComputeCommandEncoder> e) {
    [e setBuffer:lane_attn_tab offset:toff atIndex:20];
    [e setBuffer:rowlane offset:0 atIndex:21];
    [e useResources:res.data() count:res.size() usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  };
  id<MTLComputeCommandEncoder> e = rc.op("attn_prep");
  [e setComputePipelineState:k_attn_prep_l];
  [e setBuffer:yin offset:0 atIndex:0];
  [e setBuffer:rope_lanes offset:0 atIndex:1];
  [e setBuffer:l.q_norm.b offset:l.q_norm.off atIndex:2];
  [e setBuffer:l.k_norm.b offset:l.k_norm.off atIndex:3];
  [e setBuffer:l.iq_norm.b offset:l.iq_norm.off atIndex:4];
  [e setBuffer:l.ik_norm.b offset:l.ik_norm.off atIndex:5];
  [e setBuffer:q offset:0 atIndex:6];
  [e setBuffer:gate offset:0 atIndex:7];
  [e setBuffer:l0.Kc offset:0 atIndex:8];
  [e setBuffer:l0.Vc offset:0 atIndex:9];
  [e setBuffer:iq offset:0 atIndex:10];
  [e setBuffer:l0.rawk offset:0 atIndex:11];
  [e setBuffer:l0.bk offset:0 atIndex:12];
  [e setBuffer:rope_blk_lanes offset:0 atIndex:13];
  [e setBytes:&aa length:sizeof aa atIndex:14];
  lanes_bind(e);
  [e dispatchThreadgroups:MTLSizeMake(M, nh + nkv + inh, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  if (completes) {
    e = rc.op("attn_blocks");
    [e setComputePipelineState:k_attn_blocks_l];
    [e setBuffer:l0.rawk offset:0 atIndex:0];
    [e setBuffer:l.ik_norm.b offset:l.ik_norm.off atIndex:1];
    [e setBuffer:l0.bk offset:0 atIndex:2];
    [e setBuffer:rope_blk_lanes offset:0 atIndex:3];
    [e setBytes:&aa length:sizeof aa atIndex:4];
    lanes_bind(e);
    [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(ihd, 1, 1)];
  }
  // sparse block selection stays per lane: a threadgroup's rows share block keys
  for (auto& ln : jl) {
    Layer& ll = is_mtp ? ln.m->mtp : ln.m->L[li];
    const uint32_t Ml = ln.M, p0 = ln.p0;
    bool sparse = false;
    for (uint32_t r = 0; r < Ml; r++) sparse |= (p0 + r + 1) / ratio > budget / ratio;
    if (!sparse) continue;
    const size_t r0 = ln.row0, o_iq = r0 * inh * ihd * 4, o_sc = r0 * (cap / ratio) * 4, o_sel = r0 * (budget / ratio) * 4,
                 o_n = r0 * 4;
    AttnArgs al{Ml, 1, l.in.N, nh, nkv, hd, inh, ihd, rot, p0, ratio, cap, budget, ln.row0, kv8};
    e = rc.op("qsa_sc");
    [e setComputePipelineState:k_qsa_scores];
    [e setBuffer:iq offset:o_iq atIndex:0];
    [e setBuffer:ll.bk offset:0 atIndex:1];
    [e setBuffer:scores offset:o_sc atIndex:2];
    [e setBytes:&al length:sizeof al atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(((p0 + Ml) / ratio + 127) / 128, (Ml + 16 / inh - 1) / (16 / inh), 1)
         threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    e = rc.op("qsa");
    [e setComputePipelineState:k_qsa];
    [e setBuffer:iq offset:o_iq atIndex:0];
    [e setBuffer:ll.bk offset:0 atIndex:1];
    [e setBuffer:scores offset:o_sc atIndex:2];
    [e setBuffer:sel offset:o_sel atIndex:3];
    [e setBuffer:nsel offset:o_n atIndex:4];
    [e setBytes:&al length:sizeof al atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(Ml, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
  }
  e = rc.op("attn_rows");
  [e setComputePipelineState:k_attn_rows_l];
  [e setBuffer:q offset:0 atIndex:0];
  [e setBuffer:gate offset:0 atIndex:1];
  [e setBuffer:l0.Kc offset:0 atIndex:2];
  [e setBuffer:l0.Vc offset:0 atIndex:3];
  [e setBuffer:sel offset:0 atIndex:4];
  [e setBuffer:nsel offset:0 atIndex:5];
  [e setBuffer:attn_part offset:0 atIndex:6];
  [e setBytes:&aa length:sizeof aa atIndex:7];
  lanes_bind(e);
  [e dispatchThreadgroups:MTLSizeMake(M, nkv, ASPLIT) threadsPerThreadgroup:MTLSizeMake(attn_mma ? 128 : 32 * (nh / nkv), 1, 1)];
  e = rc.op("attn_merge");
  [e setComputePipelineState:k_attn_merge_l];
  [e setBuffer:attn_part offset:0 atIndex:0];
  [e setBuffer:gate offset:0 atIndex:1];
  [e setBuffer:nsel offset:0 atIndex:2];
  [e setBuffer:of offset:0 atIndex:3];
  [e setBytes:&aa length:sizeof aa atIndex:4];
  lanes_bind(e);
  [e dispatchThreadgroups:MTLSizeMake(M, nh, 1) threadsPerThreadgroup:MTLSizeMake(hd, 1, 1)];
}

std::vector<std::vector<uint32_t>> Model::forward_lanes(const std::vector<Model*>& lanes,
                                                        const std::vector<std::vector<uint32_t>>& toks, int mode,
                                                        std::vector<std::vector<float>>* logits_out, double* gpu_ms,
                                                        std::map<std::string, double>* profile) {
  if (lanes.empty() || lanes.size() != toks.size() || lanes.size() > size_t(NSLOT)) mtl::die("forward_lanes: 1..NSLOT lanes");
  if (!fused_layout) mtl::die("forward_lanes: needs the fused-layout pack");
  uint32_t M = 0;
  jl.clear();
  for (size_t k = 0; k < lanes.size(); k++) {
    Model* m = lanes[k];
    const uint32_t Ml = uint32_t(toks[k].size());
    if (Ml == 0 || m->cap != cap || m->pos + Ml > m->cap) mtl::die("forward_lanes: empty lane, capacity mismatch or full");
    jl.push_back({m, M, Ml, m->pos, int(k)});
    M += Ml;
  }
  if (M > MAXM) mtl::die("forward_lanes: more than MAXM rows");
  nsel = nsel_trunk;
  for (uint32_t r = 0; r < M; r++) ((uint32_t*)nsel.contents)[r] = 0xFFFFFFFFu;
  const uint32_t Kn = (ngram->ngram_size - 1) * ngram->heads_per * ngram->dim;
  memset(ple_e.contents, 0, size_t((M + 15) / 16 * 16) * Kn * 2);
  for (auto& ln : jl) {
    const auto& t = toks[size_t(ln.slot)];
    memcpy((uint32_t*)ids.contents + ln.row0, t.data(), ln.M * 4);
    set_rope(ln.p0, ln.M, ln.slot);
    ln.m->pending_toks = t;
    std::vector<uint32_t> hist = ln.m->history;
    hist.insert(hist.end(), t.begin(), t.end());
    ngram->rows_frag_at(hist, ln.M, ln.row0, (uint16_t*)ple_e.contents);
  }
  rope = rope_slot[0];
  rope_blk = rope_blk_slot[0];
  if (lane_batch) lanes_tables();
  // each row samples with its own lane's settings, keyed by its own position (as when the lane runs alone)
  row_samp.clear();
  bool any_sampled = false;
  for (auto& ln : jl) any_sampled |= ln.m->temp > 0.0f;
  if (any_sampled)
    for (auto& ln : jl)
      for (uint32_t r = 0; r < ln.M; r++)
        row_samp.push_back({ln.m->temp, ln.m->top_p, ln.m->top_k, uint32_t(ln.m->sample_seed),
                            uint32_t(ln.m->sample_seed >> 32), ln.p0 + r + 1});
  pre_frag_M = 0;
  Rec rc;
  begin(rc, profile);
  lane_rows = true;
  encode_trunk(rc, M, mode, jl[0].p0);
  lane_rows = false;
  row_samp.clear();
  const double ms = finish(rc, profile);
  if (gpu_ms) *gpu_ms = ms;
  std::vector<std::vector<uint32_t>> out(lanes.size());
  if (logits_out) logits_out->assign(lanes.size(), {});
  for (auto& ln : jl) {
    const size_t k = size_t(ln.slot);
    out[k].assign((const uint32_t*)amax_i.contents + ln.row0, (const uint32_t*)amax_i.contents + ln.row0 + ln.M);
    if (logits_out)
      (*logits_out)[k].assign((const float*)logits.contents + size_t(ln.row0) * vocab,
                              (const float*)logits.contents + size_t(ln.row0 + ln.M) * vocab);
    if (mode == 0) {
      ln.m->pos += ln.M;
      ln.m->conv_par ^= 1;
      ln.m->ple_par ^= 1;
      ln.m->history.insert(ln.m->history.end(), toks[k].begin(), toks[k].end());
    }
  }
  jl.clear();
  return out;
}

// After a joint verify on driver `drv`: make this lane's first j rows (rows row0.. of the step)
// permanent, from the layer inputs the step left in the driver's scratch.
void Model::commit_lane(Model& drv, uint32_t row0, uint32_t j) {
  if (j == 0) return;
  Rec rc;
  begin(rc, nullptr);
  encode_commit_lane(rc, drv, row0, j);
  finish(rc, nullptr);
}

// commit_lane's work, encoded into rc (so several lanes' commits share one command buffer); the
// position, parities and history advance now, since the encoded commands already carry them.
void Model::encode_commit_lane(Rec& rc, Model& drv, uint32_t row0, uint32_t j) {
  if (j == 0) return;
  const size_t W = size_t(H) * HC;
  for (size_t li = 0; li < L.size(); li++) {
    Layer& l = L[li];
    Layer& dl = drv.L[li];
    id<MTLComputeCommandEncoder> e;
    if (l.ple) {
      PleArgs pa{j, 1, l.ple_kv.N, H, HC, ple_par, 9, 3, 4, 2};
      e = rc.op("ple_conv");
      [e setComputePipelineState:k_ple_conv];
      [e setBuffer:drv.h offset:row0 * W * 4 atIndex:0];
      [e setBuffer:drv.ple_gv offset:row0 * W * 4 atIndex:1];
      [e setBuffer:drv.ple_gvn offset:row0 * W * 4 atIndex:2];
      [e setBuffer:l.plehist offset:0 atIndex:3];
      [e setBuffer:l.ple_conv.b offset:l.ple_conv.off atIndex:4];
      [e setBytes:&pa length:sizeof pa atIndex:5];
      [e dispatchThreads:MTLSizeMake(H * HC, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    if (l.attn) continue;
    GdnArgs ga{j, 1, l.in.N, nk, nv, dk, dv, conv_par, 2, row0};
    e = rc.op("gdn");
    [e setComputePipelineState:k_gdn];
    [e setBuffer:dl.gin offset:size_t(row0) * l.in.N * 4 atIndex:0];
    [e setBuffer:l.convh offset:0 atIndex:1];
    [e setBuffer:l.conv.b offset:l.conv.off atIndex:2];
    [e setBuffer:l.S offset:0 atIndex:3];
    [e setBuffer:l.A_log.b offset:l.A_log.off atIndex:4];
    [e setBuffer:l.dt_bias.b offset:l.dt_bias.off atIndex:5];
    [e setBuffer:l.gnorm.b offset:l.gnorm.off atIndex:6];
    [e setBuffer:drv.of offset:0 atIndex:7];
    [e setBytes:&ga length:sizeof ga atIndex:8];
    pf = {};
    [e dispatchThreadgroups:MTLSizeMake(nv, 1, 1 + pf_ride(e, nv, 1, 1024, 0)) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
  }
  conv_par ^= 1;
  ple_par ^= 1;
  pos += j;
  history.insert(history.end(), pending_toks.begin(), pending_toks.begin() + j);
}

// After a verify forward: make the first j rows permanent (bit-identical to having fed them
// with mode 0), by replaying their recurrent updates from the stored layer inputs.
void Model::commit(uint32_t j) {
  if (j == 0) return;
  Rec rc;
  begin(rc, nullptr);
  encode_commit(rc, j);
  finish(rc, nullptr);
}

std::vector<uint32_t> Model::step(const std::vector<uint32_t>& toks, double* gpu_ms, std::vector<float>* logits_out,
                                  std::map<std::string, double>* profile) {
  return forward(toks, 0, gpu_ms, logits_out, profile);
}

// MTP head over M <= 4 (token, hidden) pairs at its own positions mtp_pos.. .
// hidden rows come from src[src0 ..]; outputs: argmax per row, pre-mixer hidden in mtp_h.
// Encode one MTP call: tokens from host_toks (written now) or, when null, the previous
// MTP call's argmax (already copied into ids on the GPU). out_slot >= 0 copies the last
// row's argmax to draft_out[out_slot] and chains it into ids for the next call; the last
// row's pre-mixer streams go to seed_h.
void Model::encode_mtp(Rec& rc, uint32_t M, const uint32_t* host_toks, id<MTLBuffer> src, uint32_t src0, int slot,
                       int out_slot) {
  if (M == 0 || M * HC > MP) mtl::die("mtp: too many rows");
  if (mtp_pos + M > cap) mtl::die("mtp: draft-head KV capacity exceeded");
  nsel = nsel_mtp;
  id<MTLBuffer> idsb = mtp_ids[slot];
  if (host_toks) memcpy(idsb.contents, host_toks, M * 4);
  set_rope(mtp_pos, M, M > MAXM ? 0 : slot);
  copy(rc, src, src0, hsrc, 0, M, H * HC);
  EmbArgs ea{M, H, 1};
  id<MTLComputeCommandEncoder> e = rc.op("embed");
  [e setComputePipelineState:k_embed];
  [e setBuffer:idsb offset:0 atIndex:0];
  [e setBuffer:embed.b offset:embed.off atIndex:1];
  [e setBuffer:erows offset:0 atIndex:2];
  [e setBytes:&ea length:sizeof ea atIndex:3];
  [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
  struct { uint32_t D, K, src_stride, src_row0; } ra{H, H, H, 0}, rh{H * HC, H, H * HC, 0};
  e = rc.op("rms_frag");
  [e setComputePipelineState:k_rms_frag];
  [e setBuffer:erows offset:0 atIndex:0];
  [e setBuffer:mtp_norm_emb.b offset:mtp_norm_emb.off atIndex:1];
  [e setBuffer:xf offset:0 atIndex:2];
  [e setBytes:&ra length:sizeof ra atIndex:3];
  [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
  mm(rc, "mm.mtp.fc", mtp_fc_emb, xf, pe, M);
  e = rc.op("rms_frag");
  [e setComputePipelineState:k_rms_frag];
  [e setBuffer:hsrc offset:0 atIndex:0];
  [e setBuffer:mtp_norm_hid.b offset:mtp_norm_hid.off atIndex:1];
  [e setBuffer:hnf offset:0 atIndex:2];
  [e setBytes:&rh length:sizeof rh atIndex:3];
  [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
  mm(rc, "mm.mtp.fc", mtp_fc_hid, hnf, ph, M * HC);
  struct { uint32_t H, HC; } fa{H, HC};
  e = rc.op("mtp_fuse");
  [e setComputePipelineState:k_mtp_fuse];
  [e setBuffer:pe offset:0 atIndex:0];
  [e setBuffer:ph offset:0 atIndex:1];
  [e setBuffer:h offset:0 atIndex:2];
  [e setBytes:&fa length:sizeof fa atIndex:3];
  [e dispatchThreads:MTLSizeMake(H * HC, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  bool pending = false;
  encode_layer(rc, mtp, M, mtp_pos, 0, pending);
  uint32_t MH = M;
  if (M > MAXM) {  // prefill: every row's streams kept, logits for the last row only
    HcArgs ha{M, H, HC, 1, prev_dn_N, R, topk};
    e = rc.op("hc_pre");
    [e setComputePipelineState:k_hc_pre];
    [e setBuffer:h offset:0 atIndex:0];
    [e setBuffer:moe_o offset:0 atIndex:1];
    [e setBuffer:ydn offset:0 atIndex:2];
    [e setBuffer:mtp_mix_norm.b offset:mtp_mix_norm.off atIndex:3];
    [e setBuffer:hn offset:0 atIndex:4];
    [e setBuffer:hnf offset:0 atIndex:5];
    [e setBytes:&ha length:sizeof ha atIndex:6];
    [e dispatchThreadgroups:MTLSizeMake(M, HC, 1 + pf_ride(e, M, HC, hct, 0)) threadsPerThreadgroup:MTLSizeMake(hct, 1, 1)];
    copy(rc, h, 0, mtp_h, 0, M, H * HC);
    copy(rc, mtp_h, M - 1, h, 0, 1, H * HC);
    pending = false;
    MH = 1;
  }
  encode_head(rc, mtp_mix_dn, mtp_mix_up, mtp_mix_norm, MH, pending, (M > MAXM ? mtp_pos + M - 1 : mtp_pos) + 1,
              has_draft_head ? &mtp_head : nullptr);
  if (M <= MAXM) copy(rc, h, 0, mtp_h, 0, M, H * HC);
  copy(rc, mtp_h, M - 1, seed_h, 0, 1, H * HC);
  if (out_slot >= 0) {
    copy(rc, amax_i, MH - 1, draft_out, out_slot, 1, 1);
    copy(rc, amax_m, MH - 1, draft_m, out_slot, 1, 1);
    if (slot + 1 < NSLOT) copy(rc, amax_i, MH - 1, mtp_ids[slot + 1], 0, 1, 1);
  }
  mtp_pos += M;
}

std::vector<uint32_t> Model::mtp_forward(const std::vector<uint32_t>& toks, id<MTLBuffer> src, uint32_t src0,
                                         double* gpu_ms) {
  const uint32_t M = uint32_t(toks.size());
  for (uint32_t r = 0; r < M; r++) ((uint32_t*)nsel_mtp.contents)[r] = 0xFFFFFFFFu;
  Rec rc;
  begin(rc, mtp_profile);
  encode_mtp(rc, M, toks.data(), src, src0, 1, -1);
  const double ms = finish(rc, mtp_profile);
  if (gpu_ms) *gpu_ms = ms;
  const uint32_t MH = M > MAXM ? 1 : M;
  std::vector<uint32_t> out(MH);
  memcpy(out.data(), amax_i.contents, MH * 4);
  return out;
}



size_t Model::ckpt_bytes() const {
  size_t n = size_t(H) * HC * 4;
  for (auto& l : L)
    for (id<MTLBuffer> b : {l.S, l.convh, l.plehist})
      if (b) n += b.length;
  return n;
}

// GPU blits into pooled buffers, queued behind the work that produced the state.
void Model::save_ckpt() {
  Checkpoint c;
  c.pos = pos;
  c.conv_par = conv_par;
  c.ple_par = ple_par;
  if (!ckpt_pool.empty()) { c.blob = ckpt_pool.back(); ckpt_pool.pop_back(); }
  else c.blob = C.buf(ckpt_bytes());
  id<MTLCommandBuffer> cb = [C.q commandBuffer];
  id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
  size_t off = 0;
  for (auto& l : L)
    for (id<MTLBuffer> b : {l.S, l.convh, l.plehist})
      if (b) { [bl copyFromBuffer:b sourceOffset:0 toBuffer:c.blob destinationOffset:off size:b.length]; off += b.length; }
  [bl copyFromBuffer:h_last_row sourceOffset:0 toBuffer:c.blob destinationOffset:off size:size_t(H) * HC * 4];
  [bl endEncoding];
  [cb commit];
  for (size_t bp : branch_pts)
    if (c.pos <= bp && bp < size_t(c.pos) + CKPT) c.pinned = ++pin_tick;
  ckpts.push_back(c);
  // keep the ckpt_keep newest (8), plus every ckpt_far (8192) tokens further back (~120 MB each)
  std::vector<Checkpoint> kept;
  for (size_t i = 0; i < ckpts.size(); i++) {
    if (i + ckpt_keep >= ckpts.size() || ckpts[i].pos % ckpt_far == 0 || ckpts[i].pinned) kept.push_back(ckpts[i]);
    else ckpt_pool.push_back(ckpts[i].blob);
  }
  ckpts.swap(kept);
}

bool Model::restore_ckpt(size_t common) {
  while (!ckpts.empty() && ckpts.back().pos > common) { ckpt_pool.push_back(ckpts.back().blob); ckpts.pop_back(); }
  if (ckpts.empty()) return false;
  // a request branched from this checkpoint: the next branch (another conversation on the same
  // system prompt) will most likely want it too, so it is exempt from the retention policy.
  // At most ckpt_pins stay pinned; the least recently used pin is released.
  ckpts.back().pinned = ++pin_tick;
  {
    size_t np = 0, oldest = 0;
    for (size_t i = 0; i < ckpts.size(); i++)
      if (ckpts[i].pinned) { if (!np || ckpts[i].pinned < ckpts[oldest].pinned) oldest = i; np++; }
    if (np > ckpt_pins) ckpts[oldest].pinned = 0;
  }
  const Checkpoint& c = ckpts.back();
  id<MTLCommandBuffer> cb = [C.q commandBuffer];
  id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
  size_t off = 0;
  for (auto& l : L)
    for (id<MTLBuffer> b : {l.S, l.convh, l.plehist})
      if (b) { [bl copyFromBuffer:c.blob sourceOffset:off toBuffer:b destinationOffset:0 size:b.length]; off += b.length; }
  [bl copyFromBuffer:c.blob sourceOffset:off toBuffer:h_last_row destinationOffset:0 size:size_t(H) * HC * 4];
  [bl endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  pos = c.pos;
  conv_par = c.conv_par;
  ple_par = c.ple_par;
  history.resize(pos);
  lk.clear();
  return true;
}

void Lookup::truncate(const std::vector<uint32_t>& hist, size_t newsize) {
  const size_t lo = std::max<size_t>(newsize, LK_N);
  for (size_t p = std::min(n, hist.size()); p-- > lo;) {
    auto it = at.find(key(&hist[p - LK_N]));
    if (it != at.end() && !it->second.empty() && it->second.back() == p) it->second.pop_back();
  }
  n = std::min(n, lo);
}

Checkpoint Model::snapshot() {
  Checkpoint c;
  c.pos = pos;
  c.conv_par = conv_par;
  c.ple_par = ple_par;
  if (!ckpt_pool.empty()) { c.blob = ckpt_pool.back(); ckpt_pool.pop_back(); }
  else c.blob = C.buf(ckpt_bytes());
  id<MTLCommandBuffer> cb = [C.q commandBuffer];
  id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
  size_t off = 0;
  for (auto& l : L)
    for (id<MTLBuffer> b : {l.S, l.convh, l.plehist})
      if (b) { [bl copyFromBuffer:b sourceOffset:0 toBuffer:c.blob destinationOffset:off size:b.length]; off += b.length; }
  [bl copyFromBuffer:h_last_row sourceOffset:0 toBuffer:c.blob destinationOffset:off size:size_t(H) * HC * 4];
  [bl endEncoding];
  [cb commit];
  return c;
}

void Model::rollback(const Checkpoint& c) {
  id<MTLCommandBuffer> cb = [C.q commandBuffer];
  id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
  size_t off = 0;
  for (auto& l : L)
    for (id<MTLBuffer> b : {l.S, l.convh, l.plehist})
      if (b) { [bl copyFromBuffer:c.blob sourceOffset:off toBuffer:b destinationOffset:0 size:b.length]; off += b.length; }
  [bl copyFromBuffer:c.blob sourceOffset:off toBuffer:h_last_row destinationOffset:0 size:size_t(H) * HC * 4];
  [bl endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  lk.truncate(history, c.pos);
  pos = c.pos;
  conv_par = c.conv_par;
  ple_par = c.ple_par;
  history.resize(pos);
}

void Model::kv_copy(id<MTLBlitCommandEncoder> bl, const Layer& s, uint32_t scap, Layer& d, size_t P) const {
  if (!s.Kc || !d.Kc || !P) return;
  for (auto [a, b] : {std::pair{s.Kc, d.Kc}, {s.Vc, d.Vc}}) {
    if (!kv8) { [bl copyFromBuffer:a sourceOffset:0 toBuffer:b destinationOffset:0 size:P * nkv * hd * 2]; continue; }
    [bl copyFromBuffer:a sourceOffset:0 toBuffer:b destinationOffset:0 size:P * nkv * hd];
    [bl copyFromBuffer:a sourceOffset:size_t(scap) * nkv * hd toBuffer:b destinationOffset:size_t(cap) * nkv * hd
                  size:P * nkv * (hd / 32) * 2];
  }
}

void Model::fork_from(const Model& src) {
  if (src.pos > cap || src.L.size() != L.size()) mtl::die("fork_from: source does not fit this slot");
  for (auto& c : ckpts) ckpt_pool.push_back(c.blob);
  ckpts.clear();
  branch_pts.clear();
  const size_t P = src.pos;
  id<MTLCommandBuffer> cb = [C.q commandBuffer];
  id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
  auto cp = [&](id<MTLBuffer> a, id<MTLBuffer> b, size_t n) {
    if (a && b && n) [bl copyFromBuffer:a sourceOffset:0 toBuffer:b destinationOffset:0 size:n];
  };
  for (size_t li = 0; li < L.size(); li++) {
    const Layer& s = src.L[li];
    Layer& d = L[li];
    if (d.attn) {  // [pos][...] caches: the rows below P; block keys of the blocks complete at P
      kv_copy(bl, s, src.cap, d, P);
      cp(s.rawk, d.rawk, P * ihd * 2);
      cp(s.bk, d.bk, (P / ratio) * ihd * 4);
    }
    for (auto [a, b] : {std::pair{s.S, d.S}, {s.convh, d.convh}, {s.plehist, d.plehist}})
      if (a) cp(a, b, a.length);
  }
  cp(src.h_last_row, h_last_row, size_t(H) * HC * 4);
  [bl endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  pos = src.pos;
  conv_par = src.conv_par;
  ple_par = src.ple_par;
  history = src.history;
  lk.clear();
  mtp_synced = false;
  mtp_pos = 0;
}

size_t Model::ckpt_below(size_t common) const {
  size_t p = 0;
  for (auto& c : ckpts) if (c.pos <= common && c.pos <= pos) p = std::max<size_t>(p, c.pos);
  return p;
}

size_t Model::fork_prefix(Model& src, size_t common) {
  if (&src == this || src.L.size() != L.size()) return 0;
  const Checkpoint* ck = nullptr;
  for (auto& c : src.ckpts)
    if (c.pos <= common && c.pos <= src.pos && c.pos <= cap && (!ck || c.pos > ck->pos)) ck = &c;
  if (!ck) return 0;
  // more siblings on the same prefix will likely follow: keep this checkpoint (as restore_ckpt does)
  const_cast<Checkpoint*>(ck)->pinned = ++src.pin_tick;
  {
    size_t np = 0, oldest = 0;
    for (size_t i = 0; i < src.ckpts.size(); i++)
      if (src.ckpts[i].pinned) { if (!np || src.ckpts[i].pinned < src.ckpts[oldest].pinned) oldest = i; np++; }
    if (np > src.ckpt_pins && &src.ckpts[oldest] != ck) src.ckpts[oldest].pinned = 0;
  }
  for (auto& c : ckpts) ckpt_pool.push_back(c.blob);
  ckpts.clear();
  branch_pts.clear();
  const size_t P = ck->pos;
  id<MTLCommandBuffer> cb = [C.q commandBuffer];
  id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
  auto cp = [&](id<MTLBuffer> a, size_t ao, id<MTLBuffer> b, size_t n) {
    if (a && b && n) [bl copyFromBuffer:a sourceOffset:ao toBuffer:b destinationOffset:0 size:n];
  };
  size_t off = 0;  // the checkpoint blob: per layer S, convh, plehist, then the last-row hidden
  for (size_t li = 0; li < L.size(); li++) {
    const Layer& s = src.L[li];
    Layer& d = L[li];
    if (d.attn) {  // positional caches: the source's rows below P are final
      kv_copy(bl, s, src.cap, d, P);
      cp(s.rawk, 0, d.rawk, P * ihd * 2);
      cp(s.bk, 0, d.bk, (P / ratio) * ihd * 4);
    }
    for (auto [a, b] : {std::pair{s.S, d.S}, {s.convh, d.convh}, {s.plehist, d.plehist}})
      if (a) { cp(ck->blob, off, b, a.length); off += a.length; }
  }
  cp(ck->blob, off, h_last_row, size_t(H) * HC * 4);
  if (has_mtp && mtp.attn) {  // draft-head cache, valid where the source fed with its MTP pass
    kv_copy(bl, src.mtp, src.cap, mtp, P);
    cp(src.mtp.rawk, 0, mtp.rawk, P * ihd * 2);
    cp(src.mtp.bk, 0, mtp.bk, (P / ratio) * ihd * 4);
  }
  [bl endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  pos = uint32_t(P);
  conv_par = ck->conv_par;
  ple_par = ck->ple_par;
  history.assign(src.history.begin(), src.history.begin() + long(P));
  lk.clear();
  mtp_synced = false;
  mtp_pos = 0;
  return P;
}

void Lookup::sync(const std::vector<uint32_t>& hist) {
  if (n > hist.size()) clear();
  for (size_t p = std::max<size_t>(n, LK_N); p < hist.size(); p++) at[key(&hist[p - LK_N])].push_back(uint32_t(p));
  n = std::max<size_t>(hist.size(), LK_N);
}

uint32_t Lookup::find(const std::vector<uint32_t>& hist, const std::vector<uint32_t>& tail, uint32_t* len) const {
  *len = 0;
  if (tail.size() < LK_N) return NONE;
  auto it = at.find(key(&tail[tail.size() - LK_N]));
  if (it == at.end()) return NONE;
  const auto& v = it->second;
  uint32_t best = NONE, bl = 0, seen = 0;
  for (size_t i = v.size(); i-- > 0 && seen < CAND; seen++) {  // most recent first
    const uint32_t p = v[i];
    if (p >= hist.size()) continue;
    uint32_t m = 0;
    while (m < MAXLEN && m < p && m < tail.size() && hist[p - 1 - m] == tail[tail.size() - 1 - m]) m++;
    if (m > bl) { bl = m; best = p; }
  }
  if (bl < LK_N) return NONE;  // hash collision or stale entry
  *len = bl;
  return best;
}

// Feed tokens through the trunk (and the MTP head, when speculating) and return the trunk's
// next-token argmax. MTP pairs (H_p, x_{p+1}): the last pair of the suffix uses the bonus.
uint32_t Model::feed(const std::vector<uint32_t>& toks, bool with_mtp, SpecStats* st) {
  static const bool PRE_ROWS = !(getenv("OPUS_PRE_ROWS") && atoi(getenv("OPUS_PRE_ROWS")) == 0);
  uint32_t last = 0;
  size_t i = 0;
  dispatch_group_t grp = nil;
  std::shared_ptr<std::vector<uint16_t>> nx;
  pre_frag_M = 0;
  while (i < toks.size()) {
    // chunks end on MP-aligned positions so checkpoints land on CKPT multiples
    const size_t room = MP - pos % MP;
    std::vector<uint32_t> c(toks.begin() + i, toks.begin() + std::min(toks.size(), i + room));
    // the n-gram rows of this chunk were fetched while the previous one ran; start the next
    // chunk's fetch now (random reads from the SSD-resident table, ~45 us per token when cold)
    if (grp) {
      dispatch_group_wait(grp, DISPATCH_TIME_FOREVER);
      grp = nil;
      pre_frag.swap(*nx);
      pre_frag_M = uint32_t(c.size());
    }
    const size_t i2 = i + c.size();
    if (PRE_ROWS && i2 < toks.size()) {
      const uint32_t m2 = uint32_t(std::min<size_t>(MP, toks.size() - i2));
      auto hist2 = std::make_shared<std::vector<uint32_t>>(history);
      hist2->insert(hist2->end(), toks.begin() + i, toks.begin() + i2 + m2);
      nx = std::make_shared<std::vector<uint16_t>>(size_t((m2 + 15) / 16 * 16) * (ngram->ngram_size - 1) * ngram->heads_per * ngram->dim);
      grp = dispatch_group_create();
      const NgramTable* tab = ngram;
      auto out = nx;
      dispatch_group_async(grp, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{ tab->rows_frag(*hist2, m2, out->data()); });
    }
    double ms;
    last = forward(c, 0, &ms).back();
    if (st) st->prefill_ms += ms;
    {
      Rec rc;
      begin(rc, nullptr);
      copy(rc, h_tgt, uint32_t(c.size()) - 1, h_last_row, 0, 1, H * HC);
      finish(rc, nullptr);
    }
    if (with_mtp) {
      for (uint32_t r = 0; r < c.size(); r += MP / HC) {
        const uint32_t m = std::min<uint32_t>(MP / HC, uint32_t(c.size()) - r);
        std::vector<uint32_t> t(m);
        for (uint32_t k = 0; k < m; k++) {
          const size_t p = i + r + k + 1;
          t[k] = p < toks.size() ? toks[p] : last;
        }
        auto d = mtp_forward(t, h_tgt, r, &ms);
        if (st) st->prefill_ms += ms;
        seed = d.back();
        copy_mtp_to_seed(m - 1);
      }
    }
    i += c.size();
    if (pos % CKPT == 0) save_ckpt();
  }
  mtp_synced = with_mtp;
  return last;
}

// Tolerant acceptance (OPUS_TOL), one verify row: may draft c stand in for g, the token the target
// emitted? p(c)/p(g) = exp(z_c - z_g), so the test is a logit difference, scaled by the temperature
// (the sampling distribution at T is p^(1/T)). A stop token (eos or any of the request's stops) is
// never a near-tie in either direction. A sampled row also needs c inside sample_rows' own support
// (the K highest logits, ties to the lower index, then the top-p prefix at T): a wide window
// (high T) or g near the support's edge would otherwise emit a token the request excludes.
static bool tol_ok(const float* lg, size_t NV, uint32_t g, uint32_t c, float tol, bool stop, float temp, float top_p,
                   uint32_t top_k) {
  if (tol <= 0.0f || c >= NV) return false;
  if (lg[g] - lg[c] > tol * (temp > 0.0f ? temp : 1.0f)) return false;
  // OPUS_TOL_TRACE=1: log each near-tie refused for a stop token or for falling outside the support
  static const bool trace = getenv("OPUS_TOL_TRACE") != nullptr;
  if (stop) {
    if (trace) fprintf(stderr, "opus: tol refused stop (draft %u, target %u)\n", c, g);
    return false;
  }
  if (temp <= 0.0f) return true;
  const uint32_t K = std::min<uint32_t>(top_k ? top_k : 64u, 64u);
  float tv[64];
  uint32_t ti[64], n = 0;
  auto ahead = [](float za, uint32_t ia, float zb, uint32_t ib) { return za > zb || (za == zb && ia < ib); };
  for (uint32_t v = 0; v < NV; v++) {
    const float z = lg[v];
    if (z != z || (n == K && !ahead(z, v, tv[n - 1], ti[n - 1]))) continue;
    uint32_t j = n < K ? n++ : K - 1;
    for (; j > 0 && ahead(z, v, tv[j - 1], ti[j - 1]); j--) { tv[j] = tv[j - 1]; ti[j] = ti[j - 1]; }
    tv[j] = z; ti[j] = v;
  }
  uint32_t at = n;
  for (uint32_t i = 0; i < n; i++)
    if (ti[i] == c) { at = i; break; }
  bool in = at < n;
  if (in) {
    const float mx = tv[0] / temp;
    float den = 0.0f, cum = 0.0f;
    for (uint32_t i = 0; i < n; i++) den += std::exp(tv[i] / temp - mx);
    for (uint32_t i = 0; i < at && in; i++)  // sample_rows keeps the prefix up to the first cumulative >= top_p
      in = (cum += std::exp(tv[i] / temp - mx) / den) < top_p;
  }
  if (!in && trace) fprintf(stderr, "opus: tol refused support (draft %u rank %u of K %u, T %.3g, top_p %.3g)\n", c, at, K, temp, top_p);
  return in;
}

// Continue the conversation with `prompt`: reuses everything already fed when prompt
// extends it, else starts over. Greedy; emits through cb (return false to stop).
// With spec, block-4 MTP speculation; the emitted tokens are identical either way.
uint32_t Model::generate(const std::vector<uint32_t>& prompt, uint32_t max_new, bool spec,
                         const std::function<bool(uint32_t)>& cb, SpecStats* st) {
  spec = spec && has_mtp;
  const auto tg0 = std::chrono::steady_clock::now();
  size_t reuse = 0;
  if (prompt.size() > history.size() && std::equal(history.begin(), history.end(), prompt.begin()) &&
      (!spec || mtp_synced)) {
    reuse = history.size();
  } else {
    // longest common prefix; resume from the last recurrent-state checkpoint inside it
    size_t common = 0;
    while (common < history.size() && common < prompt.size() - 1 && history[common] == prompt[common]) common++;
    if (common >= CKPT) {  // a branch point: later prefills keep the checkpoint just below it
      branch_pts.erase(std::remove(branch_pts.begin(), branch_pts.end(), common), branch_pts.end());
      branch_pts.push_back(common);
      if (branch_pts.size() > ckpt_pins) branch_pts.erase(branch_pts.begin());
    }
    if ((!spec || mtp_synced) && restore_ckpt(common)) {
      reuse = pos;
      if (spec) {
        // KV and MTP caches are positional: truncate; redo the MTP pair (H_{pos-1}, x_pos)
        mtp_pos = pos - 1;
        auto d = mtp_forward({prompt[pos]}, h_last_row, 0);
        seed = d.back();
        copy_mtp_to_seed(0);
      }
    } else {
      reset();
    }
  }
  if (st) st->reused = uint32_t(reuse);
  std::vector<uint32_t> suffix(prompt.begin() + reuse, prompt.end());
  if (pos + suffix.size() + max_new + 8 > cap) mtl::die("generate: KV capacity exceeded");
  const auto tf0 = std::chrono::steady_clock::now();
  const double th_0 = t_host, te_0 = t_encode;
  uint32_t x = feed(suffix, spec, st);
  if (getenv("OPUS_PFT"))
    fprintf(stderr, "prefill: setup %.0f ms; %zu tokens, wall %.0f ms, GPU %.0f ms, host prep %.0f ms, encode %.0f ms\n",
            std::chrono::duration<double, std::milli>(tf0 - tg0).count(), suffix.size(),
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tf0).count(), st ? st->prefill_ms : 0.0,
            t_host - th_0, t_encode - te_0);
  uint32_t n = 0;
  if (!cb(x) || ++n >= max_new) return n;
  if (!spec) {
    while (n < max_new) {
      double ms;
      x = forward({x}, 0, &ms)[0];
      if (st) { st->verify_ms += ms; st->rounds++; }
      if (!cb(x) || ++n >= max_new) break;
    }
    mtp_synced = false;
    return n;
  }
  // Each round: [commit of the last verify, MTP accept call, K-1 chained draft calls], then
  // the CPU builds the candidate chain and the verify of up to MAXM rows follows. K adapts to
  // recent acceptance: deep when drafts keep landing (code), shallow when they don't (prose).
  //
  // Hybrid chain: a verify step costs ~13 ms fixed plus ~1.1 ms per row, so rows past the MTP
  // drafts are nearly free. The chain starts with the MTP drafts and continues from the place
  // the current tail last occurred in the context (Lookup). Where both propose a token and
  // disagree, the source with the higher measured acceptance wins. The verify and commit are
  // unchanged, so the emitted tokens are identical to plain decoding whatever the chain holds.
  bool first = true;
  uint32_t a_prev = 0, keep_prev = 0;
  std::vector<uint32_t> t_acc;
  double acc_rate = 0.7;  // EMA of per-draft acceptance
  const auto loop_t0 = std::chrono::steady_clock::now();
  struct LoopWall {  // wall time of the speculation loop, whichever way it exits
    SpecStats* st; std::chrono::steady_clock::time_point t0;
    ~LoopWall() { if (st) st->wall_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
  } loop_wall{st, loop_t0};
  static const bool LK_ON = !(getenv("OPUS_LK") && atoi(getenv("OPUS_LK")) == 0);
  static const double PMIN = getenv("OPUS_PMIN") ? atof(getenv("OPUS_PMIN")) : 0.15;
  // Streak mode: once the lookup has predicted SMIN tokens in a row, bet a chunk of at most half
  // that run on the continuation and verify it through the PREFILL path (mode 0, ~0.45 ms per
  // row, bit-identical to the decode path) with a prediction for every row. All rows right:
  // the chunk is already committed. A wrong row: roll the recurrent state back to the
  // snapshot and feed the accepted prefix again. Either way the emitted tokens are exactly
  // what plain decoding emits.
  static const uint32_t SMIN = getenv("OPUS_STREAK_MIN") ? uint32_t(atoi(getenv("OPUS_STREAK_MIN"))) : 48;
  uint32_t good = 0;     // consecutive accepted tokens that the lookup proposed
  bool pending = false;  // a verify is waiting for its commit
  auto is_stop = [&](uint32_t t) { return t == eos || std::find(stop_ids.begin(), stop_ids.end(), t) != stop_ids.end(); };
  auto settle = [&] {    // commit the last verify and feed its accept pairs to the MTP head
    commit(a_prev + 1);
    Rec rs;
    begin(rs, nullptr);
    for (uint32_t r = 0; r < MAXM; r++) ((uint32_t*)nsel_mtp.contents)[r] = 0xFFFFFFFFu;
    encode_mtp(rs, uint32_t(t_acc.size()), t_acc.data(), h_tgt, keep_prev, 1, 0);
    finish(rs, nullptr);
    seed = ((const uint32_t*)draft_out.contents)[0];
    pending = false;
    first = true;
  };
  bool stopped = false;
  while (n < max_new) {
    if (LK_ON && SMIN && good >= SMIN && max_new - n >= 24) {
      if (pending) settle();
      lk.sync(history);
      std::vector<uint32_t> tail(history.end() - std::min<size_t>(history.size(), Lookup::MAXLEN), history.end());
      tail.push_back(x);
      uint32_t cl = 0;
      const uint32_t cp = lk.find(history, tail, &cl);
      const uint32_t bet = good >= 512 ? 256 : good >= 256 ? 128 : good >= 128 ? 64 : 32;
      uint32_t nr = 0;
      if (cp != Lookup::NONE && cl >= 16) {
        nr = std::min<uint32_t>(std::min<uint32_t>(bet, SROWS), max_new - n);
        nr = uint32_t(std::min<size_t>(nr, history.size() - cp + 1));
      }
      std::vector<uint32_t> v{x};
      for (uint32_t i = 0; i + 1 < nr; i++) {
        v.push_back(history[cp + i]);
        if (is_stop(history[cp + i])) break;
      }
      nr = uint32_t(v.size());
      if (nr >= 24) {
        const auto ts0 = std::chrono::steady_clock::now();
        Checkpoint s0 = snapshot();
        double ms = 0;
        want_all = true;
        const std::vector<uint32_t> g = forward(v, 0, &ms);
        want_all = false;
        uint32_t a = 0;
        while (a + 1 < nr && g[a] == v[a + 1]) a++;
        uint32_t nt = g[a];
        double ms2 = 0;
        if (a + 1 == nr) {  // every row right: committed as fed; pair the rows for the MTP head
          std::vector<uint32_t> t(v.begin() + 1, v.end());
          t.push_back(nt);
          const auto d = mtp_forward(t, h_tgt, 0, &ms2);
          seed = d.back();
          copy_mtp_to_seed(nr - 1);
          good += a;
        } else {
          rollback(s0);
          SpecStats tmp;
          const uint32_t last = feed(std::vector<uint32_t>(v.begin(), v.begin() + a + 1), true, &tmp);
          ms2 = tmp.prefill_ms;
          if (last != nt) {  // exactness broken: continue from the re-fed (committed) state instead of exiting
            fprintf(stderr, "opus: WARNING streak re-feed gave token %u where the chunk gave %u (exactness broken)\n", last, nt);
            nt = last;
          }
          good = 0;
        }
        ckpt_pool.push_back(s0.blob);
        if (st) {
          st->verify_ms += ms + ms2;
          st->rounds++;
          st->rows += nr;
          st->drafted += nr - 1;
          st->accepted += a;
          st->lk_proposed += nr - 1;
          st->lk_tested += std::min(a + 1, nr - 1);
          st->lk_accepted += a;
          st->streak_chunks++;
          st->streak_fail += a + 1 != nr;
          if (getenv("OPUS_TRACE")) fprintf(stderr, "TRACE streak rows %u accepted %u %s\n", nr, a, a + 1 != nr ? "ROLLBACK" : "ok");
          st->streak_tokens += a + 1;
          st->streak_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts0).count();
        }
        x = nt;
        first = true;
        bool go = true;
        for (uint32_t k = 0; k < a && go; k++) go = cb(v[k + 1]) && ++n < max_new;
        if (go) go = cb(nt) && ++n < max_new;
        if (!go) { stopped = true; break; }
        continue;
      }
      good = 0;  // no usable continuation: earn the next streak again
    }
    static const bool conflog = getenv("OPUS_CONFLOG") != nullptr;
    static const bool trace = getenv("OPUS_TRACE") != nullptr;
    static const uint32_t KLO = getenv("OPUS_KLO") ? atoi(getenv("OPUS_KLO")) : 3;
    static const double KTH = getenv("OPUS_KTH") ? atof(getenv("OPUS_KTH")) : 0.8;
    // Confidence-adaptive chain (OPUS_CONF=0 disables): each draft's argmax margin (best logit
    // minus runner-up) maps to an acceptance estimate through conf_rate, calibrated online.
    // While the chain so far is likely to survive, more drafts are chained in a follow-up
    // command buffer; verify rows whose cumulative estimate falls below CTH are not spent.
    static const bool CONF_ON = !(getenv("OPUS_CONF") && atoi(getenv("OPUS_CONF")) == 0);
    static const bool CONF_ALL = getenv("OPUS_CONF_ALL") != nullptr;
    static const double CTH = getenv("OPUS_CTH") ? atof(getenv("OPUS_CTH")) : 0.15;
    static const double XTH = getenv("OPUS_XTH") ? atof(getenv("OPUS_XTH")) : 0.40;
    static const uint32_t XN = getenv("OPUS_XN") ? atoi(getenv("OPUS_XN")) : 2;
    static const uint32_t KMAX = getenv("OPUS_KMAX") ? atoi(getenv("OPUS_KMAX")) : 13;
    // rows past the reply's remaining room can never be emitted, and near the slot's capacity they would
    // write K/V and draft-head rows past its end: the chain holds at most EMAX tokens after x
    const uint32_t EMAX = std::min<uint32_t>(MAXM - 1, max_new - n - 1);
    uint32_t K = std::min<uint32_t>(acc_rate > KTH ? MAXD : KLO, std::max<uint32_t>(1, EMAX));
    const bool m0ok = !first;  // draft 0's margin is known only when this round's accept call produced it
    // only where the head is unsure (K = KLO): at high acceptance the fixed deep chain is already
    // right and the estimates only cost rows
    const bool conf = CONF_ON && (CONF_ALL || K == KLO);
    Rec rc;
    begin(rc, nullptr);
    for (uint32_t r = 0; r < MAXM; r++) ((uint32_t*)nsel_mtp.contents)[r] = 0xFFFFFFFFu;
    if (!first) {
      encode_commit(rc, a_prev + 1);
      encode_mtp(rc, uint32_t(t_acc.size()), t_acc.data(), h_tgt, keep_prev, 1, 0);
    } else {
      ((uint32_t*)draft_out.contents)[0] = seed;
      ((uint32_t*)mtp_ids[2].contents)[0] = seed;
    }
    for (uint32_t k = 1; k < K; k++) encode_mtp(rc, 1, nullptr, seed_h, 0, int(k + 1), int(k));
    double dms = finish(rc, nullptr);
    first = false;
    const uint32_t* dd = (const uint32_t*)draft_out.contents;
    const float* dmg = (const float*)draft_m.contents;
    auto pc = [&](uint32_t i) { return conf && (i > 0 || m0ok) ? conf_rate[conf_bin(dmg[i])] : acc_rate; };
    if (conf) {
      double P = 1.0;
      bool stop_seen = false;
      for (uint32_t i = 0; i < K; i++) { P *= pc(i); stop_seen |= is_stop(dd[i]); }
      while (!stop_seen && P > XTH && K + XN <= KMAX && K + XN + 2 <= MAXM && K + XN <= EMAX) {
        Rec rx;
        begin(rx, nullptr);
        for (uint32_t r = 0; r < MAXM; r++) ((uint32_t*)nsel_mtp.contents)[r] = 0xFFFFFFFFu;
        const uint32_t last = dd[K - 1];
        for (uint32_t j = 0; j < XN; j++) encode_mtp(rx, 1, j == 0 ? &last : nullptr, seed_h, 0, int(j + 1), int(K + j));
        dms += finish(rx, nullptr);
        for (uint32_t j = 0; j < XN; j++) { P *= pc(K + j); stop_seen |= is_stop(dd[K + j]); }
        K += XN;
        if (st) st->ext_batches++;
      }
    }
    if (st) st->draft_ms += dms;
    std::vector<uint32_t> d(dd, dd + K);
    const uint32_t appended = K - 1;
    uint32_t Keff = K;  // leading drafts worth a verify row
    if (conf) {
      double P = 1.0;
      for (uint32_t i = 0; i < K; i++) {
        P *= pc(i);
        if (i > 0 && P < CTH) { Keff = i; break; }
      }
    }
    if (st) st->drafts_made += K;

    // candidate chain c (rows 1..E of the verify); src: 0 MTP, 1 lookup, 2 both agree
    auto th0 = std::chrono::steady_clock::now();
    std::vector<uint32_t> c, clen;
    std::vector<uint8_t> src;
    uint32_t jd = 0;  // leading chain tokens equal to the MTP drafts
    if (LK_ON) {
      lk.sync(history);
      std::vector<uint32_t> tail(history.end() - std::min<size_t>(history.size(), Lookup::MAXLEN), history.end());
      tail.push_back(x);
      uint32_t cl = 0, cp = lk.find(history, tail, &cl);
      bool mtp_ok = true;
      double p = 1.0;
      while (c.size() < EMAX) {
        const uint32_t i = uint32_t(c.size());
        const bool hm = mtp_ok && i < Keff;
        const bool hl = cp != Lookup::NONE && cp < history.size();
        if (!hm && !hl) break;
        const double pl = hl ? lk.rate[Lookup::bucket(cl)] : 0.0;
        uint32_t t;
        uint8_t s;
        const double pm = hm ? pc(i) : 0.0;
        if (hm && hl && d[i] == history[cp]) { t = d[i]; s = 2; p *= std::max(pm, pl); }
        else if (hm && (!hl || pl <= pm)) { t = d[i]; s = 0; p *= pm; }
        else {
          t = history[cp]; s = 1; p *= pl;
          if (hm && st) st->lk_override++;
          mtp_ok = false;
        }
        if (i >= Keff && p < PMIN) break;  // rows past the MTP drafts must be worth their ~1.1 ms
        if (jd == i && i < K && t == d[i]) jd++;
        c.push_back(t);
        src.push_back(s);
        clen.push_back(cl);
        tail.push_back(t);
        if (is_stop(t)) break;
        if (hl && history[cp] == t) { cp++; cl++; }
        else cp = lk.find(history, tail, &cl);
      }
    } else {
      const uint32_t ne = std::min(Keff, EMAX);
      c.assign(d.begin(), d.begin() + ne);
      src.assign(ne, 0);
      clen.assign(ne, 0);
      jd = ne;
    }
    const uint32_t E = uint32_t(c.size()), MV = E + 1;
    std::vector<uint32_t> v{x};
    v.insert(v.end(), c.begin(), c.end());
    static double tA = 0, tB = 0, tC = 0; static int tn = 0;
    auto tq0 = std::chrono::steady_clock::now();
    tA += std::chrono::duration<double, std::milli>(tq0 - th0).count();
    memcpy(ids.contents, v.data(), MV * 4);
    nsel = nsel_trunk;
    set_rope(pos, MV, 0);
    auto tq1 = std::chrono::steady_clock::now();
    tB += std::chrono::duration<double, std::milli>(tq1 - tq0).count();
    for (uint32_t r = 0; r < MV; r++) ((uint32_t*)nsel_trunk.contents)[r] = 0xFFFFFFFFu;
    // OPUS_ASYNC_ROWS=0 turns it off. The verify's n-gram rows (random reads from a ~32 GB SSD-resident table, ~1 ms
    // a round when cold) are fetched on another thread while the host encodes and the GPU runs the
    // layers before the first PLE layer, which waits for ple_event
    static const bool ASYNC_ROWS = !(getenv("OPUS_ASYNC_ROWS") && !atoi(getenv("OPUS_ASYNC_ROWS")));
    dispatch_group_t rows_grp = nil;
    {
      pending_toks = v;
      // the rows hash at most ngram_size tokens back: the history's tail is enough once it is longer
      // than that (same rows as the whole history)
      const size_t tail = ngram->ngram_size + 1;
      auto hist = std::make_shared<std::vector<uint32_t>>(history.size() > tail ? history.end() - long(tail) : history.begin(), history.end());
      hist->insert(hist->end(), v.begin(), v.end());
      if (ASYNC_ROWS) {
        const uint64_t val = ++ple_counter;
        ple_wait = val;
        rows_grp = dispatch_group_create();
        const NgramTable* tab = ngram;
        uint16_t* dst = (uint16_t*)ple_e.contents;
        id<MTLSharedEvent> evt = ple_event;
        dispatch_group_async(rows_grp, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^{
          tab->rows_frag(*hist, MV, dst);
          evt.signaledValue = val;
        });
      } else {
        ngram->rows_frag(*hist, MV, (uint16_t*)ple_e.contents);
      }
    }
    tC += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tq1).count();
    if (getenv("OPUS_HOSTT") && ++tn % 50 == 0) fprintf(stderr, "host/round: chain %.3f ms, rope %.3f ms, ngram %.3f ms\n", tA / tn, tB / tn, tC / tn);
    t_host += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - th0).count();
    if (pos + MV > cap) mtl::die("verify: KV capacity exceeded");
    Rec rv;
    begin(rv, nullptr);
    encode_trunk(rv, MV, 1, pos);
    const double vms = finish(rv, nullptr);
    if (rows_grp) {  // the GPU passed the PLE wait, so the fetch is done; no later step waits
      dispatch_group_wait(rows_grp, DISPATCH_TIME_FOREVER);
      ple_wait = 0;
    }
    if (st) { st->verify_ms += vms; st->rounds++; st->drafted += E; st->rows += MV; }
    std::vector<uint32_t> g(MV);
    memcpy(g.data(), amax_i.contents, MV * 4);
    uint32_t a = 0;
    // Tolerant acceptance (OPUS_TOL > 0, greedy only, off by default): a draft that is not the
    // target's top token is still accepted when the target scores it within TOL logits of the
    // top. The output then differs from strict greedy decoding only at such near-ties.
    // OPUS_TOL > 0 accepts a draft that is not the target's token when the target's own logits
    // score it within a likelihood ratio of the token it did emit: p(c)/p(g) = exp(z_c - z_g), so
    // the softmax denominator cancels and the test is a logit difference. At temperature T the
    // sampling distribution is p^(1/T), so the window scales with T. A greedy row has g = argmax,
    // which makes this exactly the logit-margin rule measured on 2026-11-15.
    static const float TOL_ENV = getenv("OPUS_TOL") ? float(atof(getenv("OPUS_TOL"))) : 0.0f;
    const float TOL = tol_off ? 0.0f : TOL_ENV;
    {
      const float* lg = (const float*)logits.contents;
      const size_t NV = lm_head.N;
      while (a < E) {
        if (c[a] == g[a]) { a++; continue; }
        // A stop token is never a near-tie: tolerating one would cut the reply short (draft is a stop)
        // or run past a natural stop (the top is one). eos alone is not enough: chat turns end on
        // <|im_end|>, a request stop, while the pack's eos is <|endoftext|>.
        const bool stop = is_stop(c[a]) || is_stop(g[a]);
        if (TOL > 0.0f && tol_ok(lg + size_t(a) * NV, NV, g[a], c[a], TOL, stop, temp, top_p, top_k)) {
          if (st) st->tol_accepted++;
          a++;
          continue;
        }
        break;
      }
    }
    const uint32_t nt = g[a];
    if (st) st->accepted += a;
    if (trace) {
      std::string pat;
      for (uint32_t i = 0; i < E; i++) pat += src[i] == 0 ? 'm' : src[i] == 1 ? 'L' : 'b';
      fprintf(stderr, "TRACE round K %u rows %u accepted %u chain %s good %u\n", K, MV, a, pat.c_str(), good);
    }
    // per-draft MTP acceptance over the drafts whose outcome is known: d[i] is conditioned on
    // d[0..i-1], so it counts only while the chain followed the drafts (i <= jd); the true
    // token at row a is nt.
    {
      uint32_t tested = 0, ok = 0;
      for (uint32_t i = 0; i < K && i <= a && i <= jd; i++) {
        const bool good = i < a ? (i < jd) : (nt == d[i]);
        // draft margin, outcome, and the target's own margin at the position the draft was judged
        if (conflog) fprintf(stderr, "CONF %u %.4f %d %.4f\n", i, ((const float*)draft_m.contents)[i], int(good), ((const float*)amax_m.contents)[i]);
        if (i > 0 || m0ok) { double& cr = conf_rate[conf_bin(dmg[i])]; cr = 0.95 * cr + 0.05 * (good ? 1.0 : 0.0); }
        tested++;
        ok += good;
        if (!good) break;
      }
      if (tested) acc_rate = 0.7 * acc_rate + 0.3 * (double(ok) / double(tested));
    }
    // lookup acceptance by match length, over lookup-only tokens that were actually tested
    for (uint32_t i = 0; i < E && i <= a; i++) {
      if (src[i] != 1) continue;
      double& r = lk.rate[Lookup::bucket(clen[i])];
      r = 0.9 * r + 0.1 * (i < a ? 1.0 : 0.0);
      if (st) { st->lk_tested++; st->lk_accepted += i < a; }
    }
    if (st) for (uint32_t i = 0; i < E; i++) st->lk_proposed += src[i] == 1;
    // the streak counter: accepted tokens the lookup proposed, broken by any rejection or by an
    // accepted token it did not propose
    for (uint32_t i = 0; i < a; i++) good = src[i] != 0 ? good + 1 : 0;
    if (a < E) good = 0;
    pending = true;
    // MTP cache: the chained calls appended pairs for d[0..K-2]; keep those that were accepted
    const uint32_t keep = std::min(std::min(a, appended), jd);
    mtp_pos -= appended - keep;
    t_acc.clear();
    for (uint32_t k = keep; k < a; k++) t_acc.push_back(c[k]);
    t_acc.push_back(nt);
    a_prev = a;
    keep_prev = keep;
    x = nt;
    bool go = true;
    for (uint32_t k = 0; k < a && go; k++) go = cb(c[k]) && ++n < max_new;
    if (go) go = cb(nt) && ++n < max_new;
    if (!go) break;
  }
  // settle: commit the last verify and feed the accept pairs so the state is consistent
  (void)stopped;
  if (pending) settle();
  return n;
}

// Lanes through the MTP head: lane k's rows are toks[k] with its pre-mixer streams at src[k] (buffer, row).
// One pass over the concatenated rows (at most MAXM); the MTP attention runs per lane on that lane's own
// MTP cache at its own mtp_pos, so every row's bits are those of encode_mtp on that lane alone. Each lane's
// last row leaves its pre-mixer streams in seed[k] (the next chained draft reads them). Called on the
// driver; returns each lane's last-row draft token.
std::vector<uint32_t> Model::mtp_lanes(const std::vector<Model*>& lanes, const std::vector<std::vector<uint32_t>>& toks,
                                       const std::vector<std::pair<id<MTLBuffer>, uint32_t>>& src,
                                       const std::vector<id<MTLBuffer>>& seed, double* gpu_ms) {
  if (lanes.empty() || lanes.size() > size_t(NSLOT)) mtl::die("mtp_lanes: 1..NSLOT lanes");
  uint32_t M = 0;
  jl.clear();
  for (size_t k = 0; k < lanes.size(); k++) {
    Model* m = lanes[k];
    const uint32_t Ml = uint32_t(toks[k].size());
    if (Ml == 0 || m->cap != cap || m->mtp_pos + Ml > m->cap) mtl::die("mtp_lanes: empty lane, capacity mismatch or full");
    jl.push_back({m, M, Ml, m->mtp_pos, int(k)});
    M += Ml;
  }
  if (M > MAXM) mtl::die("mtp_lanes: more than MAXM rows");
  nsel = nsel_mtp;
  for (uint32_t r = 0; r < M; r++) ((uint32_t*)nsel_mtp.contents)[r] = 0xFFFFFFFFu;
  for (auto& ln : jl) {
    memcpy((uint32_t*)ids.contents + ln.row0, toks[size_t(ln.slot)].data(), ln.M * 4);
    set_rope(ln.p0, ln.M, ln.slot);
  }
  rope = rope_slot[0];
  rope_blk = rope_blk_slot[0];
  if (lane_batch) lanes_tables();
  Rec rc;
  begin(rc, nullptr);
  for (auto& ln : jl) copy(rc, src[size_t(ln.slot)].first, src[size_t(ln.slot)].second, hsrc, ln.row0, ln.M, H * HC);
  EmbArgs ea{M, H, 1};
  id<MTLComputeCommandEncoder> e = rc.op("embed");
  [e setComputePipelineState:k_embed];
  [e setBuffer:ids offset:0 atIndex:0];
  [e setBuffer:embed.b offset:embed.off atIndex:1];
  [e setBuffer:erows offset:0 atIndex:2];
  [e setBytes:&ea length:sizeof ea atIndex:3];
  [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
  struct { uint32_t D, K, src_stride, src_row0; } ra{H, H, H, 0}, rh{H * HC, H, H * HC, 0};
  e = rc.op("rms_frag");
  [e setComputePipelineState:k_rms_frag];
  [e setBuffer:erows offset:0 atIndex:0];
  [e setBuffer:mtp_norm_emb.b offset:mtp_norm_emb.off atIndex:1];
  [e setBuffer:xf offset:0 atIndex:2];
  [e setBytes:&ra length:sizeof ra atIndex:3];
  [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
  mm(rc, "mm.mtp.fc", mtp_fc_emb, xf, pe, M);
  e = rc.op("rms_frag");
  [e setComputePipelineState:k_rms_frag];
  [e setBuffer:hsrc offset:0 atIndex:0];
  [e setBuffer:mtp_norm_hid.b offset:mtp_norm_hid.off atIndex:1];
  [e setBuffer:hnf offset:0 atIndex:2];
  [e setBytes:&rh length:sizeof rh atIndex:3];
  [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
  mm(rc, "mm.mtp.fc", mtp_fc_hid, hnf, ph, M * HC);
  struct { uint32_t H, HC; } fa{H, HC};
  e = rc.op("mtp_fuse");
  [e setComputePipelineState:k_mtp_fuse];
  [e setBuffer:pe offset:0 atIndex:0];
  [e setBuffer:ph offset:0 atIndex:1];
  [e setBuffer:h offset:0 atIndex:2];
  [e setBytes:&fa length:sizeof fa atIndex:3];
  [e dispatchThreads:MTLSizeMake(H * HC, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  bool pending = false;
  encode_layer(rc, mtp, M, jl[0].p0, 0, pending);
  // drafts sample like the verify will: each row with its lane's settings at the position it predicts
  row_samp.clear();
  bool any_sampled = false;
  for (auto& ln : jl) any_sampled |= ln.m->temp > 0.0f;
  if (any_sampled)
    for (auto& ln : jl)
      for (uint32_t r = 0; r < ln.M; r++)
        row_samp.push_back({ln.m->temp, ln.m->top_p, ln.m->top_k, uint32_t(ln.m->sample_seed),
                            uint32_t(ln.m->sample_seed >> 32), ln.p0 + r + 2});
  lane_rows = true;
  encode_head(rc, mtp_mix_dn, mtp_mix_up, mtp_mix_norm, M, pending, jl[0].p0 + 1, has_draft_head ? &mtp_head : nullptr);
  lane_rows = false;
  row_samp.clear();
  copy(rc, h, 0, mtp_h, 0, M, H * HC);
  for (auto& ln : jl) copy(rc, mtp_h, ln.row0 + ln.M - 1, seed[size_t(ln.slot)], 0, 1, H * HC);
  const double ms = finish(rc, nullptr);
  if (gpu_ms) *gpu_ms = ms;
  std::vector<uint32_t> out(lanes.size());
  for (auto& ln : jl) {
    out[size_t(ln.slot)] = ((const uint32_t*)amax_i.contents)[ln.row0 + ln.M - 1];
    ln.m->mtp_pos += ln.M;
  }
  jl.clear();
  return out;
}

// --- joint speculation -----------------------------------------------------------------------
// One round of joint speculation over the live lanes, mirroring generate()'s strict MTP chain per lane:
// (1) per lane, the accept call over the pairs its last verify settled (none in a lane's first round: its
// prefill's MTP pass already gave the first draft), then K-1 chained single drafts; (2) one joint verify
// of [x, drafts] for every lane; (3) per lane, accept the matching drafts, commit, roll back the MTP cache
// entries of rejected drafts. The emitted tokens are the verify's own argmax, so they equal plain greedy
// decoding whatever the drafts are; the drafts only decide how many tokens a round yields.
//
void Model::spec_lanes_round(Model& drv, const std::vector<Model*>& lanes, std::vector<uint32_t>& x,
                             std::vector<std::vector<uint32_t>>& acc, std::vector<std::vector<uint32_t>>& out,
                             std::vector<bool>& live, uint32_t n, SpecStats* st,
                             const std::function<uint32_t(uint32_t, const std::vector<uint32_t>&)>* cut) {
  const uint32_t NL = uint32_t(lanes.size());
  std::vector<uint32_t> live_k;
  for (uint32_t k = 0; k < NL; k++)
    if (live[k]) live_k.push_back(k);
  const uint32_t NA = uint32_t(live_k.size());
  // chain depth per lane: 1, 2, 4, 8 (at most MAXD) while the whole chain keeps landing; the verify of
  // every live lane's [x, drafts] must fit one joint step
  std::vector<uint32_t> K(NL, 0);
  for (uint32_t k : live_k) {
    uint32_t kk = std::min<uint32_t>(1u << std::min<uint32_t>(lanes[k]->spec_level, 3), MAXD);
    while (kk > 1 && uint64_t(NA) * (kk + 1) > MAXM) kk >>= 1;
    K[k] = kk;
  }
  if (uint64_t(NA) * 2 > MAXM) mtl::die("spec_lanes_round: too many lanes for one joint verify");
  std::vector<std::vector<uint32_t>> d(NL);
  for (uint32_t k : live_k) d[k].assign(K[k], 0);
  const size_t W = size_t(H) * HC;
  double dms = 0;
  // each draft pass is one batched MTP step over the lanes that draft in it (mtp_lanes)
  static const bool SERIAL = getenv("OPUS_MTP_SERIAL") != nullptr;  // A/B: the lanes' MTP calls one by one
  auto draft_pass = [&](uint32_t j) {  // j = 0: accept calls, else chained single j
    std::vector<Model*> ml;
    std::vector<std::vector<uint32_t>> mt;
    std::vector<std::pair<id<MTLBuffer>, uint32_t>> ms;
    std::vector<id<MTLBuffer>> seed;
    std::vector<uint32_t> ran;
    for (uint32_t k : live_k) {
      if (j >= K[k] || (j == 0 && acc[k].empty())) continue;  // first round: the prefill drafted d[k][0]
      Model& m = *lanes[k];
      ml.push_back(&m);
      mt.push_back(j == 0 ? acc[k] : std::vector<uint32_t>{d[k][j - 1]});
      ms.push_back(j == 0 ? std::pair{drv.h_tgt, m.acc_src0} : std::pair{m.seed_h_lane[k], 0u});
      seed.push_back(m.seed_h_lane[k]);
      ran.push_back(k);
    }
    if (ran.empty()) return;
    if (!SERIAL) {
      double ms_ = 0;
      auto o = drv.mtp_lanes(ml, mt, ms, seed, &ms_);
      dms += ms_;
      for (size_t i = 0; i < ran.size(); i++) d[ran[i]][j] = o[i];
      return;
    }
    Rec rc;
    drv.begin(rc, nullptr);
    for (size_t i = 0; i < ran.size(); i++) {
      Model& m = *ml[i];
      for (uint32_t r = 0; r < MAXM; r++) ((uint32_t*)m.nsel_mtp.contents)[r] = 0xFFFFFFFFu;
      const uint32_t Mk = uint32_t(mt[i].size());
      m.encode_mtp(rc, Mk, mt[i].data(), ms[i].first, ms[i].second, 1, -1);
      m.copy(rc, m.mtp_h, Mk - 1, seed[i], 0, 1, uint32_t(W));
    }
    dms += drv.finish(rc, nullptr);
    for (size_t i = 0; i < ran.size(); i++)
      d[ran[i]][j] = ((const uint32_t*)ml[i]->amax_i.contents)[mt[i].size() - 1];
  };
  for (uint32_t k : live_k)
    if (acc[k].empty()) d[k][0] = lanes[k]->seed;
  for (uint32_t j = 0; j < MAXD; j++) draft_pass(j);
  if (st) { st->draft_ms += dms; st->rounds++; }

  // ---- one joint verify: x then the lane's drafts
  std::vector<Model*> L;
  std::vector<std::vector<uint32_t>> tk;
  std::vector<uint32_t> row0(NL, 0);
  uint32_t M = 0;
  for (uint32_t k : live_k) {
    std::vector<uint32_t> t{x[k]};
    t.insert(t.end(), d[k].begin(), d[k].end());
    row0[k] = M;
    M += uint32_t(t.size());
    L.push_back(lanes[k]);
    tk.push_back(std::move(t));
  }
  double vms = 0;
  drv.forward_lanes(L, tk, 1, nullptr, &vms);
  if (st) { st->verify_ms += vms; st->rows += M; st->drafted += M - NA; }

  // ---- accept, commit, emit
  std::vector<uint32_t> g((const uint32_t*)drv.amax_i.contents, (const uint32_t*)drv.amax_i.contents + M);
  Rec rc_commit;  // every lane's commit in one command buffer
  drv.begin(rc_commit, nullptr);
  for (size_t i = 0; i < live_k.size(); i++) {
    const uint32_t k = live_k[i];
    Model& m = *lanes[k];
    const auto& t = tk[i];
    uint32_t a = 0;
    // Tolerant acceptance, the same rule as generate_spec's (OPUS_TOL, greedy lanes only): a draft
    // that is not this lane's top token is still accepted when the lane's own verify logits score
    // it within TOL of the top. The lane's logits and amax_v are the rows of this round's verify,
    // which has already completed; the commit encoded below does not touch them.
    static const float TOL_L_ENV = getenv("OPUS_TOL") ? float(atof(getenv("OPUS_TOL"))) : 0.0f;
    const float TOL_L = m.tol_off ? 0.0f : TOL_L_ENV;
    const size_t NVL = drv.lm_head.N;
    auto lane_stop = [&](uint32_t t) { return t == m.eos || std::find(m.stop_ids.begin(), m.stop_ids.end(), t) != m.stop_ids.end(); };
    while (a < K[k]) {
      if (t[a + 1] == g[row0[k] + a]) { a++; continue; }
      if (TOL_L > 0.0f) {
        const uint32_t row = row0[k] + a, cand = t[a + 1];
        const float* lg = (const float*)drv.logits.contents + size_t(row) * NVL;
        const bool stop = lane_stop(cand) || lane_stop(g[row]);
        if (tol_ok(lg, NVL, g[row], cand, TOL_L, stop, m.temp, m.top_p, m.top_k)) {
          a++; if (st) st->tol_accepted++; continue;
        }
      }
      break;
    }
    if (cut && *cut) {  // accepted drafts are the verify's own tokens, so stopping early is plain decoding's stop
      std::vector<uint32_t> e(t.begin() + 1, t.begin() + 1 + a);
      e.push_back(g[row0[k] + a]);
      a = std::max<uint32_t>(1, std::min<uint32_t>((*cut)(k, e), a + 1)) - 1;
    }
    const uint32_t nt = g[row0[k] + a];
    m.encode_commit_lane(rc_commit, drv, row0[k], a + 1);  // also appends x and the accepted drafts to history
    m.spec_level = (a == K[k]) ? std::min<uint32_t>(m.spec_level + 1, 3) : 0;
    // MTP cache: the chained calls appended pairs for d[0..K-2]; keep those that were accepted
    const uint32_t appended = K[k] - 1, keep = std::min(a, appended);
    m.mtp_pos -= appended - keep;
    acc[k].assign(t.begin() + 1 + keep, t.begin() + 1 + a);
    acc[k].push_back(nt);
    m.acc_src0 = row0[k] + keep;  // the verify's final hidden rows of those pairs, in the driver's h_tgt
    for (uint32_t r = 1; r <= a + 1 && live[k]; r++) {
      const uint32_t tok = r <= a ? t[r] : nt;
      out[k].push_back(tok);
      if (out[k].size() >= n || tok == m.eos) live[k] = false;
    }
    x[k] = nt;
    if (st) st->accepted += a;
  }
  const double cms = drv.finish(rc_commit, nullptr);
  if (st) st->commit_ms += cms;
}

std::vector<std::vector<uint32_t>> Model::generate_spec_lanes(Model& drv, const std::vector<Model*>& lanes,
                                                             const std::vector<std::vector<uint32_t>>& prompts,
                                                             uint32_t n, SpecStats* st) {
  const uint32_t NL = uint32_t(lanes.size());
  if (!NL || prompts.size() != NL) mtl::die("generate_spec_lanes: lanes/prompts mismatch");
  if (!has_mtp) mtl::die("generate_spec_lanes: pack has no MTP head");
  if (NL > uint32_t(NSLOT) || NL * 2 > MAXM) mtl::die("generate_spec_lanes: at most MAXM/2 lanes");
  std::vector<std::vector<uint32_t>> out(NL), acc(NL);
  std::vector<uint32_t> x(NL, 0);
  std::vector<bool> live(NL, true);
  for (uint32_t k = 0; k < NL; k++) {
    // as generate() on a fresh slot: prefill with the MTP pass, whose last row drafts the first token
    Model& m = *lanes[k];
    m.reset();
    x[k] = m.feed(prompts[k], true, st);
    memcpy(m.seed_h_lane[k].contents, m.seed_h.contents, size_t(H) * HC * 4);
    m.spec_level = 0;
    out[k].push_back(x[k]);
    if (out[k].size() >= n || x[k] == m.eos) live[k] = false;
  }
  for (uint32_t guard = 0; guard < 1000000; guard++) {
    if (std::none_of(live.begin(), live.end(), [](bool b) { return b; })) break;
    spec_lanes_round(drv, lanes, x, acc, out, live, n, st);
  }
  for (auto* m : lanes) m->mtp_synced = false;
  return out;
}

std::vector<uint32_t> Model::generate_spec(const std::vector<uint32_t>& prompt, uint32_t n, SpecStats* st) {
  reset();
  std::vector<uint32_t> out;
  generate(prompt, n, true, [&](uint32_t t) { out.push_back(t); return true; }, st);
  return out;
}

void Model::copy_mtp_to_seed(uint32_t row) {
  memcpy(seed_h.contents, (const float*)mtp_h.contents + size_t(row) * H * HC, size_t(H) * HC * 4);
}

}  // namespace opus
