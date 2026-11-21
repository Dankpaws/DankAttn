// compile every hy4 pipeline (no pack needed): clang++ -std=c++20 -fobjc-arc hy4/kcheck.mm -framework Metal -framework Foundation
#include "../engine/mtl.h"
int main() {
  @autoreleasepool {
    mtl::Ctx C;
    const std::string kd = mtl::tree_root() + "/engine/kernels/";
    const std::string mm2 = mtl::slurp(kd + "mm2.metal"), hk = mtl::slurp(kd + "hy4.metal");
    for (int b : {4, 6, 8})
      for (int g : {64, 128}) {
        C.pipe(mm2, "mm2", {{"WBITS", std::to_string(b)}, {"GS", std::to_string(g)}});
        C.pipe(mm2, "mm2", {{"WBITS", std::to_string(b)}, {"GS", std::to_string(g)}, {"EPI", "3"}});
        C.pipe(mm2, "mm2", {{"WBITS", std::to_string(b)}, {"GS", std::to_string(g)}, {"GROUPED", "1"}});
        C.pipe(mm2, "mm2", {{"WBITS", std::to_string(b)}, {"GS", std::to_string(g)}, {"GROUPED", "1"}, {"EPI", "3"}, {"SWIGLU_LIMIT", "10"}});
        C.pipe(mm2, "mmp", {{"WBITS", std::to_string(b)}, {"GS", std::to_string(g)}, {"MMP_FAST", "1"}, {"MMP_NF", "2"}});
        C.pipe(mm2, "mmpg", {{"WBITS", std::to_string(b)}, {"GS", std::to_string(g)}, {"GROUPED", "1"}, {"MMP_CH", "4"}, {"MMP_FAST", "1"}, {"MMP_NF", "2"}});
        C.pipe(mm2, "mmpg", {{"WBITS", std::to_string(b)}, {"GS", std::to_string(g)}, {"GROUPED", "1"}, {"CONTIG_X", "1"}, {"MMP_CH", "8"}, {"MMP_FAST", "1"}, {"MMP_NF", "2"}});
      }
    for (const char* k : {"h4_embed", "h4_hc_1pass", "h4_hc_part", "h4_hc_fin", "h4_rms", "h4_attn_in", "h4_q_prep", "h4_idx_pool",
                          "h4_idx_topk", "h4_attn_merge", "h4_gate", "h4_route", "h4_plan", "h4_combine",
                          "h4_eng_rows", "h4_eng_mix", "h4_argmax", "h4_plan_lists", "h4_combine_c",
                          "h4_add", "h4_dsel", "h4_add_frag", "h4_d_post_add_ln", "h4_draft_hc_in",
                          "h4_sample", "h4_pick_p", "h4_idx_score_mma", "h4_dgate", "h4_sample_lanes",
                          "h4_draft_eh_rms", "h4_draft_eh_add", "h4_dsel_lanes", "h4_draft_corr", "h4_pick_p_lanes", "h4_dgate_lanes"})
      C.pipe(hk, k);
    for (const char* k : {"h4_attn_in", "h4_idx_pool", "h4_idx_score_mma", "h4_idx_topk"})
      C.pipe(hk, k, {{"LANES", "1"}});
    for (const char* sl : {"0", "1"})
      for (const char* ln : {"0", "1"})
        C.pipe(hk, "h4_attn_mma", {{"ATT_SLIDING", sl}, {"ATT_KV8", sl[0] == '0' ? "1" : "0"}, {"LANES", ln}});
    printf("all pipelines compiled\n");
  }
}
