// Single source of truth for all DankAttn (Hy4) runtime HY4_* environment knobs and defaults.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <string>

namespace hy4 {

inline int env_i(const char* k, int d) { const char* e = getenv(k); return e ? atoi(e) : d; }
inline uint32_t env_u(const char* k, uint32_t d) { const char* e = getenv(k); return e ? uint32_t(atoi(e)) : d; }
inline int64_t env_i64(const char* k, int64_t d) { const char* e = getenv(k); return e ? atoll(e) : d; }
inline double env_f(const char* k, double d) { const char* e = getenv(k); return e ? atof(e) : d; }
inline bool env_b(const char* k, bool d) { const char* e = getenv(k); return e ? atoi(e) != 0 : d; }
inline bool env_set(const char* k) { return getenv(k) != nullptr; }
inline std::string env_s(const char* k, const std::string& d = "") { const char* e = getenv(k); return e ? std::string(e) : d; }

struct Config {
  // Prefill & KV layout
  uint32_t mp = env_u("HY4_MP", 1024);
  uint32_t pf_dec_chunk = env_u("HY4_PF_DEC_CHUNK", 256);  // prefill chunk cap when decode lanes are active
  bool kv8 = env_b("HY4_KV8", true);
  bool dense_attn = env_set("HY4_DENSE");
  uint32_t gch = env_u("HY4_GCH", 4);
  uint32_t commit_layers = env_u("HY4_COMMIT_LAYERS", 6);

  // Attention splits, concurrency & matmul split overrides
  uint32_t as_swa = env_u("HY4_AS_SWA", 8);
  uint32_t as_dsa = env_u("HY4_AS_DSA", 16);
  uint32_t as_draft = env_u("HY4_AS_DRAFT", 8);
  bool fuse_gate = env_b("HY4_FUSE_GATE", true);
  bool mm1_draft = env_b("HY4_MM1_DRAFT", true);
  bool mm1 = env_b("HY4_MM1", false);
  bool conc = env_b("HY4_CONC", true);
  std::string splits = env_s("HY4_SPLITS");
  int split_tab = env_i("HY4_SPLIT_TAB", -1);  // -1 = auto (on for Ultra/Max, off for Base/Pro); 0/1 overrides
  bool split_dump = env_set("HY4_SPLIT_DUMP");

  // DFly2 drafter & speculation
  bool no_dfly = env_set("HY4_NO_DFLY");
  uint32_t dblock = env_u("HY4_DBLOCK", 0);  // 0 = max(manifest.block, 8)
  float draft_cut = float(env_f("HY4_DRAFT_CUT", 0.2));
  bool draft_adapt = env_b("HY4_DRAFT_ADAPT", true);   // fan-out-aware expected-tok/ms draft length selector
  double draft_pscale = env_f("HY4_DRAFT_PSCALE", 1.0);
  bool dhead_w8 = env_i("HY4_DHEAD", 4) == 8;
  bool lookup = env_b("HY4_LOOKUP", true);
  double lk_pmin = env_f("HY4_LK_PMIN", 0.1);
  uint32_t lk_skip = env_u("HY4_LK_SKIP", 8);
  double lk_skip_rate = env_f("HY4_LK_SKIP_RATE", 0.8);
  uint32_t lk_emax = env_u("HY4_LK_EMAX", 15);

  // Multi-lane serving & joint speculation
  uint32_t side_lanes = env_u("HY4_SIDE_LANES", 3);
  uint32_t side_cap = env_u("HY4_SIDE_CAP", 32768);
  bool js = env_b("HY4_JS", true);
  uint32_t js_rows = env_u("HY4_JS_ROWS", 16);
  uint32_t js_lanes = env_u("HY4_JS_LANES", 16);

  // Sampling
  float min_p = float(env_f("HY4_MIN_P", 0.02));

  // Engram I/O & residency
  bool eng_warm = env_b("HY4_ENG_WARM", true);
  bool eng_sync = env_set("HY4_ENG_SYNC");
  bool eng_rdahead = env_set("HY4_ENG_RDAHEAD");
  bool eng_pf_nocache = env_b("HY4_ENG_PF_NOCACHE", true);
  bool eng_dedup = env_b("HY4_ENG_DEDUP", true);
  uint32_t eng_cache_mb = env_u("HY4_ENG_CACHE_MB", 128);
  uint32_t io_threads = env_u("HY4_IO_THREADS", 96);
  uint32_t io_pool = env_u("HY4_IO_POOL", 128);
  bool time_rows = env_set("HY4_TIME_ROWS");
  int64_t residency_ms = env_i64("HY4_RESIDENCY_MS", 1000);

  // SSD prefix cache
  std::string pcache_dir = env_s("HY4_PCACHE");
  bool pcache_set = env_set("HY4_PCACHE");
  double pcache_gb = env_f("HY4_PCACHE_GB", 40.0);
  double pcache_free_gb = env_f("HY4_PCACHE_FREE_GB", 15.0);
};

inline const Config& cfg() {
  static const Config c;
  return c;
}

}  // namespace hy4
