// Shared typed environment-variable helpers for DankAttn engines.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <string>

namespace dankattn {

inline int env_i(const char* k, int d) { const char* e = getenv(k); return e ? atoi(e) : d; }
inline uint32_t env_u(const char* k, uint32_t d) { const char* e = getenv(k); return e ? uint32_t(atoi(e)) : d; }
inline int64_t env_i64(const char* k, int64_t d) { const char* e = getenv(k); return e ? atoll(e) : d; }
inline double env_f(const char* k, double d) { const char* e = getenv(k); return e ? atof(e) : d; }
inline bool env_b(const char* k, bool d) { const char* e = getenv(k); return e ? atoi(e) != 0 : d; }
inline bool env_set(const char* k) { return getenv(k) != nullptr; }
inline std::string env_s(const char* k, const std::string& d = "") { const char* e = getenv(k); return e ? std::string(e) : d; }

}  // namespace dankattn
