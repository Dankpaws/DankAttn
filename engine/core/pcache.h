// Shared SSD prefix-cache filesystem, LRU eviction, and sub-prefix pruning helpers.
#pragma once
#include "proto.h"
#include <algorithm>
#include <cstdint>
#include <dirent.h>
#include <string>
#include <sys/mount.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace dankattn::pcache {

inline size_t free_disk_bytes(const std::string& dir) {
  struct statfs f{};
  return statfs(dir.c_str(), &f) == 0 ? size_t(f.f_bavail) * f.f_bsize : 0;
}

// Scan `dir` for files ending in `ext` (unlinking stale `.tmp` files). Calls `load_fn(full_path, entry)`
// which returns true if valid; invalid/mismatched files are unlinked. Returns entries sorted oldest-first.
template <typename Entry, typename LoadFn>
inline bool scan_dir(const std::string& dir, const std::string& ext,
                     std::vector<Entry>& ents, uint64_t& clock, size_t& out_total, LoadFn&& load_fn) {
  mkdir(dir.c_str(), 0700);
  chmod(dir.c_str(), 0700);
  DIR* d = opendir(dir.c_str());
  if (!d) return false;
  std::vector<std::pair<double, Entry>> found;
  while (dirent* de = readdir(d)) {
    const std::string n = de->d_name;
    if (n.size() <= ext.size() || n.substr(n.size() - ext.size()) != ext) {
      if (n.size() > 4 && n.substr(n.size() - 4) == ".tmp") unlink((dir + "/" + n).c_str());
      continue;
    }
    Entry en;
    en.path = dir + "/" + n;
    struct stat st{};
    if (!load_fn(en.path, en) || stat(en.path.c_str(), &st) != 0) {
      unlink(en.path.c_str());
      continue;
    }
    found.push_back({double(st.st_mtimespec.tv_sec) + st.st_mtimespec.tv_nsec * 1e-9, std::move(en)});
  }
  closedir(d);
  std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  out_total = 0;
  for (auto& [t, en] : found) {
    en.used = ++clock;
    out_total += en.bytes;
    ents.push_back(std::move(en));
  }
  return true;
}

template <typename Entry>
inline size_t total_bytes(const std::vector<Entry>& ents) {
  size_t t = 0;
  for (const auto& e : ents) t += e.bytes;
  return t;
}

// Evict least-recently-used entries (excluding `protect`) until `need` fits within `budget` and `reserve`.
template <typename Entry>
inline bool make_room(std::vector<Entry>& ents, const std::string& dir,
                      size_t need, size_t budget, size_t reserve, const std::string& protect) {
  auto drop_oldest = [&]() -> bool {
    long o = -1;
    for (size_t i = 0; i < ents.size(); i++)
      if (ents[i].path != protect && (o < 0 || ents[i].used < ents[size_t(o)].used)) o = long(i);
    if (o < 0) return false;
    unlink(ents[size_t(o)].path.c_str());
    ents.erase(ents.begin() + o);
    return true;
  };
  while ((total_bytes(ents) + need > budget || free_disk_bytes(dir) < reserve + need) && drop_oldest()) {}
  return total_bytes(ents) + need <= budget && free_disk_bytes(dir) >= reserve + need;
}

// Unlink and erase any existing entry whose token sequence is a strict prefix of `newest.toks`.
template <typename Entry>
inline void prune_subprefixes(std::vector<Entry>& ents, const Entry& newest) {
  for (size_t i = 0; i < ents.size();) {
    if (ents[i].path != newest.path && ents[i].toks.size() <= newest.toks.size() &&
        proto::common_prefix(ents[i].toks, newest.toks, ents[i].toks.size()) == ents[i].toks.size()) {
      unlink(ents[i].path.c_str());
      ents.erase(ents.begin() + long(i));
    } else {
      i++;
    }
  }
}

}  // namespace dankattn::pcache
