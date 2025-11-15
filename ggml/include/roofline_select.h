#pragma once
#include <vector>

struct FreqCandidates {
    std::vector<int> cpu;   // CPU freq candidates (kHz)
    std::vector<int> mem;   // Mem freq candidates (kHz)
};

// gid 는 GGML_DVFS_GRP_SDPA 또는 GGML_DVFS_GRP_OTHER 로 들어옴
void build_freq_candidates_for_group(int gid, FreqCandidates& out);