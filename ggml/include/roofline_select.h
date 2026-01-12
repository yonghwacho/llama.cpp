#pragma once
#include <vector>
#include "perf_frame.h"   // ggml_stage, perf_group

// -----------------------------
// Frequency candidate container
// -----------------------------
struct FreqCandidates {
    std::vector<int> cpu;   // CPU freq candidates (kHz)
    std::vector<int> mem;   // Mem freq candidates (kHz)
};

// gid 는 GGML_DVFS_GRP_SDPA 또는 GGML_DVFS_GRP_OTHER 로 들어옴
void build_freq_candidates_for_group(int gid, FreqCandidates& out);

// ------------------------------------------------------------
// Energy prediction callback (used by roofline + MCKP)
// ------------------------------------------------------------
using energy_predict_cb_t =
    double (*)(ggml_stage st,
               perf_group pg,
               int cpu_khz,
               int mem_khz,
               double latency_ms,
               void * user);

energy_predict_cb_t get_energy_cb();