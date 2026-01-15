// ggml/src/roofline_gflops.h
#pragma once
#include <cstdint>
#include <vector>
#include <string>

#include "perf_frame.h"     // ggml_stage, perf_group, PerfFrame
#include "roofline_select.h"  // FreqCandidates
#include "ggml-mckp-freq.h"   // ChoiceC

enum class RooflineSweepAxis : uint8_t {
    INVALID = 0,
    CPU_SWEEP,
    MEM_SWEEP,
};

struct RooflineGflopsParams {
    // ridge 상수: CPU = k * MEM * AI (후보 생성용)
    double k_c1_over_c2 = 1.0;

    // offline에서 어떤 축만 sweep할지
    RooflineSweepAxis sweep_axis = RooflineSweepAxis::INVALID;

    // candidate cap
    int max_candidates = 64;

    // Python과 동일한 latency scaling을 위한 baseline(실측 기반)
    double baseline_lat_us  = 0.0;   // measured @ (cpu0,maxmif) 또는 (maxcpu,mif0) 등
    double baseline_flops   = 0.0;   // measured flops per run (baseline)
    int    baseline_cpu_ridge = 0;   // cpu_ridge baseline (python: baseline_cpu_ridge)
    int    baseline_mif_ridge = 0;   // mif_ridge baseline (python: baseline_mif_ridge)
};


void roofline_set_params(ggml_stage st, perf_group pg, RooflineGflopsParams p);
RooflineGflopsParams roofline_get_params(ggml_stage st, perf_group pg);


// energy callback: return energy (J)
using energy_predict_cb_t =
    double (*)(ggml_stage st,
               perf_group pg,
               int cpu_khz,
               int mem_khz,
               double lat_ms,
               void * user);


// ridge 후보 생성 + latency/energy 채움 (Python과 동일한 latency scaling)
std::vector<ChoiceC> roofline_build_ridge_choices(
    const PerfFrame & frame,
    const FreqCandidates & cand,
    energy_predict_cb_t energy_cb,
    void * energy_user
);