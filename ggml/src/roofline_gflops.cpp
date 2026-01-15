// ggml/src/roofline_gflops.cpp
#include "roofline_gflops.h"
#include "roofline_select.h"
#include "ggml-mckp-freq.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>
#include <string>
#include <cstdio>

// ----------------------------
// small helpers
// ----------------------------
static inline int quantize_nearest(const std::vector<int> & freqs, double target) {
    if (freqs.empty()) return -1;
    int best = freqs[0];
    double bestd = std::abs((double)best - target);
    for (int f : freqs) {
        const double d = std::abs((double)f - target);
        if (d < bestd) { bestd = d; best = f; }
    }
    return best;
}

static inline void dedup_pairs(std::vector<std::pair<int,int>> & pairs) {
    std::sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
}

static inline const char * sweep_axis_str(RooflineSweepAxis ax) {
    switch (ax) {
        case RooflineSweepAxis::CPU_SWEEP: return "CPU_SWEEP";
        case RooflineSweepAxis::MEM_SWEEP: return "MEM_SWEEP";
        case RooflineSweepAxis::INVALID:   return "INVALID";
        default:                           return "UNKNOWN";
    }
}

// log once helpers (to avoid spam)
static bool g_warned_invalid_axis[GGML_STAGE_COUNT][PERF_G_COUNT]  = {};
static bool g_warned_invalid_k[GGML_STAGE_COUNT][PERF_G_COUNT]     = {};
static bool g_warned_invalid_frame[GGML_STAGE_COUNT][PERF_G_COUNT] = {};
static bool g_warned_invalid_base[GGML_STAGE_COUNT][PERF_G_COUNT]  = {};

// ----------------------------
// Default params table
// ----------------------------
static RooflineGflopsParams g_params[GGML_STAGE_COUNT][PERF_G_COUNT];
static bool g_inited = false;

static void init_defaults() {
    if (g_inited) return;
    g_inited = true;

    for (int st = 0; st < GGML_STAGE_COUNT; ++st) {
        for (int pg = 0; pg < PERF_G_COUNT; ++pg) {
            RooflineGflopsParams p{};
            p.k_c1_over_c2        = 1.0; // placeholder
            p.sweep_axis          = RooflineSweepAxis::INVALID; // safe default
            p.max_candidates      = 64;

            // baseline fields (must be filled by init_roofline_params())
            p.baseline_lat_us     = 0.0;
            p.baseline_flops      = 0.0;
            p.baseline_cpu_ridge  = 0;
            p.baseline_mif_ridge  = 0;

            g_params[st][pg] = p;
        }
    }
}

void roofline_set_params(ggml_stage st, perf_group pg, RooflineGflopsParams p) {
    init_defaults();
    if ((int)st < 0 || (int)st >= GGML_STAGE_COUNT) return;
    if ((int)pg < 0 || (int)pg >= PERF_G_COUNT) return;
    g_params[(int)st][(int)pg] = p;
}

RooflineGflopsParams roofline_get_params(ggml_stage st, perf_group pg) {
    init_defaults();
    if ((int)st < 0 || (int)st >= GGML_STAGE_COUNT) return g_params[0][0];
    if ((int)pg < 0 || (int)pg >= PERF_G_COUNT) return g_params[0][0];
    return g_params[(int)st][(int)pg];
}

// -----------------------------------------------------------------------------
// ridge 후보 생성 + latency/energy 채움 (Python ridge-scaling과 1:1 동일)
//
// Python:
//   cpu_scale = target_cpu / baseline_cpu_ridge
//   mif_scale = target_mif / baseline_mif_ridge
//   throughput_scale = min(cpu_scale, mif_scale)
//   lat_us = baseline_lat_us * (target_flops / baseline_flops) / throughput_scale
// -----------------------------------------------------------------------------
std::vector<ChoiceC> roofline_build_ridge_choices(
    const PerfFrame & frame,
    const FreqCandidates & cand,
    energy_predict_cb_t energy_cb,
    void * energy_user
) {
    init_defaults();

    std::vector<ChoiceC> out;

    const ggml_stage st = frame.stage;
    const perf_group pg = frame.group;
    const RooflineGflopsParams p = roofline_get_params(st, pg);

    // 1) frame sanity
    if (!(frame.flops > 0.0) || !(frame.ai > 0.0)) {
        if (!g_warned_invalid_frame[(int)st][(int)pg]) {
            g_warned_invalid_frame[(int)st][(int)pg] = true;
            std::printf("[roofline] invalid frame (st=%d pg=%d): flops=%.3e ai=%.3e\n",
                        (int)st, (int)pg, frame.flops, frame.ai);
        }
        return out;
    }

    // 2) candidates sanity
    if (cand.cpu.empty() || cand.mem.empty()) return out;

    std::vector<int> cpu = cand.cpu;
    std::vector<int> mem = cand.mem;
    std::sort(cpu.begin(), cpu.end());
    std::sort(mem.begin(), mem.end());

    // 3) k sanity
    const double ai = frame.ai;
    const double k  = p.k_c1_over_c2;

    if (!(k > 0.0) || !std::isfinite(k)) {
        if (!g_warned_invalid_k[(int)st][(int)pg]) {
            g_warned_invalid_k[(int)st][(int)pg] = true;
            std::printf("[roofline] invalid k_c1_over_c2 (st=%d pg=%d): k=%.12f\n",
                        (int)st, (int)pg, k);
        }
        return out;
    }

    // 4) baseline sanity (Python ridge-scaling needs these!)
    const bool base_ok =
        (p.baseline_lat_us > 0.0) &&
        (p.baseline_flops  > 0.0) &&
        (p.baseline_cpu_ridge > 0) &&
        (p.baseline_mif_ridge > 0);

    if (!base_ok) {
        if (!g_warned_invalid_base[(int)st][(int)pg]) {
            g_warned_invalid_base[(int)st][(int)pg] = true;
            std::printf("[roofline] baseline params missing (st=%d pg=%d): "
                        "lat_us=%.3f flops=%.3e cpu_ridge=%d mif_ridge=%d. "
                        "Please set via init_roofline_params().\n",
                        (int)st, (int)pg,
                        p.baseline_lat_us, p.baseline_flops,
                        p.baseline_cpu_ridge, p.baseline_mif_ridge);
        }
        return out;
    }

    // 5) sweep axis selection (must be set offline; fallback warns once)
    RooflineSweepAxis axis = p.sweep_axis;
    if (axis == RooflineSweepAxis::INVALID) {
        if (!g_warned_invalid_axis[(int)st][(int)pg]) {
            g_warned_invalid_axis[(int)st][(int)pg] = true;
            std::printf("[roofline] sweep_axis INVALID (st=%d pg=%d). "
                        "Fallback to MEM_SWEEP. (please set via offline)\n",
                        (int)st, (int)pg);
        }
        axis = RooflineSweepAxis::MEM_SWEEP;
    }

    // 6) build ridge pairs using ONLY chosen axis (Python-compatible policy)
    std::vector<std::pair<int,int>> pairs;

    if (axis == RooflineSweepAxis::MEM_SWEEP) {
        pairs.reserve(mem.size());
        for (int mf : mem) {
            // CPU = k * MEM * AI
            const double cpu_target = k * (double)mf * ai;
            const int cf = quantize_nearest(cpu, cpu_target);
            if (cf > 0) pairs.emplace_back(cf, mf);
        }
    } else { // CPU_SWEEP
        pairs.reserve(cpu.size());
        for (int cf : cpu) {
            // MEM = CPU / (k * AI)
            const double mem_target = (double)cf / (k * ai);
            const int mf = quantize_nearest(mem, mem_target);
            if (mf > 0) pairs.emplace_back(cf, mf);
        }
    }

    dedup_pairs(pairs);

    if (p.max_candidates > 0 && (int)pairs.size() > p.max_candidates) {
        pairs.resize(p.max_candidates);
    }

    out.reserve(pairs.size());

    // 7) Fill latency/energy using Python ridge-scaling exactly
    const double baseline_lat_us = p.baseline_lat_us;
    const double baseline_flops  = p.baseline_flops;
    const int    b_cpu_ridge     = p.baseline_cpu_ridge;
    const int    b_mif_ridge     = p.baseline_mif_ridge;

    for (auto [cf, mf] : pairs) {
        const double cpu_scale = (double)cf / (double)b_cpu_ridge;
        const double mif_scale = (double)mf / (double)b_mif_ridge;

        const double throughput_scale = std::min(cpu_scale, mif_scale);
        if (!(throughput_scale > 0.0) || !std::isfinite(throughput_scale)) {
            continue;
        }

        // lat_us = baseline_lat_us * (target_flops / baseline_flops) / throughput_scale
        const double lat_us =
            baseline_lat_us *
            (frame.flops / baseline_flops) /
            throughput_scale;

        const double lat_ms = lat_us / 1000.0;

        double E = 0.0;
        if (energy_cb) {
            E = energy_cb(st, pg, cf, mf, lat_ms, energy_user);
        }

        ChoiceC ch{};
        ch.c       = cf;
        ch.m       = mf;
        ch.latency = lat_ms;
        ch.energy  = E;
        ch.tag     = (axis == RooflineSweepAxis::CPU_SWEEP) ? "ridge_cpu_sweep" : "ridge_mem_sweep";

        out.push_back(ch);
    }

    return out;
}