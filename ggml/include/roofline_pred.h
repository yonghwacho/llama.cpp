#pragma once
#include "perf_frame.h"
#include <cmath>
#include <algorithm>

// -------------------------------------------------------------
// Model2 기반 Roofline GFLOPS 모델
//   α(AI) = alpha0 * exp(-lam_a * AI)
//   β(AI) = beta0  * (1 + lam_b * AI)
//   GFLOPS = k * core^α(AI) * mem^β(AI) * AI^γ
//
//   latency[s] = flops / (GFLOPS * 1e9)
//   latency[ms] = latency[s] * 1e3
// -------------------------------------------------------------

struct RooflineParams {
    double k;       // scale
    double alpha0;  // α0
    double lam_a;   // λ_a
    double beta0;   // β0
    double lam_b;   // λ_b
    double gamma;   // γ (AI exponent)
};

// (stage, group) 별 파라미터
//   - prefill: g_roof_prefill[g]
//   - decode : g_roof_decode[g]
// 를 selector 쪽 .cpp에서 채워넣으면 됨.
extern RooflineParams g_roof_prefill[PERF_G_COUNT];
extern RooflineParams g_roof_decode [PERF_G_COUNT];

// -------------------------------------------------------------
// PerfFrame + (cpu,mem) → latency [ms]
//   * f.ai     : arithmetic intensity
//   * f.flops  : 연산량 (FLOP)
//   * f.stage  : ST_PREFILL / ST_DECODE
//   * f.group  : perf_group (G_KQV, G_OTHER, G_LMHEAD, ...)
// -------------------------------------------------------------
inline double predict_latency_ms(
    const PerfFrame & f,
    int               cpu_khz,
    int               mem_khz
) {
    const int gi = static_cast<int>(f.group);

    const double core = static_cast<double>(cpu_khz);
    const double mem  = static_cast<double>(mem_khz);

    const double eps  = 1e-9;
    const double ai   = std::max(f.ai, eps);

    if (core <= 0.0 || mem <= 0.0) {
        return 1e9;
    }

    const bool is_prefill = (f.stage == ST_PREFILL);

    const RooflineParams &p =
        is_prefill ? g_roof_prefill[gi] : g_roof_decode[gi];

    // α(AI), β(AI)
    const double alpha = p.alpha0 * std::exp(-p.lam_a * ai);
    const double beta  = p.beta0  * (1.0 + p.lam_b * ai);

    // GFLOPS = k * core^α * mem^β * AI^γ
    //  - core, mem 단위: kHz (Python에서도 같은 단위로 fit했다고 가정)
    double gflops = p.k
                  * std::pow(core, alpha)
                  * std::pow(mem,  beta )
                  * std::pow(ai,   p.gamma);

    if (!(gflops > 0.0) || !std::isfinite(gflops)) {
        // 말도 안 되는 값이면 그냥 엄청 큰 latency로 처리
        return 1e9;
    }

    const double flops = f.flops;  // [FLOP], training 때 사용한 동일 값
    const double t_s   = flops / (gflops * 1e9); // [s]
    const double t_ms  = t_s * 1e3;              // [ms]

    if (!std::isfinite(t_ms) || t_ms <= 0.0) {
        return 1e9;
    }
    return t_ms;
}