#pragma once
#include "perf_frame.h"
#include <cmath>
#include <algorithm>

// prefill / decode GFLOPS log–log 모델 계수
//   log(GFLOPS_g) = logk1[g]
//                 + alpha_core[g] * log(core_khz)
//                 + beta_mem[g]  * log(mem_khz)
//                 + gamma_ai[g]  * log(ai)
extern double g_logk1_prefill      [PERF_G_COUNT];
extern double g_alpha_core_prefill [PERF_G_COUNT];
extern double g_beta_mem_prefill   [PERF_G_COUNT];
extern double g_gamma_ai_prefill   [PERF_G_COUNT];

extern double g_logk1_decode       [PERF_G_COUNT];
extern double g_alpha_core_decode  [PERF_G_COUNT];
extern double g_beta_mem_decode    [PERF_G_COUNT];
extern double g_gamma_ai_decode    [PERF_G_COUNT];

// PerfFrame + (cpu,mem) → latency [ms]
inline double predict_latency_ms(
    const PerfFrame & f,
    int cpu_khz,
    int mem_khz
) {
    const int gi = static_cast<int>(f.group);

    const double core = static_cast<double>(cpu_khz);
    const double mem  = static_cast<double>(mem_khz);
    const double ai   = std::max(f.ai, 1e-9);
    const double eps  = 1e-9;

    const double log_core = std::log(core + eps);
    const double log_mem  = std::log(mem  + eps);
    const double log_ai   = std::log(ai   + eps);

    const bool is_prefill = (f.stage == ST_PREFILL);

    const double *logk1   = is_prefill ? g_logk1_prefill      : g_logk1_decode;
    const double *a_core  = is_prefill ? g_alpha_core_prefill : g_alpha_core_decode;
    const double *b_mem   = is_prefill ? g_beta_mem_prefill   : g_beta_mem_decode;
    const double *g_ai    = is_prefill ? g_gamma_ai_prefill   : g_gamma_ai_decode;

    const double log_gflops =
        logk1[gi]
      + a_core[gi] * log_core
      + b_mem[gi]  * log_mem
      + g_ai[gi]   * log_ai;

    const double gflops = std::exp(log_gflops);
    if (gflops <= 0.0) {
        // 말도 안 되는 값이면 그냥 엄청 큰 latency로 처리
        return 1e9;
    }

    const double flops = f.flops;          // [FLOP]
    const double t_s   = flops / (gflops * 1e9); // s
    return t_s * 1e3;                      // ms
}