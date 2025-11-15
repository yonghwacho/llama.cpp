#pragma once
#include "perf_frame.h"

// 스테이지(프리필/디코드) × 그룹(KQV / OTHER / LMHEAD) 별
// Claude-style 에너지 모델 계수:
//   E[J] = (a[s,g] * (cpu_khz + mem_khz) + b[s,g]) * latency_us
struct EnergyModel {
    double a[GGML_STAGE_COUNT][PERF_G_COUNT] = {};
    double b[GGML_STAGE_COUNT][PERF_G_COUNT] = {};
};

extern EnergyModel g_em;

inline double predict_energy_j(
    const EnergyModel & em,
    ggml_stage          stage,
    perf_group          group,
    int                 cpu_khz,
    int                 mem_khz,
    double              latency_us)
{
    const int s = static_cast<int>(stage);
    const int g = static_cast<int>(group);

    const double coef_a = em.a[s][g];
    const double coef_b = em.b[s][g];

    return (coef_a * (cpu_khz + mem_khz) + coef_b) * latency_us;
}