#pragma once
#include <cmath>
#include "perf_frame.h"

// perf_frame.h 에서:
//   enum ggml_stage { ST_PREFILL = 0, ST_DECODE = 1, ... };
//   enum perf_group { G_KQV = 0, G_OTHER = 1, G_LMHEAD = 2 };
//   constexpr int GGML_STAGE_COUNT, PERF_G_COUNT; 가 정의되어 있다고 가정

// ---------------------------------------------------------------------
// 1. (stage, group)별 Hybrid fc^3+fm^3 × MLP(1 hidden layer, 64 units) 파라미터
//    Python: fit_hybrid_fc3_fm3_mlp() (hidden_layer_sizes=(64,)) 과 1:1 대응
// ---------------------------------------------------------------------
struct HybridFc3Fm3Mlp {
    // Python에서 사용한 feature 개수
    // z_raw = [log(fc_g), log(fm_g), log(fc_g/fm_g), log(lat), log(fc_g^3+fm_g^3)]
    static constexpr int IN_DIM  = 5;
    // MLP hidden layer 크기 (64 units)
    static constexpr int H1      = 64;
    static constexpr int OUT_DIM = 1;

    // 1) analytic base term: E_base = alpha * (fc^3 + fm^3) * latency_scaled
    //    (fc, fm 단위: kHz, latency_scaled 단위: Python에서 쓴 그대로 (지금은 us))
    double alpha;

    // 2) 표준화 파라미터 (각 feature별 mean/std)
    double mean[IN_DIM];
    double std [IN_DIM];

    // 3) MLP weight/bias
    // layer 0: IN_DIM -> H1 (ReLU)
    double W1[IN_DIM][H1];
    double b1[H1];

    // layer 1: H1 -> OUT_DIM(=1) (linear)
    double W2[H1][OUT_DIM];
    double b2[OUT_DIM];
};

// ---------------------------------------------------------------------
// 2. 전체 EnergyModel = (stage, group)별 Hybrid 모델 + enable 플래그
// ---------------------------------------------------------------------
struct EnergyModel {
    HybridFc3Fm3Mlp model[GGML_STAGE_COUNT][PERF_G_COUNT];
};

// selector.cpp 에서 전역으로 정의
extern EnergyModel g_em;

// ---------------------------------------------------------------------
// 3. 내부 헬퍼: MLP forward (log-residual)
//    Python _mlp_fc3fm3_log_residual() 과 동일 동작
// ---------------------------------------------------------------------
inline double hybrid_log_residual(
    const HybridFc3Fm3Mlp & m,
    double                  cpu_khz,
    double                  mem_khz,
    double                  latency_scaled 
) {
    using std::log;

    double fc_g = cpu_khz / 1e6;
    double fm_g = mem_khz / 1e6;
    if (fc_g <= 0.0) fc_g = 1e-9;
    if (fm_g <= 0.0) fm_g = 1e-9;

    double ratio    = fc_g / fm_g;
    double pow_term = fc_g*fc_g*fc_g + fm_g*fm_g*fm_g;
    // printf("pow_term %.6f \n", pow_term);
    // z_raw = [log(fc_g), log(fm_g), log(fc_g/fm_g), log(lat), log(fc_g^3+fm_g^3)]
    double z_raw[HybridFc3Fm3Mlp::IN_DIM];
    z_raw[0] = log(fc_g);
    z_raw[1] = log(fm_g);
    z_raw[2] = log(ratio + 1e-9);
    z_raw[3] = log(latency_scaled);
    z_raw[4] = log(pow_term + 1e-9);

    // 표준화
    double z[HybridFc3Fm3Mlp::IN_DIM];
    for (int i = 0; i < HybridFc3Fm3Mlp::IN_DIM; ++i) {
        double s = m.std[i];
        if (s == 0.0) s = 1e-9;
        z[i] = (z_raw[i] - m.mean[i]) / s;
    }

    // layer 0: IN_DIM -> H1 (ReLU)
    double h1[HybridFc3Fm3Mlp::H1];
    for (int j = 0; j < HybridFc3Fm3Mlp::H1; ++j) {
        double acc = m.b1[j];
        for (int i = 0; i < HybridFc3Fm3Mlp::IN_DIM; ++i) {
            acc += m.W1[i][j] * z[i];
        }
        if (acc < 0.0) acc = 0.0;  // ReLU
        h1[j] = acc;
    }

    // layer 1: H1 -> OUT_DIM(=1), linear
    double out = m.b2[0];
    for (int i = 0; i < HybridFc3Fm3Mlp::H1; ++i) {
        out += m.W2[i][0] * h1[i];
    }
    // printf("mlp_out %6f", out);

    return out;  // log-residual
}

// ---------------------------------------------------------------------
// 4. 최종 에너지 예측 함수 (시그니처 유지)
//    E = alpha_(s,g)*(fc^3+fm^3)*lat * exp( MLP(z) )
// ---------------------------------------------------------------------
inline double predict_energy_j(
    const EnergyModel & em,
    ggml_stage          stage,
    perf_group          group,
    int                 cpu_khz,
    int                 mem_khz,
    double              latency_scaled
) {
    using std::log;
    using std::exp;

    const int s = static_cast<int>(stage);
    const int g = static_cast<int>(group);

    if (s < 0 || s >= GGML_STAGE_COUNT ||
        g < 0 || g >= PERF_G_COUNT) {
        printf("[EMDBG] invalid index s=%d g=%d\n", s, g);
        return 1e9;
    }

    const HybridFc3Fm3Mlp & m = em.model[s][g];

    const double cpu = static_cast<double>(cpu_khz);
    // printf("cpu_freq %6f", cpu);
    const double mem = static_cast<double>(mem_khz);
    // printf("mem_freq %6f", mem);
    const double lat = static_cast<double>(latency_scaled);


    double base = m.alpha * (cpu*cpu*cpu + mem*mem*mem) * lat;
    if (base <= 0.0) base = 1e-12;

    double log_base = log(base);
    // printf("temp %.6f \n", base);
    double log_res  = hybrid_log_residual(m, cpu, mem, lat);
    // printf("temp_log_res %.6f \n", log_res);
    double logE     = log_base + log_res;
    double E        = exp(logE);

    if (!std::isfinite(E) || E <= 0.0) {
        printf("[EMDBG] nan/neg: s=%d g=%d logE=%.6f base=%.6e log_res=%.6f -> E=1e9\n",
               s, g, logE, base, log_res);
        return 1e9;
    }

    return E;
}