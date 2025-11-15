// selector.cpp
#include "selector.h"
#include "roofline_pred.h"
#include "energy_model.h"

// ===========================================
// 1) GFLOPS log–log 모델 계수 정의 (stage별, group별)
//    그룹: G_KQV / G_OTHER / G_LMHEAD
// ===========================================

// --- Prefill 계수 ---
// 일단 기존 네 그룹(G_ATTN, G_FFN, ...)에서 쓰던 값 그대로 3개 다 복붙해둠.
// 나중에 KQV / OTHER / LMHEAD별로 따로 피팅해서 바꿔도 됨.
double g_logk1_prefill[PERF_G_COUNT] = {
    /* G_KQV    */ -12.5257,
    /* G_OTHER  */ -5.723660,
    /* G_LMHEAD */ -5.723660,
};

double g_alpha_core_prefill[PERF_G_COUNT] = {
    /* G_KQV    */ 0.962,
    /* G_OTHER  */ 0.019364,
    /* G_LMHEAD */ 0.019364,
};

double g_beta_mem_prefill[PERF_G_COUNT] = {
    /* G_KQV    */ 0.0556,
    /* G_OTHER  */ 0.052005,
    /* G_LMHEAD */ 0.052005,
};

double g_gamma_ai_prefill[PERF_G_COUNT] = {
    /* G_KQV    */ 0.0682,
    /* G_OTHER  */ 0.895831,
    /* G_LMHEAD */ 0.895831,
};

// --- Decode 계수 ---
// 일단 prefill과 동일하게 두고, 나중에 decode 데이터로 다시 피팅해서 바꿔도 됨.
double g_logk1_decode[PERF_G_COUNT] = {
    /* G_KQV    */ -5.723660,
    /* G_OTHER  */ -5.723660,
    /* G_LMHEAD */ -5.723660,
};

double g_alpha_core_decode[PERF_G_COUNT] = {
    /* G_KQV    */ 0.019364,
    /* G_OTHER  */ 0.019364,
    /* G_LMHEAD */ 0.019364,
};

double g_beta_mem_decode[PERF_G_COUNT] = {
    /* G_KQV    */ 0.052005,
    /* G_OTHER  */ 0.052005,
    /* G_LMHEAD */ 0.052005,
};

double g_gamma_ai_decode[PERF_G_COUNT] = {
    /* G_KQV    */ 0.895831,
    /* G_OTHER  */ 0.895831,
    /* G_LMHEAD */ 0.895831,
};

// ===========================================
// 2) 에너지 Claude-style 모델 전역 정의 (stage×group)
//    E[J] = (a[stage,group] * (cpu_khz + mem_khz) + b[stage,group]) * latency_us
// ===========================================

EnergyModel g_em{};

// 파이썬 energy_predictor_fitting.py 에서 나온 계수로 초기화
void init_energy_model() {
    constexpr int PREF = static_cast<int>(ST_PREFILL);
    constexpr int DECO = static_cast<int>(ST_DECODE);

    // === Prefill / KQV (ATTENTION core: QK^V) ===
    g_em.a[PREF][G_KQV]    = 3.5e-11;
    g_em.b[PREF][G_KQV]    = 1.0e-3;

    // === Prefill / OTHER (QKV proj + AttnOut proj + FFN) ===
    g_em.a[PREF][G_OTHER]  = 3.899083404309e-11;
    g_em.b[PREF][G_OTHER]  = 1.117274324846e-03;

    // === Prefill / LMHEAD ===
    // 일단 OTHER(=FFN쪽)과 같은 계수 사용. 필요하면 나중에 별도 피팅
    g_em.a[PREF][G_LMHEAD] = g_em.a[PREF][G_OTHER];
    g_em.b[PREF][G_LMHEAD] = g_em.b[PREF][G_OTHER];

    // === Decode / KQV ===
    g_em.a[DECO][G_KQV]    = 3.5e-11;
    g_em.b[DECO][G_KQV]    = 1.0e-3;

    // === Decode / OTHER ===
    g_em.a[DECO][G_OTHER]  = 3.899083404309e-11;
    g_em.b[DECO][G_OTHER]  = 1.117274324846e-03;

    // === Decode / LMHEAD ===
    g_em.a[DECO][G_LMHEAD] = g_em.a[DECO][G_OTHER];
    g_em.b[DECO][G_LMHEAD] = g_em.b[DECO][G_OTHER];
}

// ===========================================
// 3) PerfFrame + freq 후보 => ChoiceC 리스트
//    (stage에 따라 latency/energy 자동 분기)
// ===========================================

std::vector<ChoiceC> build_choices_for_mckp(
    const PerfFrame &      f,
    const EnergyModel &    em,
    const FreqCandidates & cand)
{
    std::vector<ChoiceC> out;
    out.reserve(cand.cpu.size() * cand.mem.size());

    for (int fc : cand.cpu) {
        for (int fm : cand.mem) {
            ChoiceC c{};

            // 1) latency(ms) 예측: f.stage 에 따라 prefill/decode 계수 선택
            double t_ms = predict_latency_ms(f, fc, fm);

            // 2) energy(J) 예측: f.stage + f.group 기반
            double latency_us = t_ms * 1000.0;
            double e_j = predict_energy_j(
                em,
                f.stage,   // ST_PREFILL / ST_DECODE
                f.group,   // G_KQV / G_OTHER / G_LMHEAD
                fc,
                fm,
                latency_us
            );

            c.latency = t_ms;
            c.energy  = e_j;
            c.c       = fc;
            c.m       = fm;
            c.tag     = nullptr;

            out.push_back(c);
        }
    }

    return out;
}