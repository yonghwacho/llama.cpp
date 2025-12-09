// selector.cpp
#include "selector.h"
#include "roofline_pred.h"
#include "energy_model.h"
#include <iostream>

// ===========================================
// 1) GFLOPS log–log 모델 계수 정의 (stage별, group별)
//    그룹: G_KQV / G_OTHER / G_LMHEAD
// ===========================================

// ===========================================
// 1) Roofline Model2 계수 정의 (stage별, group별)
//    α(AI) = alpha0 * exp(-lam_a * AI)
//    β(AI) = beta0  * (1 + lam_b * AI)
//    GFLOPS = k * core^α(AI) * mem^β(AI) * AI^γ
// ===========================================

// ⚠ PERF_G_COUNT 순서가
//    { G_KQV, G_GROUP2(or OTHER), G_LMHEAD } 라고 가정하고 채울게.
//    enum 순서에 맞게만 넣으면 된다.

// ----- Prefill -----
RooflineParams g_roof_prefill[PERF_G_COUNT] = {
    // [0] G_KQV (KQVONLY_prefill)  ← 아직 피팅 안 했으면 대충 0 넣고 disabled 로 두거나,
    //                                나중에 숫자 채워 넣기
    { /*k*/ 3.55E-01, /*alpha0*/ 1.4578, /*lam_a*/ 0.049097,
      /*beta0*/ 0.1057, /*lam_b*/ 0.0, /*gamma*/ 0.1944 },

    // [1] G_GROUP2 (Group2_prefill)  ← 마찬가지로 나중에 값 채우기
    { 1.0, 0.136, 0.006323,
      0.0623, 0.0, 0.2344 },

    // [2] G_LMHEAD (LMHead_prefill)
    { 1.0, 0.0496, 0.00289, 0.0448, 0.0, 0.2486 }
};


// struct RooflineParams {
//     double k;       // scale
//     double alpha0;  // α0
//     double lam_a;   // λ_a
//     double beta0;   // β0
//     double lam_b;   // λ_b
//     double gamma;   // γ (AI exponent)
// };



// ----- Decode -----
RooflineParams g_roof_decode[PERF_G_COUNT] = {
    // [0] G_KQV (KQVONLY_decode)
    //   * 엑셀의 "log(k1)" 컬럼이:
    //      - 만약 Python에서 k를 그대로 print했으면 → 그냥 그대로 k에 넣기
    //      - 진짜 log(k1)를 저장했다면 → std::exp(4.78e-2) 를 k로 넣기
    //   지금 스크립트는 k 자체를 print하고 있으니까 보통은 "k"로 보는 게 맞음.
    { 4.78e-02, 0.1915, 0.310862, 0.1128, .263874, 1.0732 },

    // [1] G_GROUP2 (Group2_decode)
    //   log(k1)=1.0e-5, α=0.32092, β=0.57976,
    //   lam_α=0.1648615, lam_β=0.1754076, γ=0
    { 2.53E-06, 0.248, 2.208322, 0.5964, 0.80837, 3.9999},

    // [2] G_LMHEAD (LMHead_decode)
    { 3.01E-06, 0.1956, 1.419014, 0.5726, 0.753977, 1.4716 }
};
// ===========================================
// 2) 에너지 모델 전역 정의 (stage×group)
//    이제는 Claude-style a,b가 아니라
//    HybridFc3Fm3Mlp 기반 EnergyModel을 사용.
// ===========================================

// energy_model.h 에서 선언된 전역
EnergyModel g_em{};

// Python 스크립트에서 자동 생성한 초기화 함수들.
// 예: decode / LMHEAD용 하이브리드 모델 초기화.
// (파일 이름 예: hybrid_decode_lmhead_init.cpp 에서 정의)
extern void init_energy_model_decode_lmhead();
extern void init_energy_model_decode_KQVONLY();
extern void init_energy_model_decode_group2();
extern void init_energy_model_prefill_KQVONLY();
extern void init_energy_model_prefill_group2();
extern void init_energy_model_prefill_lmhead();

void init_energy_model() {
    // std::cout << "init_energy_model" << "\n";
    // 2) Python에서 fit한 하이브리드 모델들을 등록
    //    지금은 decode / LMHEAD만 있다고 가정.
    //    (hybrid_decode_lmhead_init.cpp 안에서
    //     g_em.model[ST_DECODE][G_LMHEAD] 을 채우고
    //     g_em.enabled[ST_DECODE][G_LMHEAD] = true 로 설정.)
    init_energy_model_decode_KQVONLY();
    init_energy_model_decode_group2();
    init_energy_model_decode_lmhead();
    init_energy_model_prefill_KQVONLY();
    init_energy_model_prefill_group2();
    init_energy_model_prefill_lmhead();
}

// ===========================================
// 3) PerfFrame + freq 후보 => ChoiceC 리스트
//    (stage에 따라 latency/energy 자동 분기)
//    predict_energy_ms() / predict_energy_j()의 인터페이스는 그대로 사용
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
                //    이때 latency는 us 단위로 맞춰서 넘김
                double latency_us = t_ms * 1000.0;
                
                double e_j = predict_energy_j(
                    em,
                    f.stage,   // ST_PREFILL / ST_DECODE
                    f.group,   // G_KQV / G_OTHER / G_LMHEAD
                    fc,
                    fm,
                    latency_us
                );
            // std::cout << "d" << "\n";

            c.latency = t_ms;
            c.energy  = e_j;
            c.c       = fc;
            c.m       = fm;
            c.tag     = nullptr;

            // printf("[MCKP raw] st=%d grp=%d cpu=%d mem=%d  t_ms=%.3f  E=%.6e\n",(int)f.stage, (int)f.group, fc, fm, t_ms, e_j);

            out.push_back(c);
        }
    }

    return out;
}