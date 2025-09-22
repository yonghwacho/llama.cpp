#pragma once
#include <vector>
#include <array>
#include <cstdint>
#include <chrono>
#include <functional>
// ===== 장치/시스템 캡 요약 (system-data-profiler가 채워줌) =====
struct RooflineCaps {
    double peak_flops;   // FLOPs/s (ex: 1.2e12)
    double peak_bw;      // Bytes/s (ex: 30e9)
};

struct SystemSnapshot {
    double therm_scale = 1.0;   // 0.5~1.0 (스로틀/온도 여유 반영)
    double batt_scale  = 1.0;   // 0.5~1.0 (배터리 절약 모드 반영)
    // ★ 추가: caps 기준 대비 현재 주파수 비율 (없으면 1.0)
    double core_scale  = 1.0;  // e.g., 현재_core_khz / caps_base_core_khz
    double mem_scale   = 1.0;  // e.g., 현재_mem_khz  / caps_base_mem_khz
};

// ===== 쿼리/운영 컨텍스트 (길이 예측 등) =====
struct QueryContext {
    int    predicted_len_tokens = -1;  // 미지정이면 -1
    int    latency_budget_ms    = -1;  // 옵션
};

// ===== 오퍼레이터 컨텍스트 (AI 계산 결과를 담아서 넘김) =====
struct OpContext {
    int     op_id;
    double  ai_ewma;      // op별 EWMA AI
};

// ===== 주파수 규칙(빈 구간별) =====
struct FreqRule {
    double ai_lo;
    double ai_hi;
    int    cpu_khz;       // 목표 CPU freq
    int    mem_khz;       // 선택: 메모리 컨트롤러/DDR용
};

// ===== 최종 결정값 =====
struct Decision {
    int cpu_khz;
    int mem_khz;
};

// ===== 정책 설정 =====
struct PolicyConfig {
    int    f_min_khz   = 600000;
    int    f_max_khz   = 3000000;
    int    f_mid1_khz  = 1400000;   // 메모리-바운드 약
    int    f_mid2_khz  = 2200000;   // 컴퓨트-바운드 약

    int    mem_min_khz =  575000;   // 최저
    int    mem_max_khz = 2020000;   // 최고
    int    mem_mid1_khz = 1300000;   // 중간(임의, 필요 시 조정)
    int    mem_mid2_khz = 1700000;   // 중간(임의, 필요 시 조정)

    double low_margin  = 0.6;       // knee*low_margin
    double high_margin = 1.4;       // knee*high_margin
    int    cooldown_us = 5000;      // 같은 op 주파수 재적용 최소 간격

    // 길이 기반 에너지 바이어스 (긴 생성일수록 보수적으로)
    int    long_len_threshold = 150;   // tokens
    double long_len_scale     = 0.85;  // 길이 길면 f *= scale

    // 히스테리시스: 경계 근처에서 출렁임 방지
    double hysteresis_ratio   = 0.10;  // 경계 ±10%
};

// ===== RL 훅: 필요 시 RL이 최종 khz를 오버라이드 가능 =====
using RlHook = std::function<int(int op_id, double ai, int base_khz)>;

// ===== 의사결정기 =====
class FreqDecision {
public:
    void   configure(const RooflineCaps& caps, const PolicyConfig& cfg);
    void   update_system(const SystemSnapshot& sys);
    void   update_query (const QueryContext&   q);
    void   set_rl_hook  (RlHook hook);
    static int clamp_int(int x, int lo, int hi);

    // op별로 호출: 현재 AI(EWMA) 기반으로 khz 결정. 쿨다운/히스테리시스 반영.
    Decision    decide_and_schedule(const OpContext& oc);

private:
    std::vector<FreqRule> rules_;
    PolicyConfig cfg_;
    double knee_ai_ = 0.0;

    SystemSnapshot sys_;
    QueryContext   q_;

    RlHook rl_;

    struct OpState {
        double last_ai = 0.0;
        int    last_cpu_khz = 0;
        int    last_mem_khz = 0;
        std::chrono::steady_clock::time_point last_apply{};
        bool   initialized = false;
    };
    std::array<OpState, 256> per_op_; // GGML_OP_COUNT 최대치 커버(필요시 늘려도 됨)

    Decision pick_from_rules(double ai) const;
    int      apply_hysteresis   (int op_id, double ai, int cpu_khz); // CPU용
    int      pick_mem_hysteresis(int op_id, double ai, int mem_khz); // MEM용
    bool     cooldown_ok(int op_id) const;
};