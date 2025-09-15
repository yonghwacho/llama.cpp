#include "freq_ai_policy.h"
#include <algorithm>
#include <cmath>
#include <chrono>

int FreqDecision::clamp_int(int x, int lo, int hi) const {
    return std::max(lo, std::min(hi, x));
}

void FreqDecision::configure(const RooflineCaps& caps, const PolicyConfig& cfg) {
    cfg_ = cfg;
    knee_ai_ = std::max(1e-9, caps.peak_flops / caps.peak_bw);

    rules_.clear();
    const double a0 = 0.0;
    const double a1 = cfg_.low_margin  * knee_ai_;
    const double a2 = knee_ai_;
    const double a3 = cfg_.high_margin * knee_ai_;
    const double a4 = 1e300;

    // 메모리-바운드 강/약, 컴퓨트-바운드 약/강
    rules_.push_back({a0, a1, cfg_.f_min_khz, cfg_.mem_min_khz});
    rules_.push_back({a1, a2, cfg_.f_mid1_khz, cfg_.mem_mid1_khz});
    rules_.push_back({a2, a3, cfg_.f_mid2_khz, cfg_.mem_mid2_khz});
    rules_.push_back({a3, a4, cfg_.f_max_khz, cfg_.mem_max_khz});
}

void FreqDecision::update_system(const SystemSnapshot& sys) { sys_ = sys; }
void FreqDecision::update_query (const QueryContext&   q ) {   q_ = q;  }
void FreqDecision::set_rl_hook(RlHook hook) { rl_ = std::move(hook); }

Decision FreqDecision::pick_from_rules(double ai) const {
    for (const auto& r : rules_) {
        if (ai >= r.ai_lo && ai < r.ai_hi) {
            return { r.cpu_khz, r.mem_khz };
        }
    }
    if (rules_.empty())
        return { cfg_.f_mid2_khz, cfg_.mem_mid1_khz };
    else
        return { rules_.back().cpu_khz, rules_.back().mem_khz };
}

int FreqDecision::apply_hysteresis(int op_id, double ai, int khz) {
    auto& st = per_op_[op_id];
    if (!st.initialized) return khz;

    // 경계 근처에서 이전 결정 유지 (±hysteresis_ratio)
    for (const auto& r : rules_) {
        double lo = r.ai_lo * (1.0 - cfg_.hysteresis_ratio);
        double hi = r.ai_hi * (1.0 + cfg_.hysteresis_ratio);
        if (st.last_ai >= lo && st.last_ai < hi) {
            // 이전 구간과 현재 구간이 겹치면 이전 khz 유지
            if (khz != st.last_cpu_khz && ai >= lo && ai < hi) return st.last_cpu_khz;
        }
    }
    return khz;
}

int FreqDecision::pick_mem_hysteresis(int op_id, double ai, int mem_khz) {
    auto& st = per_op_[op_id];
    if (!st.initialized) return mem_khz;

    // CPU와 같은 경계 완화 영역에서 이전 값 유지
    for (const auto& r : rules_) {
        double lo = r.ai_lo * (1.0 - cfg_.hysteresis_ratio);
        double hi = r.ai_hi * (1.0 + cfg_.hysteresis_ratio);
        if (st.last_ai >= lo && st.last_ai < hi) {
            if (mem_khz != st.last_mem_khz && ai >= lo && ai < hi) return st.last_mem_khz;
        }
    }
    return mem_khz;
}

bool FreqDecision::cooldown_ok(int op_id) const {
    const auto& st = per_op_[op_id];
    if (!st.initialized) return true;
    auto now = std::chrono::steady_clock::now();
    auto us  = std::chrono::duration_cast<std::chrono::microseconds>(now - st.last_apply).count();
    return us > cfg_.cooldown_us;
}

Decision FreqDecision::decide_and_schedule(const OpContext& oc) {
    // 0) 런타임 knee 보정 효과: ai를 스케일해서 고정 rules에 넣는다
    const double ai_eff = oc.ai_ewma * (sys_.mem_scale / sys_.core_scale);

    // 1) 고정 rules로 기본 주파수 선택 (ai_eff 사용!)
    Decision base = pick_from_rules(ai_eff);

    // 2) 길이 기반 에너지 바이어스 (긴 생성이면 살짝 다운)
    if (q_.predicted_len_tokens >= 0 && q_.predicted_len_tokens >= cfg_.long_len_threshold) {
        base.cpu_khz = (int)std::lround(base.cpu_khz * cfg_.long_len_scale);
        base.mem_khz = (int)std::lround(base.mem_khz * cfg_.long_len_scale);
    }

    // 3) 시스템 스케일 (열/배터리)
    double cpu_scale = std::clamp(sys_.therm_scale * sys_.batt_scale, 0.5, 1.0);
    base.cpu_khz = (int) std::lround(base.cpu_khz * cpu_scale);
    base.cpu_khz = clamp_int(base.cpu_khz, cfg_.f_min_khz, cfg_.f_max_khz);

    double mem_scale = std::clamp(sys_.therm_scale * sys_.batt_scale, 0.6, 1.0);
    base.mem_khz = (int) std::lround(base.mem_khz * mem_scale);
    base.mem_khz = clamp_int(base.mem_khz, cfg_.mem_min_khz, cfg_.mem_max_khz);

    // 4) 히스테리시스
    base.cpu_khz = apply_hysteresis    (oc.op_id, ai_eff, base.cpu_khz);
    base.mem_khz = pick_mem_hysteresis (oc.op_id, ai_eff, base.mem_khz);

    // 5) RL 훅 (있으면 최종 오버라이드 허용)
    if (rl_) {
        base.cpu_khz = rl_(oc.op_id, ai_eff, base.cpu_khz);
        base.cpu_khz = clamp_int(base.cpu_khz, cfg_.f_min_khz, cfg_.f_max_khz);
    }

    // 6) 쿨다운: 적용할지 여부는 caller에서 판단할 수도 있지만 여기서 상태 갱신
    auto& st = per_op_[oc.op_id];
    const bool changed =
        !st.initialized ||
        base.cpu_khz != st.last_cpu_khz ||
        base.mem_khz != st.last_mem_khz;

    if (changed || cooldown_ok(oc.op_id)) {
        st.last_cpu_khz = base.cpu_khz;
        st.last_mem_khz = base.mem_khz;
        st.last_ai      = ai_eff;
        st.last_apply   = std::chrono::steady_clock::now();
        st.initialized  = true;
        return base;
    } else {
        // 쿨다운 중이면 이전 값 유지
        return { st.last_cpu_khz, st.last_mem_khz };
    }
}