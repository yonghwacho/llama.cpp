#pragma once
#include <vector>
#include "perf_frame.h"
#include "freq_ai_policy.h"   // RooflineCaps 재사용

// 런타임에 Roofline가 만들어 줄 후보들
struct FreqCandidates {
  std::vector<int> cpu;  // 예: {900000, 1400000, 2200000, 3000000}
  std::vector<int> mem;  // 예: {575000, 1300000, 1700000, 2020000}
};

// 정책 파라미터(마진/스케일 등)
struct RLPolicyCfg {
  double low_margin  = 0.6;
  double high_margin = 1.4;
  double long_len_scale = 0.9;
  int    long_len_threshold = 150;
  double hysteresis_ratio = 0.10;
  int    cooldown_us = 5000;
};

// 시스템 스케일(열/배터리/현재비율)
struct RLSystem {
  double therm_scale = 1.0;  // 0.5~1.0
  double batt_scale  = 1.0;  // 0.5~1.0
  double core_scale  = 1.0;  // 현재/기준 코어주파수 비
  double mem_scale   = 1.0;  // 현재/기준 메모리주파수 비
};

void rl_set_system(const RLSystem& s);
void rl_configure(const RooflineCaps& caps, const RLPolicyCfg& cfg);

// ★ 여기서 후보를 “런타임”에 만든다
void rl_build_candidates(const PerfFrame& f,
                         const RooflineCaps& caps,
                         const RLPolicyCfg& cfg,
                         FreqCandidates& out);