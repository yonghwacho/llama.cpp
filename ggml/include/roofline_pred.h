// roofline_pred.h
#pragma once
#include "perf_frame.h"
#include "freq_ai_policy.h"   // ★ RooflineCaps 여기서 가져옴
#include <algorithm>

inline double predict_latency_ms(const PerfFrame& f,
                                 int cpu_khz, int mem_khz,
                                 const RooflineCaps& caps,
                                 int f_ref_core_khz = 3000000,
                                 int f_ref_mem_khz  = 2020000) {
  // 간단 선형 스케일 (보정 상수는 보드별 교정 필요)
  const double flop_rate = caps.peak_flops * (cpu_khz / (double)f_ref_core_khz);
  const double bw_rate   = caps.peak_bw    * (mem_khz / (double)f_ref_mem_khz);
  const double t_comp = (f.flops / std::max(1.0, flop_rate)) * 1e3; // s→ms
  const double t_mem  = (f.bytes / std::max(1.0, bw_rate))   * 1e3;
  return std::max(t_comp, t_mem);
}