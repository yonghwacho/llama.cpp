// energy_model.h
#pragma once
#include "perf_frame.h"

struct EnergyModel {
  double a0 = 0.8;     // [W]  정적 오프셋
  double aC = 0.9;     // [W/GHz] 코어
  double aM = 0.5;     // [W/GHz] 메모리
  double aG[GGML_G_COUNT] = {0.2, 0.25, 0.1, 0.15}; // 그룹 바이어스
  double aAI = 0.0;    // AI 의존(원하면 사용)
};

inline double predict_power_w(const EnergyModel& m,
                              ggml_group g,
                              int cpu_khz, int mem_khz,
                              double ai=0.0) {
  const double fc = cpu_khz / 1e6; // GHz
  const double fm = mem_khz / 1e6; // GHz
  return m.a0 + m.aC*fc + m.aM*fm + m.aG[g] + m.aAI*ai;
}