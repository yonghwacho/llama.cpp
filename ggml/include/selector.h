#pragma once
#include <algorithm>
#include "perf_frame.h"
#include "roofline_select.h"  // FreqCandidates
#include "energy_model.h"
#include "freq_ai_policy.h"   // RooflineCaps 재사용
#include "roofline_pred.h"

struct SelectorCfg {
  double lambda_energy = 1.0; // 1.0=에너지 최소, 0.0=지연 최소
};

inline Decision select_with_energy(const PerfFrame& f,
                                   const RooflineCaps& caps,
                                   const EnergyModel& em,
                                   const SelectorCfg& scfg,
                                   const FreqCandidates& cand) {
  Decision best { 0, 0 };
  double best_obj = 1e300;

  for (int fc : cand.cpu) {
    for (int fm : cand.mem) {
      const double t_ms = predict_latency_ms(f, fc, fm, caps);
      const double p_w  = predict_power_w(em, f.group, fc, fm, f.ai);
      const double e_j  = p_w * (t_ms / 1e3);
      const double obj  = scfg.lambda_energy * e_j + (1.0 - scfg.lambda_energy) * t_ms;

      if (obj < best_obj) {
        best_obj = obj;
        best.cpu_khz = fc;
        best.mem_khz = fm;
      }
    }
  }
  return best;
}