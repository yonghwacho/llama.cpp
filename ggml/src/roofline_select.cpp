#include "roofline_select.h"
#include <algorithm>
#include <cmath>

static RooflineCaps g_caps{};
static RLPolicyCfg  g_cfg{};
static RLSystem     g_sys{};

static inline int clamp_khz(int x, int lo, int hi) {
  return std::max(lo, std::min(hi, x));
}

void rl_set_system(const RLSystem& s) { g_sys = s; }
void rl_configure(const RooflineCaps& caps, const RLPolicyCfg& cfg) { g_caps = caps; g_cfg = cfg; }

void rl_build_candidates(const PerfFrame& f,
                         const RooflineCaps& caps,
                         const RLPolicyCfg& cfg,
                         FreqCandidates& out) {
  out.cpu.clear(); out.mem.clear();

  const double knee_ai = std::max(1e-9, caps.peak_flops / caps.peak_bw);
  const double ai_eff  = f.ai * (g_sys.mem_scale / g_sys.core_scale);

  // 기준 격자(보드 맞춰 조정 가능)
  const int CPU_MIN = 600000, CPU_M1 = 1400000, CPU_M2 = 2200000, CPU_MAX = 3000000;
  const int MEM_MIN = 575000,  MEM_M1 = 1300000, MEM_M2 = 1700000, MEM_MAX = 2020000;

  double len_scale = 1.0;
  if (f.token_id >= 0 && f.token_id >= cfg.long_len_threshold) len_scale = cfg.long_len_scale;

  const double env_scale = std::clamp(g_sys.therm_scale * g_sys.batt_scale, 0.5, 1.0);

  auto push_cpu = [&](std::initializer_list<int> base){
    for (int khz : base) out.cpu.push_back(clamp_khz((int)std::lround(khz * env_scale * len_scale), CPU_MIN, CPU_MAX));
  };
  auto push_mem = [&](std::initializer_list<int> base){
    for (int khz : base) out.mem.push_back(clamp_khz((int)std::lround(khz * env_scale), MEM_MIN, MEM_MAX));
  };

  if (ai_eff < cfg.low_margin * knee_ai) {                 // 메모리 바운드
    push_cpu({ CPU_MIN, CPU_M1, CPU_M2 });
    push_mem({ MEM_M1, MEM_M2, MEM_MAX });
  } else if (ai_eff < cfg.high_margin * knee_ai) {         // 혼합
    push_cpu({ CPU_M1, CPU_M2, CPU_MAX });
    push_mem({ MEM_M1, MEM_M2, MEM_MAX });
  } else {                                                 // 컴퓨트 바운드
    push_cpu({ CPU_M1, CPU_M2, CPU_MAX });
    push_mem({ MEM_MIN, MEM_M1, MEM_M2 });
  }

  auto uniq_sort = [](std::vector<int>& v){
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
  };
  uniq_sort(out.cpu);
  uniq_sort(out.mem);

  if (out.cpu.empty()) out.cpu.push_back(CPU_M2);
  if (out.mem.empty()) out.mem.push_back(MEM_M1);
}