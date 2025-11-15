// selector.h
#pragma once

#include <vector>

#include "perf_frame.h"       // ggml_group, PerfFrame
#include "roofline_select.h"  // FreqCandidates { std::vector<int> cpu, mem }
#include "energy_model.h"     // EnergyModel, predict_energy_j()
#include "roofline_pred.h"    // predict_latency_ms()
#include "ggml-mckp-freq.h"   // ChoiceC

// 전역 에너지 모델 (정의는 selector.cpp 쪽)
extern EnergyModel g_em;

// 각 ggml_group(G_ATTN, G_FFN, G_NORM, G_MISC)에 대해
// g_em.a[g], g_em.b[g] 계수를 채워주는 초기화 함수
void init_energy_model();

// PerfFrame + freq 후보들 → MCKP용 ChoiceC 리스트 생성
std::vector<ChoiceC> build_choices_for_mckp(
    const PerfFrame &      f,
    const EnergyModel &    em,
    const FreqCandidates & cand);