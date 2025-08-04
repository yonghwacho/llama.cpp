#pragma once

#include "ggml.h"
#include "ggml-common.h"
#include <atomic>

// 요청 플래그 선언
extern std::atomic<bool> probe_requested;

// 프로브 요청 API
inline void request_probe();

// 그래프 분석 호출 함수
void ggml_analyze_arithmetic_intensity(const ggml_cgraph * graph);

// maybe_probe_api: 외부 요청이 있을 때만 AI 계산 실행
void maybe_probe_ai(const ggml_cgraph * graph);