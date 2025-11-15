// arithmetic_intensity.h
#pragma once

#include "ggml.h"

// llama 내부 타입은 여기서 forward-declare 만
struct llama_ubatch;
struct llama_model;

// 그래프 구조는 무시하지만, 시그니처는 그대로 둬도 됨(안에서 graph 안 씀)
void ggml_analyze_arithmetic_intensity(
    const ggml_cgraph * graph,
    const llama_ubatch & ubatch,
    const llama_model  & model);

void maybe_probe_ai(
    const ggml_cgraph * graph,
    const llama_ubatch & ubatch,
    const llama_model  & model);

void setup_probe_signal();