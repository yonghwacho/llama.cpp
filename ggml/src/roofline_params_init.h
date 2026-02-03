// ggml/src/roofline_params_init.h
#pragma once

// selector.cpp에서 init_energy_model() 옆에서 같이 호출할 용도
void init_roofline_params();
void init_qwen_roofline_params();
void init_llama3b_roofline_params();