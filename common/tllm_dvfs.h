#pragma once

// llama_model forward declaration
struct llama_model;

// Select DVFS/energy/roofline model pack by id (e.g., "qwen2.5-1.5b").
// If pack_id == "auto", this function may infer pack id from model metadata / filename.
void tllm_dvfs_select_model_pack(const char * pack_id,
                                 llama_model * model,
                                 const char * model_path);

// Optional: for debugging/logging
const char * tllm_dvfs_get_selected_pack_id();