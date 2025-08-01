#pragma once
#include <stdatomic.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_DVFS_MAX_OP 128
extern _Atomic int g_op_freq_table[GGML_DVFS_MAX_OP];
void ggml_dvfs_apply_if_needed(int op_id);
void ggml_dvfs_set(int op_id, int khz);   // 선택 : API 형태로 공개
int  ggml_dvfs_get(int op_id);

#ifdef __cplusplus
}
#endif
