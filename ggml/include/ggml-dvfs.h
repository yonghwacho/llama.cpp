#pragma once
#include <stdatomic.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_DVFS_MAX_OP 128
extern _Atomic int g_op_freq_table[GGML_DVFS_MAX_OP];
void ggml_dvfs_set(int op_id, int khz);   // 선택 : API 형태로 공개
int  ggml_dvfs_get(int op_id);

void ggml_memfreq_set(int op_id, int khz);
int  ggml_memfreq_get(int op_id);

/* 필요한 경우 실제 sysfs 적용 (CPU+MEM 모두 내부에서 처리) */
void ggml_dvfs_apply_if_needed(int op_id);
int infer_group_from_node_c(const char *name);

#ifdef __cplusplus
}
#endif

enum {
    GGML_DVFS_GRP_ATTN = 0,
    GGML_DVFS_GRP_FFN  = 1,
    GGML_DVFS_GRP_NORM = 2,
    GGML_DVFS_GRP_MISC = 3,
    GGML_DVFS_GRP_COUNT = 4,
    GGML_DVFS_GRP_SKIP = -1
};