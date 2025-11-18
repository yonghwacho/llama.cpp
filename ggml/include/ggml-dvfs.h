// ggml-dvfs.h
#pragma once
#include <stdatomic.h>
#include "perf_frame.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------- (A) per-op DVFS 테이블 ----------
#define GGML_DVFS_MAX_OP 128

extern _Atomic int g_op_freq_table[GGML_DVFS_MAX_OP];
extern _Atomic int g_op_memfreq_table[GGML_DVFS_MAX_OP];

// op_id별 목표 주파수 설정/조회
void ggml_dvfs_set(int op_id, int khz);
int  ggml_dvfs_get(int op_id);

void ggml_memfreq_set(int op_id, int khz);
int  ggml_memfreq_get(int op_id);

// ---------- (B) 그룹 기반 DVFS (stage × perf_group) ----------

// stage × perf_group 별 계획
typedef struct GroupDvfsPlan {
    int cpu_khz;
    int mem_khz;
} GroupDvfsPlan;

// 이 행렬도 C링케이지 필요하니까 extern "C" 안에 둬야 함
extern GroupDvfsPlan g_group_dvfs_plan[GGML_STAGE_COUNT][PERF_G_COUNT];

// 현재 stage 변경 (prefill / decode 시작할 때 호출)
void ggml_dvfs_begin_stage(ggml_stage st);

// (stage, group)별 계획 세팅용 (offline planner 결과를 여기다 넣으면 됨)
void ggml_dvfs_set_group_plan(ggml_stage st, perf_group g, int cpu_khz, int mem_khz);

// 노드 단위에서 호출할 함수 (compute_thread에서 부름)
void ggml_dvfs_apply_if_needed(int op, const char *name);

#ifdef __cplusplus
} // extern "C"
#endif