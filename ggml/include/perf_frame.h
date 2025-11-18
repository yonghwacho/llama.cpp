// perf_frame.h
#pragma once

#include <stdint.h>   // C/C++ 겸용

#ifdef __cplusplus
extern "C" {
#endif

// 프리필 / 디코드 스테이지
typedef enum ggml_stage {
    ST_PREFILL = 0,
    ST_DECODE  = 1,
    GGML_STAGE_COUNT
} ggml_stage;

// DVFS/에너지 모델에서 쓸 *논리* 그룹
//  - G_KQV    : KQV core (QK^T + AV)
//  - G_OTHER  : QKV proj + FFN + 기타
//  - G_LMHEAD : LM head
typedef enum perf_group {
    G_KQV    = 0,
    G_OTHER  = 1,
    G_LMHEAD = 2,
    PERF_G_COUNT
} perf_group;

// DVFS / 에너지 모델용 프레임
typedef struct PerfFrame {
    // 식별
    uint32_t   token_id;   // 0xFFFFFFFF == invalid
    uint16_t   layer;      // 0xFFFF      == invalid
    ggml_stage stage;      // ST_PREFILL / ST_DECODE
    perf_group group;      // G_KQV / G_OTHER / G_LMHEAD

    // 작업 양
    double flops;   // [FLOP]
    double bytes;   // [B]
    double ai;      // [FLOP/B] = flops / bytes

    // 선택/적용에 참고할 현재 상태(옵션)
    uint32_t freq_core_khz; // CPU freq
    uint32_t freq_mem_khz;  // MEM freq

    // 예측 결과(선택기가 채움)
    double t_pred_ms; // Latency [ms]
    double e_pred_j;  // Energy  [J]

    // 메타 (디버그/로그용)
    uint64_t ts_ns;   // timestamp [ns]
    uint32_t flags;   // VALID_AI 등 bitflag
    uint32_t version; // 스키마 버전
} PerfFrame;

#ifdef __cplusplus
}
#endif