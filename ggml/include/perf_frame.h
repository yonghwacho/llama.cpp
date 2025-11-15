// perf_frame.h
#pragma once
#include <cstdint>

// 프리필 / 디코드 스테이지
enum ggml_stage : uint8_t {
    ST_PREFILL = 0,
    ST_DECODE  = 1,
    GGML_STAGE_COUNT
};

// DVFS/에너지 모델에서 쓸 *논리* 그룹
//  - G_KQV    : QK^V core (QK^T + AV)
//  - G_OTHER  : QKV proj + Attention out proj + FFN
//  - G_LMHEAD : LM head
enum perf_group : uint8_t {
    G_KQV    = 0,
    G_OTHER  = 1,
    G_LMHEAD = 2,
    PERF_G_COUNT
};

struct PerfFrame {
  // 식별
  uint32_t   token_id = 0xFFFFFFFF;
  uint16_t   layer    = 0xFFFF;
  ggml_stage stage    = ST_DECODE;
  perf_group group    = G_OTHER;

  // 작업 양
  double flops = 0.0;   // [FLOP]
  double bytes = 0.0;   // [B]
  double ai    = 0.0;   // [FLOP/B] = flops/bytes

  // 선택/적용에 참고할 현재 상태(옵션)
  uint32_t freq_core_khz = 0;
  uint32_t freq_mem_khz  = 0;

  // 예측 결과(선택기가 채움)
  double t_pred_ms = 0.0; // Latency
  double e_pred_j  = 0.0; // Energy

  // 메타 (디버그/로그용)
  uint64_t ts_ns = 0;
  uint32_t flags = 0;     // VALID_AI 등 bitflag로 유효성 표시 가능
  uint32_t version = 1;   // 스키마 버전
};