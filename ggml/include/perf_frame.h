// perf_frame.h
#pragma once
#include <cstdint>

enum ggml_stage   : uint8_t { ST_PREFILL=0, ST_DECODE=1 };
enum ggml_group   : uint8_t { G_ATTN=0, G_FFN=1, G_NORM=2, G_MISC=3, GGML_G_COUNT=4 };

struct PerfFrame {
  // 식별
  uint32_t token_id = 0xFFFFFFFF;
  uint16_t layer    = 0xFFFF;
  ggml_stage stage  = ST_DECODE;
  ggml_group group  = G_MISC;

  // 작업 양
  double flops = 0.0;   // [FLOP]
  double bytes = 0.0;   // [B]
  double ai    = 0.0;   // [FLOP/B] = flops/bytes

  // 선택/적용에 참고할 현재 상태(옵션)
  uint32_t freq_core_khz = 0;
  uint32_t freq_mem_khz  = 0;

  // 예측 결과(선택기가 채움)
  double t_pred_ms = 0.0; // Latency, Energy array
  double e_pred_j  = 0.0; // SLO

  // 메타 (디버그/로그용)
  uint64_t ts_ns = 0;
  uint32_t flags = 0;     // VALID_AI 등 bitflag로 유효성 표시 가능
  uint32_t version = 1;   // 스키마 버전
};