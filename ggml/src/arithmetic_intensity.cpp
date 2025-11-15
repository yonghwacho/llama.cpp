// arithmetic_intensity.cpp
#include "arithmetic_intensity.h"
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-dvfs.h"

#include "roofline_select.h"   // build_freq_candidates_for_group
#include "perf_frame.h"
#include "selector.h"          // build_choices_for_mckp, g_em (extern)
#include "roofline_pred.h"
#include "energy_model.h"
#include "ggml-mckp-freq.h"

// llama 내부 타입 정의
#include "llama-batch.h"   // struct llama_ubatch
#include "llama-model.h"   // struct llama_model

#include <atomic>
#include <csignal>
#include <cstring>
#include <vector>
#include <array>
#include <cstdio>
#include <cstdint>
#include <algorithm>

// ──────────────────────────────────────────────
// DVFS 그룹 ID
//   - 0: ATTENTION_CORE (QK^T + AV = KQVONLY)
//   - 1: PROJ+FFN       (QKV proj + Wo + FFN)
//   - 2: LMHEAD
// ──────────────────────────────────────────────
#ifndef GGML_DVFS_GRP_ATT_CORE
#define GGML_DVFS_GRP_ATT_CORE   0
#endif

#ifndef GGML_DVFS_GRP_MID
#define GGML_DVFS_GRP_MID        1
#endif

#ifndef GGML_DVFS_GRP_LM
#define GGML_DVFS_GRP_LM         2
#endif

// ──────────────────────────────────────────────
// probe 플래그 + API
// ──────────────────────────────────────────────
static std::atomic<bool> probe_requested{true};

inline void request_probe() {
    probe_requested.store(true);
}

// ──────────────────────────────────────────────
// 그룹별 집계용
//   - Group 0: ATTENTION_CORE (KQVONLY: QK^T + AV)
//   - Group 1: PROJ+FFN       (Q/K/V proj + Wo proj + FFN)
//   - Group 2: LMHEAD         (logits projection)
// PerfFrame.group 은
//   - G_KQV    : Group1
//   - G_OTHER  : Group2
//   - G_LMHEAD : Group3
// ──────────────────────────────────────────────
struct GroupAgg {
    int      gid   = -1;   // DVFS 그룹 ID
    bool     active = false;
    PerfFrame frame{};
};

// 그룹 3개
static GroupAgg g_groups[3] = {
    { GGML_DVFS_GRP_ATT_CORE, false, {} },  // Group1: KQVONLY core
    { GGML_DVFS_GRP_MID,      false, {} },  // Group2: QKV proj + Wo + FFN
    { GGML_DVFS_GRP_LM,       false, {} },  // Group3: LM Head
};

static GroupAgg * find_group_agg(int gid) {
    for (auto & g : g_groups) {
        if (g.gid == gid) return &g;
    }
    return nullptr;
}

// ──────────────────────────────────────────────
// helper: 타입 별 바이트 수
// ──────────────────────────────────────────────
static inline int bpp_ggml_type(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        case GGML_TYPE_Q8_0: return 1;
        case GGML_TYPE_Q4_0: return 1;
        default:             return 4;
    }
}

// PREFILL / DECODE 판별: n_tokens 기준
static ggml_stage infer_stage_from_ubatch(const llama_ubatch & ubatch) {
    return (ubatch.n_tokens > 1) ? ST_PREFILL : ST_DECODE;
}

// ──────────────────────────────────────────────
// Closed-form 비용 모델
//   - Group1: Attention core(KQVONLY) = QK^T + AV
//   - Group2: Q/K/V projection + Wo projection + FFN
//   - Group3: LMHead
//   (tllm_profiler와 동일한 구조 / 거의 동일한 수식)
// ──────────────────────────────────────────────

// 1) Attention core = KQVONLY (QK^T + AV 만)
struct AttnCoreCost {
    double flops;
    double bytes;
};

// tllm_profiler 의 KQVONLY 경량 버전과 동일한 형태
static inline AttnCoreCost estimate_kqvonly_closed_form(
    int E, int H, int Hkv,
    int N,     // tokens per batch
    int L      // KV context length
) {
    const double dE   = (double) E;
    const double dH   = (double) H;
    const double dHkv = (double) Hkv;
    const double dD   = dE / dH;      // head_dim
    const double dN   = (double) N;
    const double dL   = (double) L;

    // FLOPs: QK^T + AV
    //   - QK^T : 2 * H * N * L * D
    //   - AV   : 2 * H * N * L * D
    const double fl_qkt = 2.0 * dH * dN * dL * dD;
    const double fl_av  = 2.0 * dH * dN * dL * dD;
    const double flops_total = fl_qkt + fl_av;

    // Bytes:
    //   - Q activations : [H, D, N]
    //   - K/V KV cache read : (E/H * Hkv) * L (per K,V) * bKV
    const int bA  = bpp_ggml_type(GGML_TYPE_F32);  // activations
    const int bKV = bpp_ggml_type(GGML_TYPE_F16);  // KV cache

    const double bytes_Q = dH * dD * dN * bA;

    const double dim_kv       = (dE / dH) * dHkv;  // (E/H) * Hkv
    const double bytes_read_K = dim_kv * dL * bKV;
    const double bytes_read_V = dim_kv * dL * bKV;

    // Wo weight는 Group2에서만 카운트 (중복 방지)
    const double bytes_total = bytes_Q + bytes_read_K + bytes_read_V;

    AttnCoreCost ac{};
    ac.flops = flops_total;
    ac.bytes = bytes_total;
    return ac;
}

// 2) Q/K/V projection + Wo projection per-layer → Group2
struct ProjCost {
    double flops;
    double bytes;
};

// tllm_profiler 의 GROUP2(QKVPROJ+Wo)와 동일 개념
static inline ProjCost estimate_qkvproj_closed_form(
    int E, int H, int Hkv,
    int N,
    ggml_type wtype
) {
    const double dE   = (double) E;
    const double dH   = (double) H;
    const double dHkv = (double) Hkv;
    const double dD   = dE / dH;      // head_dim
    const double dN   = (double) N;

    // FLOPs
    //   Q : 2 * E * E * N
    //   K : 2 * E * (D * Hkv) * N
    //   V : 2 * E * (D * Hkv) * N
    //   Wo: 2 * E * E * N
    const double fl_q  = 2.0 * dE * dE * dN;
    const double fl_k  = 2.0 * dE * (dD * dHkv) * dN;
    const double fl_v  = fl_k;
    const double fl_wo = 2.0 * dE * dE * dN;

    const double flops_total = fl_q + fl_k + fl_v + fl_wo;

    // Bytes (Weights + Activations, tllm_profiler GROUP2와 구조 맞춤)
    const int bW = bpp_ggml_type(wtype);
    const int bA = bpp_ggml_type(GGML_TYPE_F32);

    const double bytes_wq = dE * dE         * bW;
    const double bytes_wk = dE * (dD*dHkv)  * bW;
    const double bytes_wv = dE * (dD*dHkv)  * bW;
    const double bytes_wo = dE * dE         * bW;

    const double bytes_X = dE * dN * bA;
    const double bytes_Q = dE * dN * bA;
    const double bytes_K = (dD * dHkv) * dN * bA;
    const double bytes_V = (dD * dHkv) * dN * bA;

    const double bytes_total =
        (bytes_wq + bytes_wk + bytes_wv + bytes_wo) +
        (bytes_X + bytes_Q + bytes_K + bytes_V);

    ProjCost pc{};
    pc.flops = flops_total;
    pc.bytes = bytes_total;
    return pc;
}

// 3) FFN per-layer → Group2에 합산
struct FFNCost {
    double flops;
    double bytes;
};

// tllm_profiler 의 FFN 경량 모델과 동일 개념
static inline FFNCost estimate_ffn_closed_form(
    int E, int F, int N,
    ggml_type wtype, ggml_type act_type
) {
    const double dE = (double) E;
    const double dF = (double) F;
    const double dN = (double) N;

    // FLOPs (한 레이어 기준)
    // up    : 2 * F * N * E
    // gate  : 2 * F * N * E
    // silu  : 4 * F * N
    // fused : 1 * F * N
    // down  : 2 * E * N * F
    const double fl_up    = 2.0 * dF * dN * dE;
    const double fl_gate  = 2.0 * dF * dN * dE;
    const double fl_silu  = 4.0 * dF * dN;
    const double fl_fused = 1.0 * dF * dN;
    const double fl_down  = 2.0 * dE * dN * dF;

    const double flops_total = fl_up + fl_gate + fl_silu + fl_fused + fl_down;

    // Bytes
    const int bW = bpp_ggml_type(wtype);
    const int bA = bpp_ggml_type(act_type);

    const double bytes_w_up   = dE * dF * bW;
    const double bytes_w_gate = dE * dF * bW;
    const double bytes_w_down = dF * dE * bW;

    const double bytes_X     = dE * dN * bA;
    const double bytes_up    = dF * dN * bA;
    const double bytes_gate  = dF * dN * bA;
    const double bytes_fused = dF * dN * bA;
    const double bytes_out   = dE * dN * bA;

    const double bytes_total =
        (bytes_w_up + bytes_w_gate + bytes_w_down) +
        (bytes_X + bytes_up + bytes_gate + bytes_fused + bytes_out);

    FFNCost fc{};
    fc.flops = flops_total;
    fc.bytes = bytes_total;
    return fc;
}

// 4) LM Head (한 번만, Group3)
static inline void estimate_lmhead_closed_form(
    int E, int V, int N,
    ggml_type wtype, ggml_type act_type,
    double & flops_out,
    double & bytes_out
) {
    const double dE = (double) E;
    const double dV = (double) V;
    const double dN = (double) N;

    // FLOPs: [V,E] * [E,N]
    const double fl = 2.0 * dE * dV * dN;

    const int bW = bpp_ggml_type(wtype);
    const int bA = bpp_ggml_type(act_type);

    const double bytes_w = dE * dV * bW;
    const double bytes_x = dE * dN * bA;
    const double bytes_y = dV * dN * bA;

    flops_out = fl;
    bytes_out = bytes_w + bytes_x + bytes_y;
}

// ──────────────────────────────────────────────
// 메인 분석 함수
// ──────────────────────────────────────────────
void ggml_analyze_arithmetic_intensity(
    const ggml_cgraph * graph,
    const llama_ubatch & ubatch,
    const llama_model  & model) {

    GGML_UNUSED(graph); // 안 쓰는 파라미터 워닝 제거

    const auto & hp = model.hparams;

    const int n_embd    = (int) hp.n_embd;
    const int n_head    = (int) hp.n_head();
    const int n_head_kv = (int) hp.n_head_kv();
    const int n_ff      = (int) hp.n_ff();
    const int n_layer   = (int) hp.n_layer;

    const int n_vocab   = (int) model.vocab.n_tokens();
    const int n_tokens  = (int) ubatch.n_tokens;

    // KV 길이 추정: decode의 경우 pos 최대값 + 1, prefill이면 대략 n_tokens
    int n_ctx = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        n_ctx = std::max(n_ctx, (int) ubatch.pos[i] + 1);
    }
    if (n_ctx == 0) {
        n_ctx = n_tokens;
    }

    if (n_tokens <= 0 || n_embd <= 0 || n_layer <= 0) {
        printf("[AI] invalid hparams: n_tokens=%d, n_embd=%d, n_layer=%d\n",
               n_tokens, n_embd, n_layer);
        return;
    }

    ggml_stage stage = infer_stage_from_ubatch(ubatch);
    printf("[AI] stage=%s, n_tokens=%d, n_ctx=%d\n",
           stage == ST_PREFILL ? "PREFILL" : "DECODE",
           n_tokens, n_ctx);

    printf("[AI] params: n_layer=%d, n_embd=%d, n_head=%d, n_head_kv=%d, "
           "n_ff=%d, n_vocab=%d, n_tokens=%d, n_ctx=%d\n",
           n_layer, n_embd, n_head, n_head_kv, n_ff, n_vocab, n_tokens, n_ctx);

    // ──────────────────────────────────────────────
    // 그룹 초기화 (+ stage 세팅)
    // ──────────────────────────────────────────────
    for (auto & g : g_groups) {
        g.active = false;
        g.frame  = {};        // zero-init
        g.frame.stage = stage;
    }

    double total_flops = 0.0;
    double total_bytes = 0.0;

    // ──────────────────────────────────────────────
    // 1) Attention core (KQVONLY = QK^T + AV) → Group1
    // ──────────────────────────────────────────────
    AttnCoreCost att_core = estimate_kqvonly_closed_form(
        n_embd,
        n_head,
        n_head_kv,
        n_tokens,
        n_ctx
    );

    GroupAgg * g_att = find_group_agg(GGML_DVFS_GRP_ATT_CORE);
    if (g_att) {
        g_att->active        = true;
        g_att->frame.group   = G_KQV;  // PerfFrame용 논리 그룹
        g_att->frame.flops   = att_core.flops * n_layer;
        g_att->frame.bytes   = att_core.bytes * n_layer;
        g_att->frame.ai      = g_att->frame.bytes > 0.0
                                 ? g_att->frame.flops / g_att->frame.bytes
                                 : 0.0;
        total_flops += g_att->frame.flops;
        total_bytes += g_att->frame.bytes;
    }

    // ──────────────────────────────────────────────
    // 2) Proj(Q/K/V, Wo) + FFN → Group2
    // ──────────────────────────────────────────────
    ProjCost proj_layer = estimate_qkvproj_closed_form(
        n_embd,
        n_head,
        n_head_kv,
        n_tokens,
        GGML_TYPE_F16   // proj weights
    );

    FFNCost ffn_layer = estimate_ffn_closed_form(
        n_embd,
        n_ff,
        n_tokens,
        GGML_TYPE_Q8_0,  // FFN weights
        GGML_TYPE_F32    // activations
    );

    GroupAgg * g_mid = find_group_agg(GGML_DVFS_GRP_MID);
    if (g_mid) {
        g_mid->active        = true;
        g_mid->frame.group   = G_OTHER;  // Proj + FFN
        g_mid->frame.flops   = (proj_layer.flops + ffn_layer.flops) * n_layer;
        g_mid->frame.bytes   = (proj_layer.bytes + ffn_layer.bytes) * n_layer;
        g_mid->frame.ai      = g_mid->frame.bytes > 0.0
                                 ? g_mid->frame.flops / g_mid->frame.bytes
                                 : 0.0;
        total_flops += g_mid->frame.flops;
        total_bytes += g_mid->frame.bytes;
    }

    // ──────────────────────────────────────────────
    // 3) LM Head → Group3
    // ──────────────────────────────────────────────
    double fl_lm    = 0.0;
    double bytes_lm = 0.0;
    estimate_lmhead_closed_form(
        n_embd,
        n_vocab,
        n_tokens,
        GGML_TYPE_Q8_0,   // lm_head weight
        GGML_TYPE_F32,    // activations
        fl_lm,
        bytes_lm
    );

    GroupAgg * g_lm = find_group_agg(GGML_DVFS_GRP_LM);
    if (g_lm) {
        g_lm->active        = true;
        g_lm->frame.group   = G_LMHEAD;
        g_lm->frame.flops   = fl_lm;   // 한 번만
        g_lm->frame.bytes   = bytes_lm;
        g_lm->frame.ai      = g_lm->frame.bytes > 0.0
                                ? g_lm->frame.flops / g_lm->frame.bytes
                                : 0.0;
        total_flops += g_lm->frame.flops;
        total_bytes += g_lm->frame.bytes;
    }

    // 그룹별 로그
    for (auto & g : g_groups) {
        if (!g.active) continue;

        const char * gname = "UNKNOWN";
        if (g.gid == GGML_DVFS_GRP_ATT_CORE) gname = "KQV_CORE(QK^T+AV)";
        else if (g.gid == GGML_DVFS_GRP_MID) gname = "PROJ+FFN";
        else if (g.gid == GGML_DVFS_GRP_LM)  gname = "LMHEAD";

        printf("[AI] Group %s: FLOP=%.0f  Bytes=%.0f  AI=%.6f\n",
               gname, g.frame.flops, g.frame.bytes, g.frame.ai);
    }

    // ──────────────────────────────────────────────
    // MCKP용 GroupC/ChoiceC 구성
    // ──────────────────────────────────────────────
    std::vector<GroupC>               groups_c;
    std::vector<std::vector<ChoiceC>> choices_storage;
    std::vector<int>                  gid_list;

    for (auto & g : g_groups) {
        if (!g.active || g.frame.bytes <= 0.0) continue;

        FreqCandidates cand{};
        build_freq_candidates_for_group(g.gid, cand);

        auto choices = build_choices_for_mckp(g.frame, g_em, cand);
        if (choices.empty()) continue;

        choices_storage.emplace_back(std::move(choices));

        GroupC gc{};
        gc.name      = nullptr;
        gc.repeat    = 1;                            // 토큰당 1번
        gc.n_choices = choices_storage.back().size();
        gc.choices   = choices_storage.back().data();

        groups_c.push_back(gc);
        gid_list.push_back(g.gid);
    }

    if (groups_c.empty()) {
        printf("[mckp] no active group, skip\n");
        double total_ai = total_bytes > 0.0 ? total_flops / total_bytes : 0.0;
        printf("=== TOTAL: FLOP=%.0f  Bytes=%.0f  AI=%.6f ===\n",
               total_flops, total_bytes, total_ai);
        return;
    }

    // ──────────────────────────────────────────────
    // T_budget 설정
    //   - PREFILL : baseline + 10% slack
    //   - DECODE  : 300 ms 고정
    // ──────────────────────────────────────────────
    double T_min = 0.0;
    for (const auto & gc : groups_c) {
        if (gc.n_choices == 0) continue;
        double best = 1e300;
        for (size_t j = 0; j < gc.n_choices; ++j) {
            best = std::min(best, gc.choices[j].latency);
        }
        T_min += best * gc.repeat;
    }

    const double slack_ratio      = 0.10;
    const double T_budget_prefill = T_min * (1.0 + slack_ratio);

    double T_budget;
    if (stage == ST_PREFILL) {
        T_budget = T_budget_prefill;
    } else { // ST_DECODE 등
        T_budget = 300.0;   // ms
    }

    const double time_unit = 0.1;   // 0.1ms resolution

    DPResultC res = freq_table_mckp_solver_c(
        groups_c.data(),
        groups_c.size(),
        T_budget,
        time_unit,
        /*use_dp=*/true
    );

    if (!res.feasible) {
        printf("[mckp] infeasible, fallback to baseline\n");
        free_dpresult_c(&res);
        double total_ai = total_bytes > 0.0 ? total_flops / total_bytes : 0.0;
        printf("=== TOTAL: FLOP=%.0f  Bytes=%.0f  AI=%.6f ===\n",
               total_flops, total_bytes, total_ai);
        return;
    }

    // ──────────────────────────────────────────────
    // MCKP 결과 → 그룹별 freq 적용
    // ──────────────────────────────────────────────
    for (size_t gi = 0; gi < groups_c.size(); ++gi) {
        int idx = res.selected_index_per_group
                  ? res.selected_index_per_group[gi]
                  : 0;
        if (idx < 0 || (size_t) idx >= groups_c[gi].n_choices) {
            idx = 0;
        }

        const ChoiceC & ch = groups_c[gi].choices[idx];

        int cpu_khz = ch.c;
        int mem_khz = ch.m;
        int gid     = gid_list[gi];

        ggml_dvfs_set    (gid, cpu_khz);
        ggml_memfreq_set (gid, mem_khz);
        ggml_dvfs_apply_if_needed(gid);

        printf("[mckp dvfs] gid=%d  cpu=%d kHz  mem=%d kHz  (lat=%.3f ms, E=%.3f J)\n",
               gid, cpu_khz, mem_khz, ch.latency, ch.energy);
    }

    free_dpresult_c(&res);

    double total_ai = total_bytes > 0.0 ? total_flops / total_bytes : 0.0;
    printf("=== TOTAL: FLOP=%.0f  Bytes=%.0f  AI=%.6f ===\n",
           total_flops, total_bytes, total_ai);
}

// ──────────────────────────────────────────────
// maybe_probe / signal
// ──────────────────────────────────────────────
void maybe_probe_ai(
    const ggml_cgraph * graph,
    const llama_ubatch & ubatch,
    const llama_model  & model
) {
    if (!probe_requested.exchange(false)) {
        return;
    }
    ggml_analyze_arithmetic_intensity(graph, ubatch, model);
}

void setup_probe_signal() {
    std::signal(SIGUSR1, [](int){
        probe_requested.store(true);
    });
}