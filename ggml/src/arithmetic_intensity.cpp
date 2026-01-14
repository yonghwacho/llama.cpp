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
#include "roofline_gflops.h"

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

// gid(GGML_DVFS_GRP_*) → perf_group(G_KQV, G_OTHER, G_LMHEAD) 매핑
static inline perf_group perf_group_from_gid(int gid) {
    switch (gid) {
        case GGML_DVFS_GRP_ATT_CORE:
            return G_KQV;       // Attention core → KQV 그룹
        case GGML_DVFS_GRP_MID:
            return G_OTHER;     // Proj + FFN   → OTHER
        case GGML_DVFS_GRP_LM:
            return G_LMHEAD;    // LM Head      → LMHEAD
        default:
            return G_OTHER;     // fallback
    }
}

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
    const int bW  = bpp_ggml_type(GGML_TYPE_F16);
    const int bA  = bpp_ggml_type(GGML_TYPE_F32);  // activations
    const int bKV = bpp_ggml_type(GGML_TYPE_F16);  // KV cache

    const double bytes_w_wo   = dE * dE * bW;
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

    // -------------------------
    // FLOPs  (원래 GROUP2와 동일)
    // -------------------------
    //   Q : 2 * E * E * N
    //   K : 2 * E * (D * Hkv) * N
    //   V : 2 * E * (D * Hkv) * N
    //   Wo: 2 * E * E * N
    const double fl_q  = 2.0 * dE * dE * dN;
    const double fl_k  = 2.0 * dE * (dD * dHkv) * dN;
    const double fl_v  = fl_k;
    const double fl_wo = 2.0 * dE * dE * dN;

    const double flops_total = fl_q + fl_k + fl_v + fl_wo;

    // -------------------------
    // Bytes (Wq/Wk/Wv/Wo + X/Q/K/V/Y/out)
    //   - QKV 블록:
    //       weights: Wq, Wk, Wv
    //       acts   : X + Q + K + V
    //   - out proj 블록:
    //       weights: Wo
    //       acts   : Y(in) + out
    // -------------------------
    const int bW = bpp_ggml_type(wtype);
    const int bA = bpp_ggml_type(GGML_TYPE_F32); // act_ty 대신 F32 고정 근사

    // weights (INCLUDE_WEIGHT_BYTES = true 가정)
    const double bytes_wq = dE * dE         * bW;            // Wq
    const double bytes_wk = dE * (dD*dHkv)  * bW;            // Wk
    const double bytes_wv = dE * (dD*dHkv)  * bW;            // Wv
    const double bytes_wo = dE * dE         * bW;            // Wo

    // activations
    // QKV 블록: X + Q + K + V
    const double bytes_X   = dE * dN * bA;                   // X
    const double bytes_Q   = dE * dN * bA;                   // Q  (dH*dD = dE)
    const double bytes_K   = (dD * dHkv) * dN * bA;          // K
    const double bytes_V   = (dD * dHkv) * dN * bA;          // V

    // out proj 블록: Y(in) + out
    const double bytes_Y   = dE * dN * bA;                   // Y (input to Wo)
    const double bytes_out = dE * dN * bA;                   // out (output of Wo)

    const double bytes_total =
        (bytes_wq + bytes_wk + bytes_wv + bytes_wo) +
        (bytes_X + bytes_Q + bytes_K + bytes_V + bytes_Y + bytes_out);

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

// 원래 tllm_profiler 의 FFN cost (estimate_ffn_cost_closed_form_runtime)와 동일한 식
static inline FFNCost estimate_ffn_closed_form(
    int E,              // n_embd
    int F,              // n_ff (expand size)
    int N,              // n_tokens
    ggml_type wtype,    // weight dtype (e.g. GGML_TYPE_Q8_0)
    ggml_type act_type  // activation dtype (e.g. GGML_TYPE_F32)
) {
    if (E <= 0 || F <= 0 || N <= 0) {
        return { 0.0, 0.0 };
    }

    const double dE = (double) E;
    const double dF = (double) F;
    const double dN = (double) N;

    // FLOPs (행렬곱은 2*M*N*K 규약, SiLU/elemwise는 근사)
    const double fl_up     = 2.0 * dF * dN * dE;  // w_up[E,F]   * X[E,N]   -> [F,N]
    const double fl_gate   = 2.0 * dF * dN * dE;  // w_gate[E,F] * X[E,N]   -> [F,N]
    const double fl_silu   = 4.0 * dF * dN;       // 근사 (원하면 0으로 둬도 무방)
    const double fl_fused  = 1.0 * dF * dN;       // elementwise mul
    const double fl_down   = 2.0 * dE * dN * dF;  // w_down[F,E] * fused[F,N] -> [E,N]

    const double flops_total = fl_up + fl_gate + fl_silu + fl_fused + fl_down;

    // Bytes (최소 근사: 각 텐서를 1-pass로 다룬다고 가정)
    const int bW = bpp_ggml_type(wtype);
    const int bA = bpp_ggml_type(act_type);

    const double bytes_w_up   = dE * dF * bW;
    const double bytes_w_gate = dE * dF * bW;
    const double bytes_w_down = dF * dE * bW;

    const double bytes_X      = dE * dN * bA;   // read
    const double bytes_up     = dF * dN * bA;   // matmul output
    const double bytes_gate   = dF * dN * bA;   // matmul output
    const double bytes_fused  = dF * dN * bA;   // mul output
    const double bytes_out    = dE * dN * bA;   // final write

    const double bytes_total =
        (bytes_w_up + bytes_w_gate + bytes_w_down) +
        (bytes_X + bytes_up + bytes_gate + bytes_fused + bytes_out);

    FFNCost fc{};
    fc.flops = flops_total;
    //printf("FFN_flops %6f", flops_total);
    fc.bytes = bytes_total;
    //printf("FFN_bytes %6f", bytes_total);
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
    init_energy_model();

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
        // printf("[AI] invalid hparams: n_tokens=%d, n_embd=%d, n_layer=%d\n",
        //        n_tokens, n_embd, n_layer);
        return;
    }

    ggml_stage stage = infer_stage_from_ubatch(ubatch);
    // printf("[AI] stage=%s, n_tokens=%d, n_ctx=%d\n",
    //        stage == ST_PREFILL ? "PREFILL" : "DECODE",
    //        n_tokens, n_ctx);

    // printf("[AI] params: n_layer=%d, n_embd=%d, n_head=%d, n_head_kv=%d, "
    //        "n_ff=%d, n_vocab=%d, n_tokens=%d, n_ctx=%d\n",
    //        n_layer, n_embd, n_head, n_head_kv, n_ff, n_vocab, n_tokens, n_ctx);

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
        g_att->frame.flops   = att_core.flops;
        g_att->frame.bytes   = att_core.bytes;
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
        GGML_TYPE_F16,  // FFN weights
        GGML_TYPE_F32    // activations
    );

    GroupAgg * g_mid = find_group_agg(GGML_DVFS_GRP_MID);
    if (g_mid) {
        g_mid->active        = true;
        g_mid->frame.group   = G_OTHER;  // Proj + FFN
        g_mid->frame.flops   = (proj_layer.flops + ffn_layer.flops);
        g_mid->frame.bytes   = (proj_layer.bytes + ffn_layer.bytes);
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
        GGML_TYPE_F16,   // lm_head weight
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

    choices_storage.reserve(10);  // 충분히 큰 값
    groups_c.reserve(10);
    gid_list.reserve(10);

    for (auto & g : g_groups) {
        if (!g.active || g.frame.bytes <= 0.0) continue;

        FreqCandidates cand{};
        build_freq_candidates_for_group(g.gid, cand);

        auto choices = roofline_build_ridge_choices(
            g.frame,
            cand,
            get_energy_cb(),
            (void*)&g_em
        );
        if (choices.empty()) continue;

        choices_storage.emplace_back(std::move(choices));

        GroupC gc{};
        gc.name      = nullptr;
        perf_group pg = perf_group_from_gid(g.gid);

        // 🔥 FIX: repeat 값 설정 (이 부분이 완전히 빠져있었음!)
        int repeat = 1;
        if (pg == G_LMHEAD) {
            repeat = 1;  // LM Head는 1번만
        } else {
            repeat = n_layer;  // KQV, OTHER는 레이어 수만큼
        }
        gc.repeat = repeat;

        // 🔍 디버깅: 포인터 주소 확인
        // printf("[DEBUG] gi=%zu, gid=%d, pg=%d, repeat=%d, choices ptr=%p, size=%zu, first c=%d, first m=%d\n",
        //     groups_c.size(),
        //     g.gid,
        //     (int)pg,
        //     repeat,
        //     (void*)choices_storage.back().data(),
        //     choices_storage.back().size(),
        //     choices_storage.back()[0].c,
        //     choices_storage.back()[0].m);
        
        gc.n_choices = choices_storage.back().size();
        gc.choices   = choices_storage.back().data();

        groups_c.push_back(gc);
        gid_list.push_back(g.gid);
    }
    if (groups_c.empty()) {
        // printf("[mckp] no active group, skip\n");
        double total_ai = total_bytes > 0.0 ? total_flops / total_bytes : 0.0;
        // printf("=== TOTAL: FLOP=%.0f  Bytes=%.0f  AI=%.6f ===\n",
        //        total_flops, total_bytes, total_ai);
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

    const double slack_ratio      = 0.01;
    const double T_budget_prefill = T_min * (1.0 + slack_ratio);

    double T_budget;
    if (stage == ST_PREFILL) {
        T_budget = T_budget_prefill;
    } else { // ST_DECODE 등
        T_budget = 200.0;   // ms
    }

    const double time_unit = 0.1;   // 0.1ms resolution

    // 🔍 MCKP 호출 직전 검증
    for (size_t i = 0; i < groups_c.size(); ++i) {
        // printf("[DEBUG] Before MCKP gi=%zu, ptr=%p, n_choices=%zu, first c=%d, first m=%d\n",
        //     i,
        //     (void*)groups_c[i].choices,
        //     groups_c[i].n_choices,
        //     groups_c[i].choices[0].c,
        //     groups_c[i].choices[0].m);
    }

    DPResultC res = freq_table_mckp_solver_c(
        groups_c.data(),
        groups_c.size(),
        T_budget,
        time_unit,
        /*use_dp=*/true
    );

    // 🔍 MCKP 결과 직후 검증
    for (size_t gi = 0; gi < groups_c.size(); ++gi) {
        int idx = res.selected_index_per_group[gi];
        // // printf("[DEBUG] After MCKP gi=%zu, selected_idx=%d, ptr=%p\n",
        //     gi, idx, (void*)groups_c[gi].choices);
        
        const ChoiceC & ch = groups_c[gi].choices[idx];
        // printf("[DEBUG] Selected choice: c=%d, m=%d, lat=%.3f, E=%.6f\n",
        //     ch.c, ch.m, ch.latency, ch.energy);
    }

    if (!res.feasible) {
        //printf("[mckp] infeasible, fallback to baseline\n");
        free_dpresult_c(&res);
        double total_ai = total_bytes > 0.0 ? total_flops / total_bytes : 0.0;
        // printf("=== TOTAL: FLOP=%.0f  Bytes=%.0f  AI=%.6f ===\n",
        //        total_flops, total_bytes, total_ai);
        return;
    }

    // ──────────────────────────────────────────────
    // MCKP 결과 → stage × perf_group 계획으로 저장
    //   runtime에서는 ggml_dvfs_begin_stage() + ggml_dvfs_apply_if_needed()
    //   가 이 계획을 보고 sysfs를 실제로 건드림
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

        // gid(GGML_DVFS_GRP_*) → perf_group(G_KQV / G_OTHER / G_LMHEAD)
        perf_group pg = perf_group_from_gid(gid);

        // 🔥 여기서 "이번 ubatch의 stage"에 대한 그룹 DVFS plan을 설정
        ggml_dvfs_set_group_plan(stage, pg, cpu_khz, mem_khz);

        printf("[mckp dvfs] stage=%s gid=%d (perf_group=%d)  cpu=%d kHz  mem=%d kHz"
            "  (lat=%.3f ms, E=%.3f J)\n",
            stage == ST_PREFILL ? "PREFILL" : "DECODE",
            gid, (int)pg, cpu_khz, mem_khz, ch.latency, ch.energy);
    }

    free_dpresult_c(&res);

    double total_ai = total_bytes > 0.0 ? total_flops / total_bytes : 0.0;
    // printf("=== TOTAL: FLOP=%.0f  Bytes=%.0f  AI=%.6f ===\n",
    //        total_flops, total_bytes, total_ai);
}

// ──────────────────────────────────────────────
// maybe_probe / signal
// ──────────────────────────────────────────────
void maybe_probe_ai(
    const ggml_cgraph * graph,
    const llama_ubatch & ubatch,
    const llama_model  & model
) {
    ggml_analyze_arithmetic_intensity(graph, ubatch, model);
}

void setup_probe_signal() {
    std::signal(SIGUSR1, [](int){
        probe_requested.store(true);
    });
}