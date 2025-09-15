#include "arithmetic_intensity.h"
#include "ggml.h"
#include "ggml-impl.h"   

#include "freq_ai_policy.h"
#include "ggml-dvfs.h"

#include <cstdio>
#include <cstdint>
#include <array>
#include <atomic>
#include <csignal>

// ==== EWMA 토글 & 파라미터 ================================================
// 1: EWMA 사용, 0: 미사용(즉시값)
#ifndef GGML_AI_USE_EWMA
#define GGML_AI_USE_EWMA 1
#endif
// EWMA 알파 (0.0~1.0), 높을수록 최근값 가중↑
#ifndef GGML_AI_EWMA_ALPHA
#define GGML_AI_EWMA_ALPHA 0.20
#endif
// ==========================================================================

std::atomic<bool> probe_requested{true};
// 프로브 요청 API
inline void request_probe() {
    probe_requested.store(true);
}

// 그래프 분석 호출 함수
void ggml_analyze_arithmetic_intensity(const ggml_cgraph * graph);

// maybe_probe_api: 외부 요청이 있을 때만 AI 계산 실행
void maybe_probe_ai(const ggml_cgraph * graph);

// Op별 통계 정보
struct OpStats {
    double flops_per_elem;
    int    n_src;
    bool   writes_dst;
};

static std::array<OpStats, GGML_OP_COUNT> op_stats = [](){
    std::array<OpStats, GGML_OP_COUNT> a;
    // 기본값 채우기
    for(auto & s : a){
        s = { 0.0, /*n_src=*/0, /*writes_dst=*/false };
    }
    // 필요한 연산자만 덮어쓰기
    a[GGML_OP_ADD]       = { 1.0, 2, true  };
    a[GGML_OP_MUL]       = { 1.0, 2, true  };
    a[GGML_OP_RMS_NORM]  = { 2.0, 1, true  };
    a[GGML_OP_MUL_MAT]   = { 0.0, 2, true  }; // flops_per_elem 런타임 재계산
    a[GGML_OP_CPY]       = { 0.0, 1, true  };
    a[GGML_OP_CONT]      = { 0.0, 1, true  };
    a[GGML_OP_RESHAPE]   = { 0.0, 1, false };
    a[GGML_OP_VIEW]      = { 0.0, 1, false };
    a[GGML_OP_PERMUTE]   = { 0.0, 1, false };
    a[GGML_OP_TRANSPOSE] = { 0.0, 1, false };
    a[GGML_OP_SOFT_MAX]  = { 3.0, 1, true  };
    a[GGML_OP_ROPE]      = { 2.0, 2, true  };
    a[GGML_OP_GLU]       = { 2.0, 2, true  };
    // … 추가가 필요하면 여기에 더 …
    return a;
}();

// ──────────────────────────────────────────────────────────────
// 정책(decider) + (옵션) op별 EWMA 상태
// ──────────────────────────────────────────────────────────────
static FreqDecision g_decider;

#if GGML_AI_USE_EWMA
struct OpAgg { double ewma_ai = 0.0; bool init = false; };
static std::array<OpAgg, GGML_OP_COUNT> g_op_ai;
#endif

static inline double ewma(double prev, double x, double alpha) {
    return prev * (1.0 - alpha) + x * alpha;
}

// 최초 1회 정책 초기화 (device caps/정책은 실제 값으로 교체 권장)
static void ensure_decider_initialized() {
    static bool inited = false;
    if (inited) return;
    inited = true;

    RooflineCaps caps{
        /*peak_flops=*/1.2e12,  // TODO: system-data-profiler 실측치 입력
        /*peak_bw=*/   30e9     // TODO: system-data-profiler 실측치 입력
    };
    PolicyConfig cfg{};
    // 필요시 cfg.low_margin / high_margin / cooldown_us 등 조정
    g_decider.configure(caps, cfg);

    // 초기 스냅샷(온도/배터리/쿼리 길이 등은 외부에서 주기적으로 업데이트 가능)
    g_decider.update_system(SystemSnapshot{.therm_scale=1.0, .batt_scale=1.0});
    g_decider.update_query (QueryContext{.predicted_len_tokens=-1, .latency_budget_ms=-1});

    // RL 훅 사용 시 set_rl_hook(...) 등록
    g_decider.set_rl_hook(nullptr);
}

// 외부에서 시스템/쿼리 스냅샷 갱신
void ggml_freq_policy_update_system(const SystemSnapshot& s) { ensure_decider_initialized(); g_decider.update_system(s); }
void ggml_freq_policy_update_query (const QueryContext&   q) { ensure_decider_initialized(); g_decider.update_query(q);  }

// ──────────────────────────────────────────────────────────────
// 메인: 그래프 노드들을 훑어 AI(FLOPs/Bytes) 계산 + 정책 호출
// ──────────────────────────────────────────────────────────────

void ggml_analyze_arithmetic_intensity(const ggml_cgraph * graph) {
    ensure_decider_initialized();
    double total_flops  = 0.0;
    double total_bytes  = 0.0;

    for (int i = 0; i < graph->n_nodes; ++i) {
        const ggml_tensor * dst  = graph->nodes[i];
        OpStats stats            = op_stats[dst->op];

        /* --- FLOPs ---------------------------------------------------- */
        double flops = 0.0;
        if (dst->op == GGML_OP_MUL_MAT) {
            const ggml_tensor * A = dst->src[0];
            const ggml_tensor * B = dst->src[1];
            int64_t M = dst->ne[0];           // row
            int64_t N = dst->ne[1];           // col
            int64_t K = A->ne[0];             // 공통 차원 (A row == K)
            flops = 2.0 * (double)M * N * K;  // 2*M*N*K
        } else {
            flops = stats.flops_per_elem * ggml_nelements(dst);
        }

        /* --- Bytes ---------------------------------------------------- */
        double bytes = 0.0;
        for (int si = 0; si < stats.n_src; ++si) {
            bytes += ggml_nbytes(dst->src[si]);   // 입력별 실제 바이트
        }
        if (stats.writes_dst) {
            bytes += ggml_nbytes(dst);
        }

        /* --- 출력 ---------------------------------------------------- */
        double ai = bytes ? flops / bytes : 0.0;  // divide-by-zero guard

        // === EWMA 토글: 정책에 넘길 AI 결정 ===
#if GGML_AI_USE_EWMA
        auto &agg = g_op_ai[dst->op];
        if (!agg.init) { agg.ewma_ai = ai; agg.init = true; }
        else           { agg.ewma_ai = ewma(agg.ewma_ai, ai, GGML_AI_EWMA_ALPHA); }
        const double ai_for_policy = agg.ewma_ai;
#else
        const double ai_for_policy = ai;   // 즉시값 사용
#endif

        OpContext oc{ .op_id = dst->op, .ai_ewma = ai_for_policy };
        Decision d = g_decider.decide_and_schedule(oc);
        // CPU / MEM 각각 action table 갱신
        ggml_dvfs_set    (dst->op, d.cpu_khz);
        ggml_memfreq_set (dst->op, d.mem_khz);

        printf("node[%2d]: op=%-12s  FLOP=%12.0f  Bytes=%12.0f  AI=%6.2f\n",
               i, ggml_op_name(dst->op), flops, bytes, ai);

        total_flops  += flops;
        total_bytes  += bytes;
    }

    double total_ai = total_bytes ? total_flops / total_bytes : 0.0;
    printf("=== TOTAL:  FLOP=%0.f  Bytes=%0.f  AI=%0.2f ===\n",
           total_flops, total_bytes, total_ai);
}



void maybe_probe_ai(const ggml_cgraph * graph) {
    // 요청 플래그가 false일 경우 아무 작업 없이 반환
    if (!probe_requested.exchange(false)) {
        return;
    }
    ggml_analyze_arithmetic_intensity(graph);
}

void setup_probe_signal() {
    std::signal(SIGUSR1, [](int){ probe_requested.store(true); });
}
