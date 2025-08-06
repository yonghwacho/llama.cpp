#include "arithmetic_intensity.h"
#include "ggml.h"
#include "ggml-impl.h"   

#include <cstdio>
#include <cstdint>
#include <array>
#include <atomic>
#include <csignal>


// 요청 플래그 선언
extern std::atomic<bool> probe_requested;

// 프로브 요청 API
inline void request_probe() {
    probe_requested.store(true);
}

// 그래프 분석 호출 함수
void ggml_analyze_arithmetic_intensity(const ggml_cgraph * graph);

// maybe_probe_api: 외부 요청이 있을 때만 AI 계산 실행
void maybe_probe_ai(const ggml_cgraph * graph);


// arithmetic_intensity.cpp
#include "arithmetic_intensity.h"
#include "ggml.h"
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <atomic>
#include <csignal>

// 요청 플래그 정의
std::atomic<bool> probe_requested{true};

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

// void ggml_analyze_arithmetic_intensity(const ggml_cgraph * graph) {
//     double total_flops = 0.0;
//     double total_bytes = 0.0;

//     for (int i = 0; i < graph->n_nodes; ++i) {
//         const ggml_tensor * node = graph->nodes[i];
//         OpStats stats = op_stats[node->op];

//         // 행렬곱인 경우 요소당 FLOP 재계산
//         if (node->op == GGML_OP_MUL_MAT) {
//             int64_t K = node->ne[1];
//             stats.flops_per_elem = 2.0 * K;
//         }

//         int64_t elems = ggml_nelements(node);
//         size_t  type_size = ggml_type_size(node->type);

//         double flops = stats.flops_per_elem * elems;
//         double bytes = (double)type_size * elems * stats.n_src
//                      + (stats.writes_dst ? (double)type_size * elems : 0.0);

//         printf("node[%2d]: op=%-12s elems=%8lld  FLOP=%10.0f  Bytes=%10.0f  AI=%6.2f\n",
//                i,
//                ggml_op_name(node->op),
//                (long long)elems,
//                flops,
//                bytes,
//                flops / bytes);

//         total_flops += flops;
//         total_bytes += bytes;
//     }

//     printf("=== TOTAL: FLOP=%.0f  Bytes=%.0f  AI=%.2f ===\n",
//            total_flops, total_bytes, total_flops / total_bytes);
// }


void ggml_analyze_arithmetic_intensity(const ggml_cgraph * graph) {
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
