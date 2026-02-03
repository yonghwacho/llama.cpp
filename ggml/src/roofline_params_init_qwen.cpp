// ggml/src/roofline_params_init.cpp
#include "roofline_params_init.h"
#include "roofline_gflops.h"

static inline RooflineGflopsParams make_params(
    double k,
    RooflineSweepAxis axis,
    int max_cands,
    double baseline_lat_us,
    double baseline_flops,
    int64_t baseline_cpu_ridge,
    int64_t baseline_mif_ridge
) {
    RooflineGflopsParams p{};
    p.k_c1_over_c2         = k;
    p.sweep_axis           = axis;
    p.max_candidates       = max_cands;

    // ✅ python latency scaling baseline 세트
    p.baseline_lat_us      = baseline_lat_us;
    p.baseline_flops       = baseline_flops;
    p.baseline_cpu_ridge   = baseline_cpu_ridge;
    p.baseline_mif_ridge   = baseline_mif_ridge;
    return p;
}

void init_qwen_roofline_params() {
    roofline_set_params(
        ST_DECODE, G_OTHER,
        RooflineGflopsParams{
            /*k_c1_over_c2=*/0.668275894274126148,
            /*sweep_axis=*/RooflineSweepAxis::CPU_SWEEP,
            /*max_candidates=*/64,

            // Python ridge-scaling baseline
            /*baseline_lat_us=*/4052.404,
            /*baseline_flops=*/67663900,
            /*baseline_cpu_ridge=*/2499000,
            /*baseline_mif_ridge=*/3744000,
        }
    );

    roofline_set_params(
        ST_DECODE, G_KQV,
        RooflineGflopsParams{
            /*k_c1_over_c2=*/0.335204595501860603,
            /*sweep_axis=*/RooflineSweepAxis::MEM_SWEEP,
            /*max_candidates=*/64,

            // Python ridge-scaling baseline
            /*baseline_lat_us=*/542.721000000000004,
            /*baseline_flops=*/3078140,
            /*baseline_cpu_ridge=*/2687000,
            /*baseline_mif_ridge=*/1352000,
        }
    );

    roofline_set_params(
        ST_DECODE, G_LMHEAD,
        RooflineGflopsParams{
            /*k_c1_over_c2=*/0.574205360544610355,
            /*sweep_axis=*/RooflineSweepAxis::CPU_SWEEP,
            /*max_candidates=*/64,

            // Python ridge-scaling baseline
            /*baseline_lat_us=*/23335.3630000000012,
            /*baseline_flops=*/466747000,
            /*baseline_cpu_ridge=*/2147000,
            /*baseline_mif_ridge=*/3744000,
        }
    );

    roofline_set_params(
        ST_PREFILL, G_LMHEAD,
        RooflineGflopsParams{
            /*k_c1_over_c2=*/0.00878507699970807431,
            /*sweep_axis=*/RooflineSweepAxis::MEM_SWEEP,
            /*max_candidates=*/64,

            // Python ridge-scaling baseline
            /*baseline_lat_us=*/9205335.90000000037,
            /*baseline_flops=*/233374000000,
            /*baseline_cpu_ridge=*/2687000,
            /*baseline_mif_ridge=*/1014000,
        }
    );

    roofline_set_params(
        ST_PREFILL, G_KQV,
        RooflineGflopsParams{
            /*k_c1_over_c2=*/0.0092746548230724854,
            /*sweep_axis=*/RooflineSweepAxis::MEM_SWEEP,
            /*max_candidates=*/64,

            // Python ridge-scaling baseline
            /*baseline_lat_us=*/94475,
            /*baseline_flops=*/1536000000,
            /*baseline_cpu_ridge=*/2687000,
            /*baseline_mif_ridge=*/676000,
        }
    );

    roofline_set_params(
        ST_PREFILL, G_OTHER,
        RooflineGflopsParams{
            /*k_c1_over_c2=*/0.00310359619941370183,
            /*sweep_axis=*/RooflineSweepAxis::MEM_SWEEP,
            /*max_candidates=*/64,

            // Python ridge-scaling baseline
            /*baseline_lat_us=*/1330089.30000000005,
            /*baseline_flops=*/33831900000,
            /*baseline_cpu_ridge=*/2687000,
            /*baseline_mif_ridge=*/3172000,
        }
    );
}