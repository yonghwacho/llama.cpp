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

void init_llama3b_roofline_params() {
    // ------------------------------------------------------------
    // 예시: 너가 준 profile_summary 한 줄을 ST_PREFILL + G_KQV에 넣는 예
    //
    // n_tokens_profile=1024
    // baseline_lat_us=903.356
    // baseline_flops=8396800
    // k_cpu=0.1801, k_mif=0.6650, k_median=0.42259
    // cpu_needed=2687000, mif_needed=1014000 (예시 row 기준)
    // ------------------------------------------------------------

    const int MAX_CANDS_DEFAULT = 64;

    // ✅ “median method”를 쓸 거면 baseline_cpu_ridge / baseline_mif_ridge도 median 방식에 맞게
    // python 코드에서 median은:
    // baseline_cpu_ridge = cpu_needed
    // baseline_mif_ridge = mif_needed
    roofline_set_params(
        ST_DECODE, G_KQV,
        RooflineGflopsParams{
            /*k_c1_over_c2=*/0.000204661126952782486,
            /*sweep_axis=*/RooflineSweepAxis::CPU_SWEEP,
            /*max_candidates=*/64,

            // Python ridge-scaling baseline
            /*baseline_lat_us=*/589.780999999999949,
            /*baseline_flops=*/12595200,
            /*baseline_cpu_ridge=*/1958400,
            /*baseline_mif_ridge=*/3199000000,
        }
    );

    roofline_set_params(
        ST_DECODE, G_LMHEAD,
        RooflineGflopsParams{
            /*k_c1_over_c2=*/0.000612599175772006563,
            /*sweep_axis=*/RooflineSweepAxis::CPU_SWEEP,
            /*max_candidates=*/64,

            // Python ridge-scaling baseline
            /*baseline_lat_us=*/25429.497000000003,
            /*baseline_flops=*/788005000,
            /*baseline_cpu_ridge=*/1958400,
            /*baseline_mif_ridge=*/3199000000,
        }
    );

    roofline_set_params(
        ST_DECODE, G_LMHEAD,
        RooflineGflopsParams{
            /*k_c1_over_c2=*/0.000612599175772006563,
            /*sweep_axis=*/RooflineSweepAxis::CPU_SWEEP,
            /*max_candidates=*/64,

            // Python ridge-scaling baseline
            /*baseline_lat_us=*/25429.497000000003,
            /*baseline_flops=*/788005000,
            /*baseline_cpu_ridge=*/1958400,
            /*baseline_mif_ridge=*/3199000000,
        }
    );


    roofline_set_params(
        ST_PREFILL, G_LMHEAD,
        RooflineGflopsParams{
            /*k_c1_over_c2=*/7.9486271643291141e-06,
            /*sweep_axis=*/RooflineSweepAxis::MEM_SWEEP,
            /*max_candidates=*/64,

            // Python ridge-scaling baseline
            /*baseline_lat_us=*/3019778.89999999991,
            /*baseline_flops=*/394002000000,
            /*baseline_cpu_ridge=*/1984000,
            /*baseline_mif_ridge=*/665600000,
        }
    );

    roofline_set_params(
        ST_PREFILL, G_OTHER,
        RooflineGflopsParams{
            /*k_c1_over_c2=*/1.66336339734317193e-06,
            /*sweep_axis=*/RooflineSweepAxis::CPU_SWEEP,
            /*max_candidates=*/64,

            // Python ridge-scaling baseline
            /*baseline_lat_us=*/1199097.30000000005,
            /*baseline_flops=*/138443000000,
            /*baseline_cpu_ridge=*/1881600,
            /*baseline_mif_ridge=*/3199000000,
        }
    );

    roofline_set_params(
        ST_PREFILL, G_KQV,
        RooflineGflopsParams{
            /*k_c1_over_c2=*/1.56849015317286657e-06,
            /*sweep_axis=*/RooflineSweepAxis::CPU_SWEEP,
            /*max_candidates=*/64,

            // Python ridge-scaling baseline
            /*baseline_lat_us=*/74431.5,
            /*baseline_flops=*/3072000000,
            /*baseline_cpu_ridge=*/1881600,
            /*baseline_mif_ridge=*/3199000000,
        }
    );
}