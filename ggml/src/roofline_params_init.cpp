// ggml/src/roofline_params_init.cpp
#include "roofline_params_init.h"
#include "roofline_gflops.h"

static inline RooflineGflopsParams make_params(
    double k,
    RooflineSweepAxis axis,
    int max_cands,
    double baseline_lat_us,
    double baseline_flops,
    int baseline_cpu_ridge,
    int baseline_mif_ridge
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

void init_roofline_params() {
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
        ST_PREFILL, G_KQV,
        RooflineGflopsParams{
            /*k_c1_over_c2=*/0.00323474289632642983,
            /*sweep_axis=*/RooflineSweepAxis::MEM_SWEEP,
            /*max_candidates=*/64,

            // Python ridge-scaling baseline
            /*baseline_lat_us=*/406207.799999999988,
            /*baseline_flops=*/8589930000,
            /*baseline_cpu_ridge=*/2687000,
            /*baseline_mif_ridge=*/1014000,
        }
    );

    roofline_set_params(
        ST_PREFILL, G_LMHEAD,
        RooflineGflopsParams{
            /*k_c1_over_c2=*/0.00626029284449171157,
            /*sweep_axis=*/RooflineSweepAxis::MEM_SWEEP,
            /*max_candidates=*/64,

            // Python ridge-scaling baseline
            /*baseline_lat_us=*/21112125.1999999993,
            /*baseline_flops=*/537945000000,
            /*baseline_cpu_ridge=*/2687000,
            /*baseline_mif_ridge=*/845000,
        }
    );

    roofline_set_params(
        ST_PREFILL, G_OTHER,
        RooflineGflopsParams{
            /*k_c1_over_c2=*/0.00706497641525288735,
            /*sweep_axis=*/RooflineSweepAxis::MEM_SWEEP,
            /*max_candidates=*/64,

            // Python ridge-scaling baseline
            /*baseline_lat_us=*/4868697.70000000019,
            /*baseline_flops=*/124596000000,
            /*baseline_cpu_ridge=*/2687000,
            /*baseline_mif_ridge=*/845000,
        }
    );

    roofline_set_params(
        ST_DECODE, G_KQV,
        RooflineGflopsParams{
            /*k_c1_over_c2=*/0.66506060199274919,
            /*sweep_axis=*/RooflineSweepAxis::MEM_SWEEP,
            /*max_candidates=*/64,

            // Python ridge-scaling baseline
            /*baseline_lat_us=*/903.355999999999995,
            /*baseline_flops=*/8396800,
            /*baseline_cpu_ridge=*/2687000,
            /*baseline_mif_ridge=*/1014000,
        }
    );

    roofline_set_params(
        ST_DECODE, G_OTHER,
        RooflineGflopsParams{
            /*k_c1_over_c2=*/0.613270524850239251,
            /*sweep_axis=*/RooflineSweepAxis::CPU_SWEEP,
            /*max_candidates=*/64,

            // Python ridge-scaling baseline
            /*baseline_lat_us=*/6880.48300000000017,
            /*baseline_flops=*/121676000,
            /*baseline_cpu_ridge=*/2294000,
            /*baseline_mif_ridge=*/3744000,
        }
    );

    roofline_set_params(
        ST_DECODE, G_LMHEAD,
        RooflineGflopsParams{
            /*k_c1_over_c2=*/1.98939752845365603,
            /*sweep_axis=*/RooflineSweepAxis::MEM_SWEEP,
            /*max_candidates=*/64,

            // Python ridge-scaling baseline
            /*baseline_lat_us=*/32385.7010000000009,
            /*baseline_flops=*/525337000,
            /*baseline_cpu_ridge=*/2687000,
            /*baseline_mif_ridge=*/1352000,
        }
    );
}