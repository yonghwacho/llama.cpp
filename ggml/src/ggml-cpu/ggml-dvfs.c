#include "ggml-dvfs.h"
#include <stdio.h>

/* ---- 실제 전역 메모리 정의 ---- */
_Atomic int g_op_freq_table[GGML_DVFS_MAX_OP] = {0};

/* 내부 전용 – 현재 적용된 주파수 기록 */
static _Atomic int g_applied_khz = 0;

/* forward decl. */
static void set_cpu_freq(const char *freq_str);

/* --------- 공개 API --------- */
void ggml_dvfs_set(int op_id, int khz)
{
    if (op_id < 0 || op_id >= GGML_DVFS_MAX_OP) return;
    atomic_store_explicit(&g_op_freq_table[op_id], khz,
                          memory_order_relaxed);
}

int ggml_dvfs_get(int op_id)
{
    if (op_id < 0 || op_id >= GGML_DVFS_MAX_OP) return 0;
    return atomic_load_explicit(&g_op_freq_table[op_id],
                                memory_order_relaxed);
}

/* --------- ggml 내부에서만 쓰이는 헬퍼 --------- */
void ggml_dvfs_apply_if_needed(int op_id)
{
#if defined(__gnu_linux__) || defined(__ANDROID__)
    int want = atomic_load_explicit(&g_op_freq_table[op_id],
                                    memory_order_relaxed);
    if (want > 0 &&
        want != atomic_load_explicit(&g_applied_khz,
                                     memory_order_relaxed)) {

        char buf[16];
        sprintf(buf, "%d", want);
        set_cpu_freq(buf);
        atomic_store_explicit(&g_applied_khz, want,
                              memory_order_relaxed);
    }
#else
    (void)op_id;
#endif
}

/* 실제 sysfs write — root 권한 필요 */
static void set_cpu_freq(const char *freq_str)
{
#if defined(__gnu_linux__) || defined(__ANDROID__)
    static const char *cpus[] = {
        "/sys/devices/system/cpu/cpu4/cpufreq/scaling_max_freq",
        "/sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq",
    };
    for (size_t i = 0; i < sizeof(cpus)/sizeof(cpus[0]); ++i) {
        FILE *f = fopen(cpus[i], "w");
        if (f) { fprintf(f, "%s", freq_str); fclose(f); }
    }
#else
    (void)freq_str;
#endif
}
