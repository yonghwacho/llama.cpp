#include "ggml-dvfs.h"
#include "ggml-mckp-freq.h" // (10.27, wjbang) MCKP solver header
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>   // clock_gettime
#include <stdatomic.h>

// ---- 실제 전역 메모리 정의 (per-op override 용) ----
_Atomic int g_op_freq_table[GGML_DVFS_MAX_OP]     = {0};
_Atomic int g_op_memfreq_table[GGML_DVFS_MAX_OP]  = {0};

/* 내부 전용 – 현재 적용된 주파수 기록 (실제 sysfs에 적용된 값) */
static _Atomic int g_applied_khz      = 0;
static _Atomic int g_applied_mem_khz  = 0;

/* (stage, group) 별 DVFS 계획 – 헤더에서 extern 한 것을 여기서 정의 */
struct GroupDvfsPlan g_group_dvfs_plan[GGML_STAGE_COUNT][PERF_G_COUNT] = {0};

/* 현재 stage / 마지막 사용 그룹 (GOO-wise DVFS용) */
static _Atomic int g_cur_stage  = ST_DECODE;
static _Atomic int g_last_group = -1;

/* forward decl. */
static void set_cpu_freq(const char *freq_str);
static void set_mem_freq(const char *freq_str);

static int get_cpu_freq(void) {
    FILE *f = fopen("/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq", "r");
    if (!f) return -1;
    int freq = 0;
    fscanf(f, "%d", &freq);
    fclose(f);
    return freq;
}

/* ====== ns 타임스탬프 헬퍼 ====== */
static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

/* =========================================================
 *  perf_group 매핑: 텐서 이름 -> G_KQV / G_OTHER / G_LMHEAD
 *
 *  - llama-graph.cpp 기준 실제 이름:
 *      "attn_norm", "attn_norm_w",
 *      "kqv_out", "kqv_wo",
 *      "ffn_norm", "ffn_norm_w",
 *      "ffn_up", "ffn_gate", "ffn_down", ...
 *
 *  - 정의:
 *      G_KQV    : QK^T + AV core => "kqv_out" 만 해당
 *      G_OTHER  : QKV proj + attn out proj + FFN, norm 등
 *      G_LMHEAD : lm_head / logits 등 (이 파일에는 이름이 없을 수도 있음)
 * ========================================================= */
static perf_group ggml_node_to_group(const char *name) {
    if (name == NULL || name[0] == '\0') {
        int last = atomic_load_explicit(&g_last_group, memory_order_relaxed);
        if (last >= 0 && last < PERF_G_COUNT) {
            return (perf_group) last;
        }
        return G_OTHER;
    }

    // LM HEAD: 최종 출력 쪽 (이름 규칙은 필요시 추가)
    if (strstr(name, "lm_head")  != NULL ||
        strstr(name, "lmhead")   != NULL ||
        strstr(name, "output_head") != NULL ||
        strstr(name, "logits")   != NULL) {
        return G_LMHEAD;
    }

    // KQV core: scaled dot-product attention 핵심 결과
    // llama-graph.cpp에서 "kqv_out" 으로 이름 붙여짐
    if (strstr(name, "kqv_out") != NULL) {
        return G_KQV;
    }

    // 그 외: QKV proj, attn_norm, FFN(up/gate/down), norm 등 모두 OTHER
    return G_OTHER;
}

/* --------- 공개 API: per-op override (이전과 동일) --------- */
void ggml_dvfs_set(int op_id, int khz) {
    if (op_id < 0 || op_id >= GGML_DVFS_MAX_OP) return;
    atomic_store_explicit(&g_op_freq_table[op_id], khz, memory_order_relaxed);
}

int ggml_dvfs_get(int op_id) {
    if (op_id < 0 || op_id >= GGML_DVFS_MAX_OP) return 0;
    return atomic_load_explicit(&g_op_freq_table[op_id], memory_order_relaxed);
}

void ggml_memfreq_set(int op_id, int khz) {
    if (op_id < 0 || op_id >= GGML_DVFS_MAX_OP) return;
    atomic_store_explicit(&g_op_memfreq_table[op_id], khz, memory_order_relaxed);
}

int ggml_memfreq_get(int op_id) {
    if (op_id < 0 || op_id >= GGML_DVFS_MAX_OP) return 0;
    return atomic_load_explicit(&g_op_memfreq_table[op_id], memory_order_relaxed);
}

/* --------- NEW: stage / group 기반 계획 세팅 API --------- */
void ggml_dvfs_begin_stage(ggml_stage st) {
    atomic_store_explicit(&g_cur_stage, (int) st, memory_order_relaxed);
    // stage 바뀔 때 그룹도 리셋해서 첫 노드에서 반드시 한 번은 적용
    atomic_store_explicit(&g_last_group, -1, memory_order_relaxed);
}

void ggml_dvfs_set_group_plan(ggml_stage st, perf_group g, int cpu_khz, int mem_khz) {
    if (st < 0 || st >= GGML_STAGE_COUNT) return;
    if (g  < 0 || g  >= PERF_G_COUNT)     return;
    g_group_dvfs_plan[st][g].cpu_khz = cpu_khz;
    g_group_dvfs_plan[st][g].mem_khz = mem_khz;
}

/* --------- ggml 내부에서 쓰이는 헬퍼 (노드 단위) ---------
 *   - op_id: ggml_op 를 int로 캐스팅한 값 (호출부에서 넘김)
 *   - name : tensor 이름 (llama-graph.cpp에서 cb()로 지정)
 * ---------------------------------------------------------- */
void ggml_dvfs_apply_if_needed(int op_id, const char *name)
{
#if defined(__gnu_linux__) || defined(__ANDROID__)
    // 1) 현재 stage / group 결정
    int st = atomic_load_explicit(&g_cur_stage, memory_order_relaxed);
    if (st < 0 || st >= GGML_STAGE_COUNT) {
        st = ST_DECODE;
    }

    perf_group g = ggml_node_to_group(name);
    atomic_store_explicit(&g_last_group, (int) g, memory_order_relaxed);

    // 2) 원하는 주파수 (per-op override → 없으면 group plan)
    int want_cpu = ggml_dvfs_get(op_id);
    int want_mem = ggml_memfreq_get(op_id);

    if (want_cpu <= 0) {
        want_cpu = g_group_dvfs_plan[st][g].cpu_khz;
    }
    if (want_mem <= 0) {
        want_mem = g_group_dvfs_plan[st][g].mem_khz;
    }

    int cur_cpu = atomic_load_explicit(&g_applied_khz,     memory_order_relaxed);
    int cur_mem = atomic_load_explicit(&g_applied_mem_khz, memory_order_relaxed);

    // 3) CPU freq 적용
    if (want_cpu > 0 && want_cpu != cur_cpu) {
        char buf[16];
        snprintf(buf, sizeof(buf), "%d", want_cpu);

        int before = get_cpu_freq();
        printf("Before: %d kHz\n", before);
        uint64_t t0 = now_ns();
        set_cpu_freq(buf);
        uint64_t t1 = now_ns();
        int immediately = get_cpu_freq();
        printf("Immediately after: %d kHz\n", immediately);

        atomic_store_explicit(&g_applied_khz, want_cpu, memory_order_relaxed);

        double us = (double)(t1 - t0) / 1000.0;
        // fprintf(stderr,
        //         "[dvfs] CPU st=%d grp=%d op=%d name=%s : %d -> %d kHz, overhead = %.3f us\n",
        //         st, (int)g, op_id, name ? name : "(null)",
        //         cur_cpu, want_cpu, us);
    }

    // 4) MEM freq 적용
    if (want_mem > 0 && want_mem != cur_mem) {
        char buf[16];
        snprintf(buf, sizeof(buf), "%d", want_mem);

        uint64_t t0 = now_ns();
        set_mem_freq(buf);
        uint64_t t1 = now_ns();

        atomic_store_explicit(&g_applied_mem_khz, want_mem, memory_order_relaxed);

        double us = (double)(t1 - t0) / 1000.0;
        // fprintf(stderr,
        //         "[dvfs] MEM st=%d grp=%d op=%d name=%s : %d -> %d kHz, overhead = %.3f us\n",
        //         st, (int)g, op_id, name ? name : "(null)",
        //         cur_mem, want_mem, us);
    }
#else
    (void)op_id;
    (void)name;
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
        if (f) {
            fprintf(f, "%s", freq_str);
            fclose(f);
        }
    }
#else
    (void)freq_str;
#endif
}

static void set_mem_freq(const char *freq_str)
{
#if defined(__gnu_linux__) || defined(__ANDROID__)
    static const char *mems[] = {
        "/sys/class/devfreq/17000010.devfreq_mif/max_freq",
    };
    for (size_t i = 0; i < sizeof(mems)/sizeof(mems[0]); ++i) {
        FILE *f = fopen(mems[i], "w");
        if (f) {
            fprintf(f, "%s", freq_str);
            fclose(f);
        }
    }
#else
    (void)freq_str;
#endif
}
