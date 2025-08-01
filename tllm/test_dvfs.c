// test_dvfs.c
#include "ggml-dvfs.h"
#include <stdio.h>
#include <unistd.h>

int main() {
    // 0번 연산에 800 MHz, 1번 연산에 1 GHz, 2번 연산에 1.2 GHz 설정
    ggml_dvfs_set(2, 1396000);
    ggml_dvfs_set(6, 1557000);

    int ops[] = {2, 6};
    int num_ops = sizeof(ops) / sizeof(ops[0]);

    // 내부적으로 잘 저장됐는지 확인

    for (int i = 0; i < num_ops; ++i) {
        int op = ops[i];
        printf("g_op_freq_table[%d] = %d kHz\n", op, ggml_dvfs_get(op));
    }

    // sysfs에 실제로 적용
    for (int i = 0; i < num_ops; ++i) {
        int op = ops[i];
        ggml_dvfs_apply_if_needed(op);
        sleep(1);
    }

    return 0;
}
