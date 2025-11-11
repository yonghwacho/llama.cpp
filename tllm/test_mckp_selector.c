#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "ggml-mckp-freq-test.h"

/* Helper: print solution result */
static void print_solution_c(const DPResultC* res, const GroupC* groups) {
    if (!res) {
        printf("[Error] Null result pointer.\n");
        return;
    }

    if (!res->feasible) {
        printf("[Infeasible] Even the fastest choices cannot meet the latency budget.\n");
        printf("  Fastest total latency: %.4f\n", res->totalLatency);
        printf("  Fastest total energy : %.4f\n", res->totalEnergy);
        return;
    }

    printf("== Selected (c,m) per group ==\n");
    for (size_t g = 0; g < res->n_groups; ++g) {
        int idx = res->selected_index_per_group[g];
        const GroupC* group = &groups[g];
        const ChoiceC* ch = &group->choices[idx];
        printf("  [%s] repeat=%d -> choice#%d (c=%d, m=%d, tag=%s) energy=%.3f latency=%.3f\n",
               group->name, group->repeat, idx, ch->c, ch->m,
               ch->tag ? ch->tag : "(null)",
               ch->energy, ch->latency);
    }

    printf("---------------------------------\n");
    printf("Total latency: %.4f\n", res->totalLatency);
    printf("Total energy : %.4f\n", res->totalEnergy);
    printf("Energy saved : %.4f\n", res->energySaved);
    printf("Slack used   : %.4f\n", res->slackUsed);
}

/* Main test entry */
int main(void) {
    /* Define toy groups and choices */
    ChoiceC qkv_choices[] = {
        {1.00, 1.00, 2200, 3200, "fast"},
        {0.94, 1.05, 2000, 3000, "opt1"},
        {0.90, 1.10, 1800, 2800, "opt2"},
        {0.85, 1.18, 1600, 2400, "slow"}
    };
    GroupC qkv = {"QKV", 24, qkv_choices, sizeof(qkv_choices)/sizeof(qkv_choices[0])};

    ChoiceC oproj_choices[] = {
        {0.80, 0.60, 2200, 3200, "fast"},
        {0.76, 0.64, 2000, 3000, "opt1"},
        {0.73, 0.70, 1800, 2800, "slow"}
    };
    GroupC oproj = {"O-Proj", 24, oproj_choices, sizeof(oproj_choices)/sizeof(oproj_choices[0])};

    ChoiceC fup_choices[] = {
        {1.40, 1.30, 2200, 3200, "fast"},
        {1.32, 1.36, 2000, 3000, "opt1"},
        {1.26, 1.46, 1800, 2800, "opt2"},
        {1.22, 1.58, 1600, 2400, "slow"}
    };
    GroupC fup = {"FFN-Up", 24, fup_choices, sizeof(fup_choices)/sizeof(fup_choices[0])};

    ChoiceC fdown_choices[] = {
        {0.90, 0.85, 2200, 3200, "fast"},
        {0.86, 0.90, 2000, 3000, "opt1"},
        {0.83, 0.98, 1800, 2800, "slow"}
    };
    GroupC fdown = {"FFN-Down", 24, fdown_choices, sizeof(fdown_choices)/sizeof(fdown_choices[0])};

    GroupC groups[] = {qkv, oproj, fup, fdown};
    size_t n_groups = sizeof(groups)/sizeof(groups[0]);

    /* Define test parameters */
    double T_budget = 24 * (1.00 + 0.60 + 1.30 + 0.85) * 1.05; // 5% slack
    double delta = 0.01;

    printf("==== DP Solver Test ====\n");
    DPResultC dp_res = freq_table_mckp_solver_c(groups, n_groups, T_budget, delta, true);
    print_solution_c(&dp_res, groups);
    free_dpresult_c(&dp_res);

    printf("\n==== Greedy Solver Test ====\n");
    DPResultC gr_res = freq_table_mckp_solver_c(groups, n_groups, T_budget, delta, false);
    print_solution_c(&gr_res, groups);
    free_dpresult_c(&gr_res);

    return 0;
}
