#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
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

/* Test runner function */
static void run_validation_test(void) {
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
    size_t n_groups = sizeof(groups) / sizeof(groups[0]);

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
}

static void run_overhead_profile_test(int n_layers, int num_points_per_group, int num_groups, int n_trials, 
    const char* logfile_path, int verbose) {
    /* Analyze the overhead incurred from the MCKP solver */
    
    printf("\n==== Overhead Profiling Test ====\n");
    printf("Config: n_layers=%d, points_per_group=%d, num_groups=%d, trials=%d\n\n",
           n_layers, num_points_per_group, num_groups, n_trials);
    
    /* Open logfile if provided */
    FILE* logfile = NULL;
    if (logfile_path != NULL) {
        logfile = fopen(logfile_path, "a");
        if (!logfile) {
            printf("[Warning] Could not open logfile: %s\n", logfile_path);
        }
    }

    /* Allocate memory for groups and choices */
    GroupC* groups = (GroupC*)malloc(num_groups * sizeof(GroupC));
    ChoiceC** all_choices = (ChoiceC**)malloc(num_groups * sizeof(ChoiceC*));
    
    if (!groups || !all_choices) {
        printf("[Error] Memory allocation failed.\n");
        free(groups);
        free(all_choices);
        return;
    }

    /* Generate synthetic test data */
    double baseline_latency = 0.0;
    for (int g = 0; g < num_groups; g++) {
        all_choices[g] = (ChoiceC*)malloc(num_points_per_group * sizeof(ChoiceC));
        if (!all_choices[g]) {
            printf("[Error] Memory allocation failed for choices.\n");
            for (int i = 0; i < g; i++) free(all_choices[i]);
            free(all_choices);
            free(groups);
            return;
        }

        /* Create choices with varying energy/latency tradeoffs */
        for (int i = 0; i < num_points_per_group; i++) {
            double ratio = (double)i / (num_points_per_group - 1);
            all_choices[g][i].energy = 1.0 - ratio * 0.3;  // Energy: 1.0 -> 0.7
            all_choices[g][i].latency = 1.0 + ratio * 0.5; // Latency: 1.0 -> 1.5
            all_choices[g][i].c = 2200 - i * 100;
            all_choices[g][i].m = 3200 - i * 100;
            all_choices[g][i].tag = (i == 0) ? "fast" : (i == num_points_per_group - 1) ? "slow" : "mid";
        }

        /* Setup group */
        static char group_names[32][32];
        snprintf(group_names[g], 32, "Group%d", g);
        groups[g].name = group_names[g];
        groups[g].repeat = n_layers;
        groups[g].choices = all_choices[g];
        groups[g].n_choices = num_points_per_group;

        /* Track baseline (fastest choice) */
        baseline_latency += all_choices[g][0].latency * n_layers;
    }

    double T_budget = baseline_latency * 1.10; // 10% slack
    double delta = 0.01;

    /* Benchmark DP solver */
    printf("Testing DP solver...\n");
    clock_t dp_start = clock();
    for (int trial = 0; trial < n_trials; trial++) {
        DPResultC dp_res = freq_table_mckp_solver_c(groups, num_groups, T_budget, delta, true);
        free_dpresult_c(&dp_res);
    }
    clock_t dp_end = clock();
    double dp_time_ms = ((double)(dp_end - dp_start) / CLOCKS_PER_SEC) * 1000.0 / n_trials;

    /* Benchmark Greedy solver */
    printf("Testing Greedy solver...\n");
    clock_t greedy_start = clock();
    for (int trial = 0; trial < n_trials; trial++) {
        DPResultC greedy_res = freq_table_mckp_solver_c(groups, num_groups, T_budget, delta, false);
        free_dpresult_c(&greedy_res);
    }
    clock_t greedy_end = clock();
    double greedy_time_ms = ((double)(greedy_end - greedy_start) / CLOCKS_PER_SEC) * 1000.0 / n_trials;

    /* Print results */
    if (verbose) {
        printf("\n----- Profiling Results -----\n");
        printf("DP Solver:     %.3f ms per solve\n", dp_time_ms);
        printf("Greedy Solver: %.3f ms per solve\n", greedy_time_ms);
        printf("Speedup:       %.2fx\n", dp_time_ms / greedy_time_ms);
        printf("-----------------------------\n");
    }

    /* Write to logfile if provided */
    if (logfile != NULL) {
        time_t now = time(NULL);
        char timestamp[64];
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
        
        fprintf(logfile, "\n=== Overhead Profiling Test - %s ===\n", timestamp);
        fprintf(logfile, "Configuration:\n");
        fprintf(logfile, "  n_layers: %d\n", n_layers);
        fprintf(logfile, "  points_per_group: %d\n", num_points_per_group);
        fprintf(logfile, "  num_groups: %d\n", num_groups);
        fprintf(logfile, "  n_trials: %d\n", n_trials);
        fprintf(logfile, "  T_budget: %.4f\n", T_budget);
        fprintf(logfile, "  delta: %.4f\n", delta);
        fprintf(logfile, "  baseline_latency: %.4f\n", baseline_latency);
        fprintf(logfile, "\nProfiling Results:\n");
        fprintf(logfile, "  DP Solver:     %.3f ms per solve\n", dp_time_ms);
        fprintf(logfile, "  Greedy Solver: %.3f ms per solve\n", greedy_time_ms);
        fprintf(logfile, "  Speedup:       %.2fx\n", dp_time_ms / greedy_time_ms);
        fprintf(logfile, "==========================================\n");
        
        fclose(logfile);
        printf("[Info] Results logged to: %s\n", logfile_path);
    }

    /* Cleanup */
    for (int g = 0; g < num_groups; g++) {
        free(all_choices[g]);
    }
    free(all_choices);
    free(groups);
}

/* Main test entry */
int main(void) {
    // run_validation_test();
    
    /* Run overhead profiling with different configurations */
    run_overhead_profile_test(16, 5, 2, 100, "profiling_results.log", 0);    // Small test: 16 layers, 5 points, 2 groups, 100 trials
    run_overhead_profile_test(16, 10, 2, 100, "profiling_results.log", 0);    // Small test: 16 layers, 10 points, 2 groups, 100 trials
    run_overhead_profile_test(16, 15, 2, 100, "profiling_results.log", 0);    // Small test: 16 layers, 15 points, 2 groups, 100 trials
    run_overhead_profile_test(16, 20, 2, 100, "profiling_results.log", 0);    // Small test: 16 layers, 20 points, 2 groups, 100 trials
    run_overhead_profile_test(16, 25, 2, 100, "profiling_results.log", 0);    // Small test: 16 layers, 25 points, 2 groups, 100 trials
    run_overhead_profile_test(16, 30, 2, 100, "profiling_results.log", 0);    // Small test: 16 layers, 30 points, 2 groups, 100 trials
    run_overhead_profile_test(16, 35, 2, 100, "profiling_results.log", 0);    // Small test: 16 layers, 35 points, 2 groups, 100 trials

    run_overhead_profile_test(28, 5, 2, 100, "profiling_results.log", 0);    // Medium test: 28 layers, 5 points, 2 groups, 100 trials
    run_overhead_profile_test(28, 10, 2, 100, "profiling_results.log", 0);    // Medium test: 28 layers, 10 points, 2 groups, 100 trials
    run_overhead_profile_test(28, 15, 2, 100, "profiling_results.log", 0);    // Medium test: 28 layers, 15 points, 2 groups, 100 trials
    run_overhead_profile_test(28, 20, 2, 100, "profiling_results.log", 0);    // Medium test: 28 layers, 20 points, 2 groups, 100 trials
    run_overhead_profile_test(28, 25, 2, 100, "profiling_results.log", 0);    // Medium test: 28 layers, 25 points, 2 groups, 100 trials
    run_overhead_profile_test(28, 30, 2, 100, "profiling_results.log", 0);    // Medium test: 28 layers, 30 points, 2 groups, 100 trials
    run_overhead_profile_test(28, 35, 2, 100, "profiling_results.log", 0);    // Medium test: 28 layers, 35 points, 2 groups, 100 trials
    
    return 0;
}

       
