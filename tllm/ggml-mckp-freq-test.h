/*
wjbang, 10.27
MCKP-based DVFS controller Implementation
*/

#pragma once
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdbool.h>

typedef struct {
    double energy;
    double latency;
    int c;
    int m;
    const char* tag;
} ChoiceC;

typedef struct {
    const char* name;
    int repeat;
    const ChoiceC* choices;
    size_t n_choices;
} GroupC;

typedef struct {
    bool feasible;
    int* selected_index_per_group;
    size_t n_groups;
    double totalLatency;
    double totalEnergy;
    double energySaved;
    double slackUsed;
} DPResultC;

DPResultC freq_table_mckp_solver_c(const GroupC* groups,
                                   size_t n_groups,
                                   double T_budget,
                                   double time_unit,
                                   bool use_dp);

void free_dpresult_c(DPResultC* res);

#ifdef __cplusplus
}
#endif