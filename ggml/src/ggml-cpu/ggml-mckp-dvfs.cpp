/*
wjbang, 10.27
MCKP-based DVFS controller Implementation
*/

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include "ggml-mckp-freq.h"
#include <cstdlib>
#include <string>

using namespace std;

/* Define structures for MCKP algorithms */
struct Choice {
    double energy;
    double latency;
    int c = -1; // core freq
    int m = -1; // memory freq
    string tag;
    int original_idx = -1;
};

struct Group {
    string name;
    int repeat = 1;
    vector<Choice> choices;
};

struct DPResult {
    bool feasible = false;
    vector<int> pickIndexPerGroup; 
    double totalLatency = 0.0;
    double totalEnergy = 0.0;
    double energySaved = 0.0;
    double slackUsed = 0.0;
};

/* Utility functions */
static inline bool dominated(const Choice& a, const Choice& b) {
    return (b.latency <= a.latency + 1e-12 && b.energy <= a.energy + 1e-12) &&
           (b.latency < a.latency - 1e-12 || b.energy < a.energy - 1e-12);
}

// Eliminate points which are high latency and high energy
void pareto_prune(Group& g) {
    auto &v = g.choices;
    if (v.empty()) return;

    printf("[DEBUG pareto] BEFORE: size=%zu\n", v.size());
    for (size_t i = 0; i < std::min(v.size(), (size_t)5); ++i) {
        printf("  [%zu] orig=%d, c=%d, m=%d, lat=%.3f\n",
               i, v[i].original_idx, v[i].c, v[i].m, v[i].latency);
    }

    sort(v.begin(), v.end(), [](const Choice& A, const Choice& B){
        if (A.latency != B.latency) return A.latency < B.latency;
        return A.energy < B.energy;
    });
    // Remove dominated points
    // After sorting by latency ascending, we keep points with strictly decreasing energy
    // This forms the Pareto frontier: no point is dominated by another
    vector<Choice> keep;
    keep.push_back(v[0]); // Always keep the fastest point
    
    for (size_t i = 1; i < v.size(); ++i) {
        // Only keep v[i] if it has strictly lower energy than the last kept point
        // Since latency is increasing (sorted), we need energy to decrease to be non-dominated
        if (v[i].energy < keep.back().energy - 1e-12) {
            keep.push_back(v[i]);
        }
        // Otherwise v[i] is dominated: it has higher latency and equal/higher energy
    }
    v.swap(keep);

    // 🔍 After pruning
    // printf("[DEBUG pareto] AFTER: size=%zu\n", v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        // printf("  [%zu] orig=%d, c=%d, m=%d, lat=%.3f\n",
        //        i, v[i].original_idx, v[i].c, v[i].m, v[i].latency);
    }
}

/* DP-Based Algorithm Implementation */
DPResult solve_mckp_dp(vector<Group>& groups,
                    double T_budget,
                    double time_unit) // E-L measure unit
{
    const double INF_NEG = -1e300;

    int G = (int)groups.size();
    for (auto &g : groups) pareto_prune(g);
    // ---- DEBUG: after pareto prune ----
    // printf("==== Pareto prune result ====\n");
    for (int g = 0; g < (int)groups.size(); ++g) {
        // printf("[Group %d] name=%s, repeat=%d, choices=%zu\n",
        //     g, groups[g].name.c_str(), groups[g].repeat, groups[g].choices.size());
        for (size_t j = 0; j < groups[g].choices.size(); ++j) {
            const auto &c = groups[g].choices[j];
            // printf("    idx=%zu  lat=%.3f ms  E=%.6f J  cpu=%d mem=%d\n",
            //     j, c.latency, c.energy, c.c, c.m);
        }
    }
    // printf("================================\n");

    // Build baseline (fastest option per group = smallest latency)
    vector<int> baseIdx(G, -1);
    double T_min = 0.0, E_min = 0.0;
    for (int g=0; g<G; ++g) {
        if (groups[g].choices.empty()) {
            return {}; // infeasible (no option)
        }
        // choices are latency-asc after prune
        baseIdx[g] = 0;
        T_min += groups[g].repeat * groups[g].choices[0].latency;
        E_min += groups[g].repeat * groups[g].choices[0].energy;
    }

    // ---- DEBUG: baseline check ----
    printf("==== Baseline summary (fastest choices) ====\n");
    printf("T_min = %.3f ms, E_min = %.6f J, T_budget = %.3f ms\n",
        T_min, E_min, T_budget);

    for (int g = 0; g < G; ++g) {
        const auto &c = groups[g].choices[0];
        printf("[Group %d baseline] lat=%.3f ms  E=%.6f J  cpu=%d mem=%d (repeat=%d)\n",
            g, c.latency, c.energy, c.c, c.m, groups[g].repeat);
    }
    printf("=============================================\n");

    // Terminate if baseline exceeds budget
    if (T_min - 1e-12 > T_budget) {
        DPResult res; res.feasible = false; res.totalLatency = T_min; res.totalEnergy = E_min;
        return res;
    }

    double slack = max(0.0, T_budget - T_min);
    // Discretize capacity
    long long W = (long long) floor(slack / time_unit + 1e-9);

    // Precompute per-group weights (extra latency) and values (energy saved)
    vector<vector<long long>> w(G);
    vector<vector<double>>    v(G);
    for (int g=0; g<G; ++g) {
        const auto &C = groups[g].choices;
        w[g].reserve(C.size());
        v[g].reserve(C.size());
        double baseT = groups[g].repeat * C[0].latency;
        double baseE = groups[g].repeat * C[0].energy;
        for (size_t j=0; j<C.size(); ++j) {
            double dT = groups[g].repeat * C[j].latency - baseT;
            double dE = baseE - groups[g].repeat * C[j].energy; // energy saved vs fastest
            long long ww = (long long) floor(max(0.0, dT) / time_unit + 1e-9);
            w[g].push_back(ww);
            v[g].push_back(max(0.0, dE));
        }
    }

    // dp[g][w] = best energy saving using first g groups under extra latency w
    vector<vector<double>> dp(G+1, vector<double>(W+1, INF_NEG));
    vector<vector<int>>    take(G+1, vector<int>(W+1, -1)); // which choice index used at (g,w)
    dp[0][0] = 0.0;

    for (int g=1; g<=G; ++g) {
        for (long long curW=0; curW<=W; ++curW) {
            double best = INF_NEG; int bestJ = -1;
            // try all choices of group g-1
            for (size_t j=0; j<w[g-1].size(); ++j) {
                long long ww = w[g-1][j];
                if (ww <= curW && dp[g-1][curW - ww] > INF_NEG/2) {
                    double cand = dp[g-1][curW - ww] + v[g-1][j];
                    if (cand > best) { best = cand; bestJ = (int)j; }
                }
            }
            dp[g][curW]  = best;
            take[g][curW]= bestJ;
        }
    }

    // pick best w <= W
    long long bestW = 0;
    double bestVal = INF_NEG;
    for (long long curW=0; curW<=W; ++curW) {
        if (dp[G][curW] > bestVal) { bestVal = dp[G][curW]; bestW = curW; }
    }

    DPResult res;
    if (bestVal <= INF_NEG/2) {
        // Shouldn't happen because baseline (w=0) is always feasible
        res.feasible = false; res.totalLatency = T_min; res.totalEnergy = E_min; return res;
    }

    // Reconstruct picks
    res.pickIndexPerGroup.assign(G, 0);
    long long curW = bestW;
    for (int g=G; g>=1; --g) {
        int j = take[g][curW];
        if (j < 0) j = 0; // safety
        res.pickIndexPerGroup[g-1] = j;
        curW -= w[g-1][j];
    }

    // Compute totals
    double T_used = T_min + bestW * time_unit;
    double E_saved = bestVal;
    double E_used = E_min - E_saved;

    // ---- DEBUG: DP result ----
    printf("==== DP Result ====\n");
    printf("feasible = %d\n", res.feasible);
    printf("totalLatency = %.3f ms (budget=%.3f ms)\n", T_used, T_budget);
    printf("totalEnergy  = %.6f J\n", E_used);
    printf("energySaved  = %.6f J\n", E_saved);
    printf("slackUsed    = %.3f ms\n", bestW * time_unit);

    for (int g = 0; g < G; ++g) {
        int idx = res.pickIndexPerGroup[g];
        const auto &c = groups[g].choices[idx];
        printf("[Selected] Group %d -> choice %d : lat=%.3f ms, E=%.6f J, cpu=%d, mem=%d\n",
            g, idx, c.latency, c.energy, c.c, c.m);
    }
    printf("====================\n");

    res.feasible = true;
    res.totalLatency = T_used;
    res.totalEnergy = E_used;
    res.energySaved = E_saved;
    res.slackUsed = bestW * time_unit;
    return res;
}

/* Greedy-Based Algorithm Implementation, NeuroBalancer style */
DPResult solve_mckp_greedy(vector<Group>& groups, double T_budget) {
    int G = (int)groups.size();
    for (auto &g : groups) pareto_prune(g);

    vector<int> cur(G, 0); // current index per group (start at fastest)
    double T = 0.0, E = 0.0;
    for (int g=0; g<G; ++g) {
        if (groups[g].choices.empty()) return {};
        T += groups[g].repeat * groups[g].choices[0].latency;
        E += groups[g].repeat * groups[g].choices[0].energy;
    }

    double max_E = E;
    double min_T = T;

    if (T > T_budget + 1e-12) {
        DPResult r; r.feasible=false; r.totalLatency=T; r.totalEnergy=E; return r;
    }

    struct Cand { int g; int nextIdx; double dT; double dE; double rho; };
    auto makeCand = [&](int g, int idx)->optional<Cand>{
        const auto &C = groups[g].choices;
        if (idx <= 0 || idx >= (int)C.size()) return nullopt;
        double dT = (C[idx].latency - C[idx-1].latency) * groups[g].repeat;
        double dE = (C[idx-1].energy - C[idx].energy) * groups[g].repeat;
        if (dT <= 0 || dE < 0) return nullopt; // ignore weirdness
        Cand c{g, idx, dT, dE, dE / dT};
        return c;
    };

    vector<optional<Cand>> cands(G);
    for (int g=0; g<G; ++g) cands[g] = makeCand(g, 1);

    auto pickBestFitting = [&](double avail)->int{
        // return group index to take next increment; -1 if none fits
        int bestg = -1; double bestRho = -1.0;
        for (int g=0; g<G; ++g) if (cands[g]) {
            if (cands[g]->dT <= avail + 1e-12) {
                if (cands[g]->rho > bestRho) { bestRho = cands[g]->rho; bestg = g; }
            }
        }
        return bestg;
    };

    // Greedily consume slack
    double avail = T_budget - T;
    while (true) {
        int g = pickBestFitting(avail);
        if (g < 0) break; // no increment fits
        // apply
        T += cands[g]->dT;
        E -= cands[g]->dE;
        cur[g] = cands[g]->nextIdx;
        // push next increment for this group
        cands[g] = makeCand(g, cur[g] + 1);
        avail = T_budget - T;
        if (avail <= 1e-12) break;
    }

    DPResult r;
    r.feasible = true;
    r.pickIndexPerGroup = cur;
    r.totalLatency = T;
    r.totalEnergy = E;
    r.energySaved = max_E - E; // not tracked here
    r.slackUsed = T - min_T; // not baseline-based here; kept simple
    return r;
}

// C API wrapper
DPResultC freq_table_mckp_solver_c(const GroupC* groups,
                                   size_t n_groups,
                                   double T_budget,
                                   double time_unit,
                                   bool use_dp) {
    std::vector<Group> gin;
    gin.reserve(n_groups);
    for (size_t i = 0; i < n_groups; ++i) {
        Group g;
        g.name = groups[i].name ? std::string(groups[i].name) : std::string();
        g.repeat = groups[i].repeat;
        g.choices.reserve(groups[i].n_choices);
        for (size_t j = 0; j < groups[i].n_choices; ++j) {
            const ChoiceC& cc = groups[i].choices[j];
            Choice cxx;
            cxx.energy = cc.energy;
            cxx.latency = cc.latency;
            cxx.c = cc.c;
            cxx.m = cc.m;
            cxx.tag = cc.tag ? std::string(cc.tag) : std::string();
            cxx.original_idx = (int)j;  // 추가: 원본 인덱스 저장
            g.choices.push_back(std::move(cxx));
        }
        gin.push_back(std::move(g));
    }

    DPResult core;
    if (use_dp) {
        core = solve_mckp_dp(gin, T_budget, time_unit);
    } else {
        core = solve_mckp_greedy(gin, T_budget);
    }

    DPResultC out;
    out.feasible = core.feasible;
    out.n_groups = core.pickIndexPerGroup.size();
    out.selected_index_per_group = nullptr;
    if (out.n_groups > 0) {
        out.selected_index_per_group = static_cast<int*>(std::malloc(sizeof(int) * out.n_groups));
        if (out.selected_index_per_group) {
            for (size_t k = 0; k < out.n_groups; ++k) {
                int pruned_idx = core.pickIndexPerGroup[k];

                // printf("[DEBUG convert] Group %zu: pruned_idx=%d\n", k, pruned_idx);
                // printf("[DEBUG convert] gin[%zu].choices[%d]: c=%d, m=%d\n", k, pruned_idx, gin[k].choices[pruned_idx].c, gin[k].choices[pruned_idx].m);
                
                // 🔥 추가: Pruned index → Original index 변환
                int original_idx = gin[k].choices[pruned_idx].original_idx;

                // printf("[DEBUG convert] gin[%zu].choices[%d].original_idx = %d\n", k, pruned_idx, original_idx);
                if (original_idx < 0) {
                    original_idx = pruned_idx;  // fallback (혹시 모를 경우)
                }
                
                out.selected_index_per_group[k] = original_idx;
                // printf("[DEBUG convert] >>> Returning original_idx=%d to caller <<<\n", original_idx);
            }
        }
    }
    out.totalLatency = core.totalLatency;
    out.totalEnergy = core.totalEnergy;
    out.energySaved = core.energySaved;
    out.slackUsed = core.slackUsed;
    return out;
}

void free_dpresult_c(DPResultC* res) {
    if (!res) return;
    if (res->selected_index_per_group) {
        std::free(res->selected_index_per_group);
        res->selected_index_per_group = nullptr;
    }
    res->n_groups = 0;
}


