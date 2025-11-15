#include "roofline_select.h"
#include "ggml-dvfs.h"  // gid 타입 정의되어 있지만, 여기서는 안 써도 됨

void build_freq_candidates_for_group(int gid, FreqCandidates& out) {
    (void)gid; // gid 안 쓰는 경고 방지

    out.cpu.clear();
    out.mem.clear();

    // ★ 모든 그룹 공통으로 사용할 CPU freq 후보 (kHz)
    out.cpu = {
        700000,   
        1396000,  
        1745000, 
        1999000, 
        2363000,
        2687000
    };

    // ★ 모든 그룹 공통으로 사용할 MEM freq 후보 (kHz)
    out.mem = {
        421000,
        845000,
        1352000,
        2028000,
        2730000,
        3744000        
    };
}