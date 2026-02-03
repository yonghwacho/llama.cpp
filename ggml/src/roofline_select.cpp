#include "roofline_select.h"
#include "ggml-dvfs.h"  // gid 타입 정의되어 있지만, 여기서는 안 써도 됨

void build_freq_candidates_for_group(int gid, FreqCandidates& out) {
    (void)gid; // gid 안 쓰는 경고 방지

    out.cpu.clear();
    out.mem.clear();

    // ★ 모든 그룹 공통으로 사용할 CPU freq 후보 (kHz)
    out.cpu = {
        700000,
        1164000,   
        1396000,
        1557000,  
        1745000,
        1885000, 
        1999000,
        2147000, 
        2363000,
        2294000,
        2363000,
        2499000,
        2687000
    };

    // ★ 모든 그룹 공통으로 사용할 MEM freq 후보 (kHz)
    out.mem = {
        421000,
        546000,
        676000,
        845000,
        1014000,
        1352000,
        1539000,
        1716000,
        2028000,
        2288000,
        2730000,
        3172000,
        3744000        
    };
}