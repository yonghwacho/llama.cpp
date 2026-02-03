// tllm_dvfs.cpp (simple)
#include "tllm_dvfs.h"
#include <string>
#include <cctype>
#include <cstring>
#include "log.h"

static std::string g_selected_pack = "default";
static bool        g_pack_frozen   = false;

static std::string to_lower(std::string s) {
    for (auto &ch : s) ch = (char) std::tolower((unsigned char) ch);
    return s;
}

static std::string resolve_auto_pack(const char * model_path) {
    std::string p = to_lower(model_path ? model_path : "");

    if (p.find("qwen2.5-1.5b") != std::string::npos) return "qwen2.5-1.5b";
    if (p.find("llama-3.2-1b") != std::string::npos) return "llama3.2-1b";
    if (p.find("llama-3.2-3b") != std::string::npos) return "llama3.2-3b";
    return "default";
}

static bool is_known_pack(const std::string & id) {
    return id == "default" ||
           id == "qwen2.5-1.5b" ||
           id == "llama3.2-1b" ||
           id == "llama3.2-3b";
}

void tllm_dvfs_select_model_pack(const char * pack_id, llama_model * /*model*/, const char * model_path) {
    if (g_pack_frozen) {
        LOG_WRN("DVFS: model pack already selected ('%s'); ignoring new request.\n",
                g_selected_pack.c_str());
        return;
    }

    std::string id = pack_id ? pack_id : "";
    if (id.empty()) id = "auto";

    if (id == "auto") {
        id = resolve_auto_pack(model_path);
        LOG_INF("DVFS: auto-resolved model pack = '%s' (model_path='%s')\n",
                id.c_str(), model_path ? model_path : "(null)");
    } else {
        LOG_INF("DVFS: requested model pack = '%s'\n", id.c_str());
    }

    if (!is_known_pack(id)) {
        LOG_WRN("DVFS: unknown model pack '%s' -> fallback to 'default'\n", id.c_str());
        id = "default";
    }

    g_selected_pack = id;
    g_pack_frozen   = true;
    LOG_INF("DVFS: selected model pack = '%s'\n", g_selected_pack.c_str());
}

const char * tllm_dvfs_get_selected_pack_id() {
    return g_selected_pack.c_str();
}