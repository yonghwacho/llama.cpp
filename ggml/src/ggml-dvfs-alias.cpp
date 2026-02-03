// ggml-dvfs-alias.cpp  (ggml/src/ 밑에)
#include <cstring>

static char g_model_alias[64] = "default";

extern "C" void ggml_dvfs_set_model_alias(const char * alias) {
    if (!alias || !*alias) alias = "default";
    std::strncpy(g_model_alias, alias, sizeof(g_model_alias));
    g_model_alias[sizeof(g_model_alias) - 1] = '\0';
}

extern "C" const char * ggml_dvfs_get_model_alias(void) {
    return g_model_alias;
}