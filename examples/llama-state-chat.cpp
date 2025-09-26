// examples/llama-state-chat.cpp
#include "arg.h"
#include "common.h"
#include "llama.h"
#include <cstring> 
#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

namespace {
constexpr size_t kArenaSize = 524288; // 512 KB 임시 문자열 버퍼
char * g_arena = nullptr;
size_t g_arena_off = 0;

char * copy_from_arena(const std::string & s) {
    if (!g_arena) {
        g_arena = (char *) std::malloc(kArenaSize);
        g_arena_off = 0;
    }
    const size_t need = s.size() + 1;
    if (g_arena_off + need > kArenaSize) {
        std::fprintf(stderr, "Arena overflow (need %zu)\n", need);
        std::exit(1);
    }
    char * dest = g_arena + g_arena_off;
    std::memcpy(dest, s.data(), s.size());
    dest[s.size()] = '\0';
    g_arena_off += need;
    return dest;
}
} // namespace

int main(int argc, char ** argv) {
    // 1) 공통 파라미터
    common_params params;
    params.sampling.seed = 1234;
    params.n_ctx         = 8300;
    params.n_predict     = 32;
    params.n_batch       = 8300;
    params.n_ubatch      = 16;
    params.warmup        = true;

    llama_log_set([](enum ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_DEBUG) {
            std::fputs(text, stderr);
        }
    }, nullptr);

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();

    // 2) 모델/컨텍스트 생성
    auto init_res = common_init_from_params(params);
    llama_model * model = init_res.model.get();
    llama_context * ctx = init_res.context.get();
    if (!model || !ctx) {
        std::fprintf(stderr, "ERROR: model/context initialization failed\n");
        return 1;
    }

    // 3) 템플릿 & vocab & 샘플러
    const char * tmpl = llama_model_chat_template(model, nullptr);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_sampler * sampler = llama_sampler_init_greedy();

    std::string state_name = "state.bin";

    // 4) 세션 토큰
    std::vector<llama_token> session_tokens;
    session_tokens.reserve(params.n_ctx);

    int    n_past = 0;
    size_t n_token_count = 0;

    // 5) state.bin 있으면 로드
    if (std::filesystem::exists(state_name)) {
        std::vector<llama_token> tmp(params.n_ctx);

        const auto t0 = std::chrono::high_resolution_clock::now();
        if (!llama_state_load_file(ctx, state_name.c_str(),
                                   tmp.data(), tmp.size(), &n_token_count)) {
            std::cerr << "ERROR: failed to load " << state_name << "\n";
            return 1;
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("[Timing] Loaded %s in %.2f ms\n", state_name.c_str(), ms);

        session_tokens.assign(tmp.begin(), tmp.begin() + n_token_count);
        n_past = static_cast<int>(n_token_count);
        std::cout << "[Info] Loaded " << state_name << " with " << n_token_count << " tokens\n";
    } else {
        std::cerr << "ERROR: No state file found at " << state_name << "\n"; 
        std::cerr << "Please create state file first.\n";
        return 1;
    }

    // 6) 대화 루프
    std::string user_input;
    std::vector<char> buf(1 << 20); // 템플릿 결과 버퍼(1MB)

    while (true) {
        std::cout << "[User]> ";
        if (!std::getline(std::cin, user_input)) break;
        if (user_input == "exit") break;

        if (user_input == "ss") {
            if (!llama_state_save_file(ctx, state_name.c_str(),
                                       session_tokens.data(), (uint32_t) session_tokens.size())) {
                std::cerr << "ERROR: failed to save session to " << state_name << "\n";
            } else {
                std::cout << "[Info] Session saved: " << session_tokens.size() << " tokens\n";
            }
            continue;
        }

        // 6-1) Chat 템플릿 적용 (follow-up 메시지 1개)
        g_arena_off = 0;
        llama_chat_message umsg{ "user", copy_from_arena(user_input) };

        // 템플릿이 있으면 적용, 없으면 fallback
        std::string prompt;
        if (tmpl && tmpl[0] != '\0') {
            int32_t nbytes = llama_chat_apply_template(
                tmpl, &umsg, 1,
                /*add_assistant=*/true,
                buf.data(), (int32_t) buf.size()
            );
            if (nbytes < 0) {
                std::cerr << "ERROR: chat template apply failed\n";
                return 1;
            }
            prompt.assign(buf.data(), (size_t) nbytes);
        } else {
            prompt = user_input;
        }

        // 6-2) 토크나이즈
        int n_tok = -llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(),
                                    nullptr, 0, /*bos*/true, /*special*/true);
        std::vector<llama_token> toks(n_tok);
        llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(),
                       toks.data(), (int) toks.size(), /*bos*/true, /*special*/true);

        // 6-3) batch 만들고 디코드
        llama_batch b = llama_batch_init((int) toks.size(), 0, 1);
        int32_t pos = n_past;

        for (llama_token t : toks) {
            common_batch_add(b, t, pos++, {0}, false);
        }

        // 컨텍스트 초과 방지
        const int n_ctx = llama_n_ctx(ctx);
        const int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
        if (n_ctx_used + b.n_tokens > n_ctx) {
            std::fprintf(stderr, "ERROR: context size exceeded (%d + %d > %d)\n",
                         n_ctx_used, b.n_tokens, n_ctx);
            llama_batch_free(b);
            break;
        }

        b.logits[b.n_tokens - 1] = true;
        if (llama_decode(ctx, b) != 0) {
            std::fprintf(stderr, "ERROR: llama_decode failed on prompt\n");
            llama_batch_free(b);
            break;
        }

        n_past += b.n_tokens;
        session_tokens.insert(session_tokens.end(), toks.begin(), toks.end());
        llama_batch_free(b);

        // 6-4) 첫 토큰 샘플/출력
        {
            const llama_token tok = llama_sampler_sample(sampler, ctx, -1);
            std::cout << common_token_to_piece(ctx, tok) << std::flush;

            llama_batch b1 = llama_batch_init(1, 0, 1);
            common_batch_add(b1, tok, n_past, {0}, true);

            if (llama_decode(ctx, b1) != 0) {
                std::fprintf(stderr, "\nERROR: llama_decode failed on first token\n");
                llama_batch_free(b1);
                break;
            }
            n_past += 1;
            session_tokens.push_back(tok);
            llama_batch_free(b1);
        }

        // (원하면) logits 일부 확인
        {
            const float * logits = llama_get_logits(ctx);
            for (int i = 0; i < 10; ++i) {
                std::printf("\nlogits[%d] = %f", i, logits[i]);
            }
            std::printf("\n");
        }

        // 6-5) n_predict 만큼 토큰 생성
        for (int i = 1; i < params.n_predict; ++i) {
            const llama_token tok = llama_sampler_sample(sampler, ctx, -1);
            if (tok == llama_vocab_eos(vocab)) {
                std::cout << std::endl;
                break;
            }
            std::cout << common_token_to_piece(ctx, tok) << std::flush;

            llama_batch b2 = llama_batch_init(1, 0, 1);
            common_batch_add(b2, tok, n_past, {0}, true);

            if (llama_decode(ctx, b2) != 0) {
                std::fprintf(stderr, "\nERROR: llama_decode failed on gen\n");
                llama_batch_free(b2);
                break;
            }
            n_past += 1;
            session_tokens.push_back(tok);
            llama_batch_free(b2);
        }
        std::cout << std::endl;
    }

    // 7) 정리
    llama_sampler_free(sampler);
    if (g_arena) std::free(g_arena);

    return 0;
}