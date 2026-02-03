#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <ctime>
#include <chrono>

// Structure to hold generation results
struct GenerationResult {
    std::vector<float> time_per_token;  // Time per token in milliseconds
    std::vector<double> token_start_time; // Token generation start time in seconds since epoch
    std::string output_text;            // Complete generated text
    int n_tokens_generated;             // Number of tokens generated
    float total_time_ms;                // Total generation time in milliseconds
    int error_code;                     // 0 = success, non-zero = error
};

extern "C" {
    // Main generation function that runs the complete decode loop
    GenerationResult* generate_text(
        const char* model_path,
        const char* prompt_text,
        int n_predict,
        bool use_instruct,
        int n_threads,
        bool enable_flash_attn
    ) {
        auto result = new GenerationResult();
        result->error_code = 0;
        result->n_tokens_generated = 0;
        result->total_time_ms = 0.0f;
        
        std::string prompt = prompt_text;
        
        // Apply instruct template if requested
        if (use_instruct) {
            std::time_t t = std::time(nullptr);
            std::tm* tm_info = std::localtime(&t);
            const char* months[12] = {"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"};
            int day = tm_info ? tm_info->tm_mday : 1;
            int year = tm_info ? (tm_info->tm_year + 1900) : 1970;
            const char* mon = tm_info ? months[tm_info->tm_mon % 12] : months[0];
            std::string today = std::to_string(day) + " " + mon + " " + std::to_string(year);

            const char* system_prompt = "You are a helpful LLM assistant. Answer the question.";

            std::string templ;
            templ.reserve(prompt.size() + 256);
            templ += "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n";
            templ += "Cutting Knowledge Date: December 2023\n";
            templ += "Today Date: ";
            templ += today;
            templ += "\n\n";
            templ += system_prompt;
            templ += "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n";
            templ += prompt;
            templ += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n";

            prompt = templ;
        }
        
        // Load dynamic backends
        ggml_backend_load_all();
        
        // Initialize the model
        llama_model_params model_params = llama_model_default_params();
        // CPU-only build: no GPU layers
        model_params.n_gpu_layers = 0;
        
        llama_model* model = llama_model_load_from_file(model_path, model_params);
        if (model == NULL) {
            fprintf(stderr, "%s: error: unable to load model\n", __func__);
            result->error_code = 1;
            return result;
        }
        
        const llama_vocab* vocab = llama_model_get_vocab(model);
        
        // Tokenize the prompt
        const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
        std::vector<llama_token> prompt_tokens(n_prompt);
        if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
            fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
            result->error_code = 2;
            llama_model_free(model);
            return result;
        }
        
        // Initialize the context with custom parameters
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_prompt + n_predict;
        ctx_params.n_batch = n_prompt;
        ctx_params.no_perf = false;
        ctx_params.n_threads = n_threads;
        ctx_params.flash_attn_type = enable_flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        
        llama_context* ctx = llama_init_from_model(model, ctx_params);
        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
            result->error_code = 3;
            llama_model_free(model);
            return result;
        }
        
        // Initialize the sampler
        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        llama_sampler* smpl = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        
        // Store the output text
        std::string output_text;
        
        // Prepare initial batch
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        
        // Handle encoder-decoder models
        if (llama_model_has_encoder(model)) {
            if (llama_encode(ctx, batch)) {
                fprintf(stderr, "%s: failed to encode\n", __func__);
                result->error_code = 4;
                llama_sampler_free(smpl);
                llama_free(ctx);
                llama_model_free(model);
                return result;
            }
            
            llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
            if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
                decoder_start_token_id = llama_vocab_bos(vocab);
            }
            batch = llama_batch_get_one(&decoder_start_token_id, 1);
        }
        
        // Main decode loop
        // Use system_clock for cross-process synchronization
        // This ensures that timing is synchronized between parent and subprocess
        const auto t_main_start = std::chrono::system_clock::now();
        int n_decode = 0;
        llama_token new_token_id;
        
        for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
            // Use system_clock for cross-process time synchronization
            auto token_start = std::chrono::system_clock::now();
            
            // Store token start time as seconds since epoch
            auto token_start_epoch = std::chrono::duration<double>(token_start.time_since_epoch()).count();
            
            // Evaluate the current batch
            if (llama_decode(ctx, batch)) {
                fprintf(stderr, "%s: failed to decode\n", __func__);
                result->error_code = 5;
                llama_sampler_free(smpl);
                llama_free(ctx);
                llama_model_free(model);
                return result;
            }
            
            n_pos += batch.n_tokens;
            
            // Sample the next token
            new_token_id = llama_sampler_sample(smpl, ctx, -1);
            
            auto token_end = std::chrono::system_clock::now();
            
            // Check if end of generation
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }
            
            // Convert token to text
            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                result->error_code = 6;
                llama_sampler_free(smpl);
                llama_free(ctx);
                llama_model_free(model);
                return result;
            }
            std::string token_text(buf, n);
            output_text += token_text;
            
            // Calculate time per token
            float token_time_ms = std::chrono::duration<float, std::milli>(token_end - token_start).count();
            result->time_per_token.push_back(token_time_ms);
            result->token_start_time.push_back(token_start_epoch);
            
            // Prepare next batch
            batch = llama_batch_get_one(&new_token_id, 1);
            n_decode += 1;
        }
        
        const auto t_main_end = std::chrono::system_clock::now();
        
        // Store results
        result->output_text = output_text;
        result->n_tokens_generated = n_decode;
        result->total_time_ms = std::chrono::duration<float, std::milli>(t_main_end - t_main_start).count();
        
        // Clean up
        llama_sampler_free(smpl);
        llama_free(ctx);
        llama_model_free(model);
        
        return result;
    }
    
    // Getter functions for GenerationResult
    const char* get_output_text(GenerationResult* result) {
        if (!result) return nullptr;
        return result->output_text.c_str();
    }
    
    int get_n_tokens_generated(GenerationResult* result) {
        if (!result) return 0;
        return result->n_tokens_generated;
    }
    
    float get_total_time_ms(GenerationResult* result) {
        if (!result) return 0.0f;
        return result->total_time_ms;
    }
    
    int get_error_code(GenerationResult* result) {
        if (!result) return -1;
        return result->error_code;
    }
    
    int get_time_per_token_count(GenerationResult* result) {
        if (!result) return 0;
        return result->time_per_token.size();
    }
    
    float get_time_per_token_at(GenerationResult* result, int index) {
        if (!result || index < 0 || index >= (int)result->time_per_token.size()) {
            return 0.0f;
        }
        return result->time_per_token[index];
    }
    
    int get_token_start_time_count(GenerationResult* result) {
        if (!result) return 0;
        return result->token_start_time.size();
    }
    
    double get_token_start_time_at(GenerationResult* result, int index) {
        if (!result || index < 0 || index >= (int)result->token_start_time.size()) {
            return 0.0;
        }
        return result->token_start_time[index];
    }
    
    // Free the result structure
    void free_generation_result(GenerationResult* result) {
        if (result) {
            delete result;
        }
    }
}