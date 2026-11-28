// Dump raw float32 logits for single-token decode equivalence checks.
// Usage: logits MODEL INPUT OUTPUT [TOKEN_COUNT]
// Build: c++ -O2 -std=c++17 -Iinclude -Iggml/include this.cpp -Lbuild/bin
//        -Wl,-rpath,$PWD/build/bin -lllama -lggml -lggml-base -o logits
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 5) {
        std::fprintf(stderr, "usage: %s MODEL INPUT OUTPUT [TOKEN_COUNT]\n", argv[0]);
        return 1;
    }
    std::ifstream input(argv[2], std::ios::binary);
    std::ofstream output(argv[3], std::ios::binary);
    if (!input || !output) return 1;
    const std::string text((std::istreambuf_iterator<char>(input)), {});
    const int count = argc == 5 ? std::atoi(argv[4]) : 512;
    if (count <= 0 || count > 4096) return 1;
    const char * backend_dir = std::getenv("BONSAI_BACKEND_DIR");
    if (backend_dir) ggml_backend_load_all_from_path(backend_dir);
    else ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(text.size() + 16);
    int nt = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, true);
    if (nt < count) {
        llama_model_free(model);
        return 1;
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = count + 32;
    cp.n_batch = cp.n_ubatch = 1;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); return 1; }
    int result = 0;
    const size_t bytes = size_t(llama_vocab_n_tokens(vocab)) * sizeof(float);
    for (int i = 0; i < count; ++i) {
        llama_batch batch = llama_batch_get_one(&tokens[i], 1);
        if (llama_decode(ctx, batch) != 0) { result = 1; break; }
        output.write(reinterpret_cast<const char *>(llama_get_logits(ctx)), bytes);
        if (!output) { result = 1; break; }
    }
    output.close();
    if (!output) result = 1;
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    std::fprintf(stderr, "raw float32 logits: %s, tokens=%d, bytes/token=%zu\n",
                 result ? "FAILED" : "complete", count, bytes);
    return result;
}
