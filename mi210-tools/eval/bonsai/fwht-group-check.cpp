#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <vector>

// Run twice in separate processes with GGML_MMVQ_Q2_0_FUSED_INPUT=0 and =1,
// then compare the two binary output files.
static void run_case(ggml_backend_t backend, int width, int columns, int count, int escape,
                     bool strided, std::ofstream & dump) {
    constexpr int block = 1024;
    const ggml_init_params params = {2u << 20, nullptr, true};
    ggml_context * weights = ggml_init(params);
    ggml_context * ctx = ggml_init(params);
    auto * rotation = ggml_new_tensor_2d(weights, GGML_TYPE_F32, block, block);
    auto * signs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
    const int stride = width + (strided ? 32 : 0);
    auto * storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, stride, columns);
    auto * x = strided ? ggml_view_2d(ctx, storage, width, columns, stride * sizeof(float), 0) : storage;
    auto * mul = ggml_mul(ctx, x, signs);
    auto * blocks = ggml_reshape_2d(ctx, mul, block, width / block * columns);
    auto * transform = ggml_mul_mat(ctx, rotation, blocks);
    ggml_mul_mat_set_hint(transform, GGML_HINT_SRC0_IS_HADAMARD);
    auto * act = ggml_reshape_2d(ctx, transform, width, columns);
    auto * graph = ggml_new_graph_custom(ctx, 64, false);
    std::vector<ggml_tensor *> matrices, outputs;
    for (int i = 0; i < count; ++i) {
        auto * w = ggml_new_tensor_2d(weights, GGML_TYPE_Q2_0, width, 128 + i * 72);
        matrices.push_back(w);
        outputs.push_back(ggml_mul_mat(ctx, w, act));
        ggml_set_output(outputs.back());
        ggml_build_forward_expand(graph, outputs.back());
    }
    if (escape) {
        // An observable intermediate or an extra consumer must survive fusion.
        const std::array<ggml_tensor *, 4> extra = {mul, blocks, transform, act};
        auto * result = escape <= 4 ? extra[escape - 1] : ggml_sqr(ctx, act);
        ggml_set_output(result);
        ggml_build_forward_expand(graph, result);
        outputs.push_back(result);
    }
    auto wb = ggml_backend_alloc_ctx_tensors(weights, backend);
    auto ab = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(wb && ab);
    ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_clear(ab, 0x55);
    std::vector<float> h(block * block);
    for (int i = 0; i < block; ++i) {
        for (int j = 0; j < block; ++j) {
            h[i * block + j] = __builtin_parity(unsigned(i & j)) ? -1.f / 32.f : 1.f / 32.f;
        }
    }
    ggml_backend_tensor_set(rotation, h.data(), 0, ggml_nbytes(rotation));
    std::mt19937 rng(width + count + columns);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    for (auto * w : matrices) {
        std::vector<float> f(ggml_nelements(w));
        for (float & value : f) value = dist(rng);
        std::vector<uint8_t> data(ggml_nbytes(w));
        GGML_ASSERT(ggml_quantize_chunk(w->type, f.data(), data.data(), 0, w->ne[1], width, nullptr) == data.size());
        ggml_backend_tensor_set(w, data.data(), 0, data.size());
    }
    for (int pass = 0; pass < 3; ++pass) {
        std::vector<float> s(width), f(ggml_nelements(storage));
        for (float & value : s) value = pass == 1 ? dist(rng) : (rng() & 1 ? -1.f : 1.f);
        for (float & value : f) value = pass == 2 ? 0.f : dist(rng);
        ggml_backend_tensor_set(signs, s.data(), 0, ggml_nbytes(signs));
        ggml_backend_tensor_set(storage, f.data(), 0, ggml_nbytes(storage));
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        for (auto * tensor : outputs) {
            std::vector<float> values(ggml_nelements(tensor));
            ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
            dump.write(reinterpret_cast<const char *>(values.data()), ggml_nbytes(tensor));
            GGML_ASSERT(dump.good());
        }
    }
    ggml_backend_buffer_free(ab);
    ggml_backend_buffer_free(wb);
    ggml_free(ctx);
    ggml_free(weights);
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s BACKEND_DIRECTORY OUTPUT.bin\n", argv[0]);
        return 1;
    }
    ggml_backend_load_all_from_path(argv[1]);
    auto backend = ggml_backend_init_by_name("ROCm0", nullptr);
    std::ofstream dump(argv[2], std::ios::binary);
    if (!backend || !dump) return 1;
    for (int width : {5120, 6144, 17408}) {
        for (int columns : {2, 3, 4}) {
            for (int count : {1, 2, 3}) run_case(backend, width, columns, count, 0, false, dump);
        }
    }
    for (int escape = 1; escape <= 5; ++escape) run_case(backend, 5120, 4, 2, escape, false, dump);
    run_case(backend, 5120, 4, 3, 0, true, dump);
    run_case(backend, 5120, 5, 2, 0, false, dump);
    run_case(backend, 5120, 4, 4, 0, false, dump);
    ggml_backend_free(backend);
    dump.close();
    return dump.good() ? 0 : 1;
}
