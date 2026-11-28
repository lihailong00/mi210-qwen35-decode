#include <cstdlib>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int compare(ggml_backend_t backend, int k, int cols, bool strided) {
    constexpr int rows = 256;
    ggml_init_params params = {size_t(2) << 20, nullptr, true};
    ggml_context * weights = ggml_init(params);
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * w = ggml_new_tensor_2d(weights, GGML_TYPE_Q2_0, k, rows);
    const int padding = strided ? 67 : 0;
    const int stride = k + padding;
    const int offset = strided ? 3 : 0;
    ggml_tensor * storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, stride, cols);
    ggml_tensor * x = strided ? ggml_view_2d(ctx, storage, k, cols, stride * sizeof(float), offset * sizeof(float)) : storage;
    ggml_tensor * batched = ggml_mul_mat(ctx, w, x);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, batched);
    std::vector<ggml_tensor *> inputs(cols), separate(cols);
    for (int c = 0; c < cols; ++c) {
        inputs[c] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);
        separate[c] = ggml_mul_mat(ctx, w, inputs[c]);
        ggml_build_forward_expand(graph, separate[c]);
    }
    auto wb = ggml_backend_alloc_ctx_tensors(weights, backend);
    auto ab = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!wb || !ab) return 1;
    ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::mt19937 rng(k+cols);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    std::vector<float> wf(size_t(k)*rows), xf(size_t(stride)*cols);
    for (float & value : wf) value = dist(rng);
    for (float & value : xf) value = dist(rng);
    for (int c = 0; c < cols; ++c) {
        auto * row = xf.data() + c * stride + offset;
        std::fill(row, row + 32, 0.f);
        for (int i = 32; i < 64; ++i) row[i] = i == 32 ? 127.f : ((i % 7) - 3) + 0.5f;
    }
    std::vector<uint8_t> packed(ggml_nbytes(w));
    if (ggml_quantize_chunk(GGML_TYPE_Q2_0,wf.data(),packed.data(),0,rows,k,nullptr)!=packed.size()) return 1;
    ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(storage, xf.data(), 0, ggml_nbytes(storage));
    for (int c = 0; c < cols; ++c)
        ggml_backend_tensor_set(inputs[c], xf.data()+c*stride+offset, 0, ggml_nbytes(inputs[c]));
    if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) return 1;
    std::vector<float> out(rows*cols), ref(rows);
    ggml_backend_tensor_get(batched, out.data(), 0, ggml_nbytes(batched));
    int differences = 0;
    for (int c = 0; c < cols; ++c) {
        ggml_backend_tensor_get(separate[c], ref.data(), 0, ggml_nbytes(separate[c]));
        for (int r = 0; r < rows; ++r)
            differences += std::memcmp(&ref[r], &out[c*rows+r], sizeof(float)) != 0;
    }
    std::printf("K=%d cols=%d strided=%d logits=%d mismatches=%d\n",k,cols,strided,cols*rows,differences);
    ggml_backend_buffer_free(ab);
    ggml_backend_buffer_free(wb);
    ggml_free(ctx);
    ggml_free(weights);
    return differences != 0;
}

int main() {
    if (const char * dir = std::getenv("BONSAI_BACKEND_DIR")) ggml_backend_load_all_from_path(dir);
    else ggml_backend_load_all();
    auto backend = ggml_backend_init_by_name("ROCm0", nullptr);
    if (!backend) return 1;
    int rc = 0;
    for (int k : {4096,4352,5120,6144,17408,32768})
        for (int c : {2,3,4})
            for (bool strided : {false, true}) rc |= compare(backend,k,c,strided);
    ggml_backend_free(backend);
    return rc;
}
