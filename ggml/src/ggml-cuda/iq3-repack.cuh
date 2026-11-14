#pragma once

// This experiment keeps the original GGUF data available for prefill and M>1.
struct codex_iq3_meta {
    ggml_half d;
    uint8_t scales[4];
};
static_assert(sizeof(codex_iq3_meta) == 6);

#if defined(GGML_USE_HIP)
const void * codex_iq3_get_packed(const ggml_tensor * tensor, int device, hipStream_t stream);

// generic cache of re-laid-out copies of weight tensors (created once outside graph capture, freed with the buffer)
typedef void (*ggml_cuda_repack_fn)(const void * src, void * dst, const ggml_tensor * t, hipStream_t stream);
const void * ggml_cuda_get_repacked(const ggml_tensor * tensor, int device, hipStream_t stream, int kind, size_t bytes, ggml_cuda_repack_fn fn);
#endif
