// MTP2-O2: bitwise N-token vs token-by-token test for the multi-token verification paths of the attention and
// activation producers (CUDA/HIP backend): every case is evaluated once as one N-token graph and once per token as an
// N=1 graph on the same data, and each token's outputs must be bit-identical.
//   fa      flash attention (causal, positions p0..p0+N-1, K/V cache views of the per-token n_kv; short and long KV) -> sigmoid(gate) * attn
//           -> mat-mul (IQ4_XS): FA_DEC N-query kernel + q8 epilogue vs the single-token fattn-dec kernel
//   norm    rms_norm(x) * w -> two mat-muls (Q4_K, IQ4_XS): multi-row q8 producer vs the single-row one
//   swiglu  swiglu(gate, up) -> mat-mul (IQ4_XS)
//   sigmul  sigmoid(a) * b -> mat-mul (IQ4_XS)
//   prep    q/k rms_norm * w + multi-rope, gate copy, k/v cache stores (the attn_prep chain) -> outputs and cache rows
// usage: test-mtp-nq [filter] (env as for the backend: GGML_CUDA_FA_DEC_NQ, GGML_Q8P_NC, GGML_MMVQ_NC_MAX, ...)
// exit code 0 = all cases bit-identical.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

static ggml_backend_t g_backend = nullptr;
static int    g_bench   = 0;   // bench mode (filter "bench"): extra evaluations per graph
static double g_last_us = 0.0;

struct run_io {
    std::vector<std::vector<uint8_t>> in;   // data per input tensor (in creation order)
    std::vector<std::vector<uint8_t>> out;  // data per output tensor
};

// build(ctx, inputs, outputs): creates inputs (pushed in order) and returns outputs
using builder = std::function<void(ggml_context *, std::vector<ggml_tensor *> &, std::vector<ggml_tensor *> &)>;

static bool run_graph(const builder & build, run_io & io) {
    ggml_init_params ip = { 256*ggml_tensor_overhead() + ggml_graph_overhead_custom(1024, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<ggml_tensor *> ins, outs;
    build(ctx, ins, outs);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, false);
    for (ggml_tensor * o : outs) {
        ggml_set_output(o);
        ggml_build_forward_expand(gf, o);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    if (io.in.size() != ins.size()) {
        fprintf(stderr, "input count mismatch %zu %zu\n", io.in.size(), ins.size());
        exit(2);
    }
    for (size_t k = 0; k < ins.size(); ++k) {
        if (io.in[k].size() != ggml_nbytes(ins[k])) {
            fprintf(stderr, "input %zu (%s) size mismatch %zu %zu\n", k, ins[k]->name, io.in[k].size(), ggml_nbytes(ins[k]));
            exit(2);
        }
        ggml_backend_tensor_set(ins[k], io.in[k].data(), 0, io.in[k].size());
    }
    if (ggml_backend_graph_compute(g_backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        exit(2);
    }
    if (g_bench > 0) {   // bench mode: g_bench more evaluations of the same graph, wall time per evaluation (incl. sync)
        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < g_bench; ++it) {
            ggml_backend_graph_compute(g_backend, gf);
        }
        g_last_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / g_bench;
    }
    io.out.resize(outs.size());
    for (size_t k = 0; k < outs.size(); ++k) {
        io.out[k].resize(ggml_nbytes(outs[k]));
        ggml_backend_tensor_get(outs[k], io.out[k].data(), 0, io.out[k].size());
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

static std::mt19937 rng(1234);

static std::vector<uint8_t> rand_f32(size_t n, float lo = -1.0f, float hi = 1.0f) {
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<uint8_t> b(n*4);
    float * f = (float *) b.data();
    for (size_t i = 0; i < n; ++i) {
        f[i] = u(rng);
    }
    return b;
}
static std::vector<uint8_t> rand_f16(size_t n, float lo = -1.0f, float hi = 1.0f) {
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<uint8_t> b(n*2);
    ggml_fp16_t * f = (ggml_fp16_t *) b.data();
    for (size_t i = 0; i < n; ++i) {
        f[i] = ggml_fp32_to_fp16(u(rng));
    }
    return b;
}
static std::vector<uint8_t> rand_q(ggml_type t, int64_t k, int64_t rows) {
    std::vector<uint8_t> f = rand_f32((size_t) k*rows);
    std::vector<uint8_t> q(ggml_row_size(t, k)*rows);
    ggml_quantize_chunk(t, (const float *) f.data(), q.data(), 0, rows, k, nullptr);
    return q;
}
// columns [c0, c0+n) of a [row_bytes, ncols] buffer
static std::vector<uint8_t> cols(const std::vector<uint8_t> & b, size_t row_bytes, int c0, int n) {
    return std::vector<uint8_t>(b.begin() + (size_t) c0*row_bytes, b.begin() + (size_t) (c0 + n)*row_bytes);
}

static int g_fail = 0, g_cases = 0;

// largest N whose consumer mat-muls take the per-column exact N-column route (IQ4_XS: family B) in this process
static int nc_max_env() {
    const char * e = getenv("GGML_MMVQ_NC");
    if (e && e[0] == '0') {
        return 0;
    }
    const char * m = getenv("GGML_MMVQ_NC_MAX");
    // MTP2-merge: the weights of these graphs are not in a WEIGHTS buffer, so the N-column GEMV families decline them and
    // the stock MMVQ kernels (bit-identical per column up to 4 columns only) run the consumers: out1 is required exact up to 4
    GGML_UNUSED(m);
    return 4;
}

// compare output k of the N run (token-major, tok_bytes[k] per token) with the N=1 runs. Outputs with req[k] = false
// are reported but do not fail the case (consumer mat-muls outside the N-column route, which are not column-exact).
static void check(const char * name, const run_io & rn, const std::vector<run_io> & r1, const std::vector<size_t> & tok_bytes,
        const std::vector<bool> & req) {
    g_cases++;
    bool fail = false;
    std::string msg;
    for (size_t k = 0; k < tok_bytes.size(); ++k) {
        size_t ndiff = 0, ntot = 0;
        for (size_t t = 0; t < r1.size(); ++t) {
            const uint8_t * a = rn.out[k].data() + t*tok_bytes[k];
            const uint8_t * b = r1[t].out[k].data();
            const size_t ne = tok_bytes[k] / 4;
            for (size_t e = 0; e < ne; ++e) {
                ntot++;
                ndiff += memcmp(a + 4*e, b + 4*e, 4) != 0;
            }
        }
        char s[96];
        snprintf(s, sizeof(s), " out%zu%s %zu/%zu", k, req[k] ? "" : "(info)", ndiff, ntot);
        msg += s;
        fail |= req[k] && ndiff > 0;
    }
    printf("%-36s %s  differ:%s\n", name, fail ? "FAIL" : "OK  ", msg.c_str());
    if (fail) {
        g_fail++;
    }
}
static int64_t pad256(int64_t x) { return (x + 255) / 256 * 256; }

// ------------------------------------------------------------------------------------------------ flash attention
static void case_fa(int N, int p0, int n_ctx = 1024, bool bench = false) {
    const int D = 256, H = 24, HKV = 4, M = 5120;
    const int64_t HD = (int64_t) D*H;
    const int n_kv = (int) std::min<int64_t>(n_ctx, std::max<int64_t>(256, pad256(p0 + N)));
    // inputs: q [D, H, N], kcache/vcache [D*HKV, n_ctx] f16, mask [n_kv, N], gate [HD, N], W iq4_xs [HD, M]
    const auto q = rand_f32((size_t) HD*N);
    const auto kc = rand_f16((size_t) D*HKV*n_ctx);
    const auto vc = rand_f16((size_t) D*HKV*n_ctx);
    const auto gt = rand_f32((size_t) HD*N, -3.0f, 3.0f);
    const auto W = rand_q(GGML_TYPE_IQ4_XS, HD, M);
    auto mask_rows = [&](int nkv, int t0, int nt) {
        std::vector<uint8_t> b((size_t) nkv*nt*2);
        ggml_fp16_t * m = (ggml_fp16_t *) b.data();
        for (int t = 0; t < nt; ++t) {
            for (int c = 0; c < nkv; ++c) {
                m[(size_t) t*nkv + c] = ggml_fp32_to_fp16(c <= p0 + t0 + t ? 0.0f : -INFINITY);
            }
        }
        return b;
    };
    auto mk = [&](int nt, int nkv) {
        return [=](ggml_context * ctx, std::vector<ggml_tensor *> & ins, std::vector<ggml_tensor *> & outs) {
            ggml_tensor * tq = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, nt);
            ggml_tensor * tk = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D*HKV, n_ctx);
            ggml_tensor * tv = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D*HKV, n_ctx);
            ggml_tensor * tm = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nkv, nt);
            ggml_tensor * tg = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HD, nt);
            ggml_tensor * tw = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_XS, HD, M);
            ins = { tq, tk, tv, tm, tg, tw };
            ggml_tensor * k = ggml_view_3d(ctx, tk, D, HKV, nkv, D*2, D*HKV*2, 0);
            ggml_tensor * v = ggml_view_3d(ctx, tv, D, HKV, nkv, D*2, D*HKV*2, 0);
            ggml_tensor * qp = ggml_permute(ctx, tq, 0, 2, 1, 3);
            k = ggml_permute(ctx, k, 0, 2, 1, 3);
            v = ggml_permute(ctx, v, 0, 2, 1, 3);
            ggml_tensor * fa = ggml_flash_attn_ext(ctx, qp, k, v, tm, 1.0f/16.0f, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
            ggml_tensor * cur = ggml_reshape_2d(ctx, fa, HD, nt);
            ggml_tensor * sg  = ggml_sigmoid(ctx, tg);
            ggml_tensor * g   = ggml_mul(ctx, cur, sg);
            ggml_tensor * y   = ggml_mul_mat(ctx, tw, g);
            outs = { g, y };
            if (bench) {
                outs = { g };   // bench: attention + gate only
            }
        };
    };
    run_io rn;
    rn.in = { q, kc, vc, mask_rows(n_kv, 0, N), gt, W };
    run_graph(mk(N, n_kv), rn);
    const double us_n = g_last_us;
    double us_1 = 0.0;
    std::vector<run_io> r1(N);
    for (int t = 0; t < N; ++t) {
        const int n_kv_t = (int) std::min<int64_t>(n_kv, std::max<int64_t>(256, pad256(p0 + t + 1)));
        r1[t].in = { cols(q, HD*4, t, 1), kc, vc, mask_rows(n_kv_t, t, 1), cols(gt, HD*4, t, 1), W };
        run_graph(mk(1, n_kv_t), r1[t]);
        us_1 += g_last_us;
    }
    char name[96];
    snprintf(name, sizeof(name), "fa N=%d p0=%d n_kv=%d", N, p0, n_kv);
    if (bench) {
        printf("bench %-28s N-token graph %8.1f us | single-token graph %7.1f us (x%d = %8.1f) | N/1 = %.2f\n",
            name, us_n, us_1/N, N, us_1, us_n/(us_1/N));
        return;
    }
    check(name, rn, r1, { (size_t) HD*4, (size_t) M*4 }, { true, N <= nc_max_env() });
}

// ------------------------------------------------------------------------------------------------ norm producer
static void case_norm(int N, int K = 5120) {
    const int M1 = 3072, M2 = 2048;
    const auto x = rand_f32((size_t) K*N, -2.0f, 2.0f);
    const auto w = rand_f32((size_t) K, 0.5f, 1.5f);
    const auto W1 = rand_q(GGML_TYPE_Q4_K, K, M1);
    const auto W2 = rand_q(GGML_TYPE_IQ4_XS, K, M2);
    auto mk = [=](int nt) {
        return [=](ggml_context * ctx, std::vector<ggml_tensor *> & ins, std::vector<ggml_tensor *> & outs) {
            ggml_tensor * tx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, nt);
            ggml_tensor * tw = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
            ggml_tensor * t1 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, K, M1);
            ggml_tensor * t2 = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_XS, K, M2);
            ins = { tx, tw, t1, t2 };
            ggml_tensor * n = ggml_mul(ctx, ggml_rms_norm(ctx, tx, 1e-6f), tw);
            outs = { n, ggml_mul_mat(ctx, t1, n), ggml_mul_mat(ctx, t2, n) };
        };
    };
    run_io rn;
    rn.in = { x, w, W1, W2 };
    run_graph(mk(N), rn);
    std::vector<run_io> r1(N);
    for (int t = 0; t < N; ++t) {
        r1[t].in = { cols(x, K*4, t, 1), w, W1, W2 };
        run_graph(mk(1), r1[t]);
    }
    char name[96];
    snprintf(name, sizeof(name), "norm N=%d K=%d", N, K);
    check(name, rn, r1, { (size_t) K*4, (size_t) M1*4, (size_t) M2*4 }, { true, false, N <= nc_max_env() });
}

// ------------------------------------------------------------------------------------------------ swiglu / sigmoid*mul
static void case_glu(int N, bool sigmul) {
    const int K = sigmul ? 6144 : 17408, M = 5120;
    const auto a = rand_f32((size_t) K*N, -4.0f, 4.0f);
    const auto b = rand_f32((size_t) K*N, -2.0f, 2.0f);
    const auto W = rand_q(GGML_TYPE_IQ4_XS, K, M);
    auto mk = [=](int nt) {
        return [=](ggml_context * ctx, std::vector<ggml_tensor *> & ins, std::vector<ggml_tensor *> & outs) {
            ggml_tensor * ta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, nt);
            ggml_tensor * tb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, nt);
            ggml_tensor * tw = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_XS, K, M);
            ins = { ta, tb, tw };
            ggml_tensor * s = sigmul ? ggml_mul(ctx, ggml_sigmoid(ctx, ta), tb) : ggml_swiglu_split(ctx, ta, tb);
            outs = { s, ggml_mul_mat(ctx, tw, s) };
        };
    };
    run_io rn;
    rn.in = { a, b, W };
    run_graph(mk(N), rn);
    std::vector<run_io> r1(N);
    for (int t = 0; t < N; ++t) {
        r1[t].in = { cols(a, K*4, t, 1), cols(b, K*4, t, 1), W };
        run_graph(mk(1), r1[t]);
    }
    char name[96];
    snprintf(name, sizeof(name), "%s N=%d K=%d", sigmul ? "sigmul" : "swiglu", N, K);
    check(name, rn, r1, { (size_t) K*4, (size_t) M*4 }, { true, N <= nc_max_env() });
}

// ------------------------------------------------------------------------------------------------ attn_prep chain
// the qwen35 order (src/models/qwen35.cpp build_layer_attn): q view of Qcur_full -> rms_norm * wq, k rms_norm * wk,
// gate cont, v reshape, multi-rope q / k, then the k and v cache stores (set_rows at the token cells)
static void case_prep(int N, int p0) {
    const int hd = 256, nh = 24, nkv = 4, n_ctx = 1024;
    const auto qf = rand_f32((size_t) 2*hd*nh*N, -3.0f, 3.0f);
    const auto kk = rand_f32((size_t) hd*nkv*N, -3.0f, 3.0f);
    const auto vv = rand_f32((size_t) hd*nkv*N, -3.0f, 3.0f);
    const auto wq = rand_f32(hd, 0.5f, 1.5f);
    const auto wk = rand_f32(hd, 0.5f, 1.5f);
    const auto kc = rand_f16((size_t) hd*nkv*n_ctx);
    const auto vc = rand_f16((size_t) hd*nkv*n_ctx);
    auto pos = [&](int t0, int nt) {
        std::vector<uint8_t> b((size_t) 4*nt*4);
        int32_t * p = (int32_t *) b.data();
        for (int s = 0; s < 4; ++s) {
            for (int t = 0; t < nt; ++t) {
                p[s*nt + t] = s == 3 ? 0 : p0 + t0 + t;
            }
        }
        return b;
    };
    auto idx = [&](int t0, int nt) {
        std::vector<uint8_t> b((size_t) nt*8);
        int64_t * p = (int64_t *) b.data();
        for (int t = 0; t < nt; ++t) {
            p[t] = p0 + t0 + t;
        }
        return b;
    };
    auto mk = [=](int nt) {
        return [=](ggml_context * ctx, std::vector<ggml_tensor *> & ins, std::vector<ggml_tensor *> & outs) {
            ggml_tensor * tqf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2*hd*nh, nt);
            ggml_tensor * tk  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd*nkv, nt);
            ggml_tensor * tv  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd*nkv, nt);
            ggml_tensor * twq = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
            ggml_tensor * twk = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
            ggml_tensor * tkc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hd*nkv, n_ctx);
            ggml_tensor * tvc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hd*nkv, n_ctx);
            ggml_tensor * tp  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4*nt);
            ggml_tensor * ti  = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, nt);
            ins = { tqf, tk, tv, twq, twk, tkc, tvc, tp, ti };
            ggml_tensor * q = ggml_view_3d(ctx, tqf, hd, nh, nt, 4*hd*2, 4*hd*2*nh, 0);
            q = ggml_mul(ctx, ggml_rms_norm(ctx, q, 1e-6f), twq);
            ggml_tensor * k = ggml_reshape_3d(ctx, tk, hd, nkv, nt);
            k = ggml_mul(ctx, ggml_rms_norm(ctx, k, 1e-6f), twk);
            ggml_tensor * g = ggml_view_3d(ctx, tqf, hd, nh, nt, 4*hd*2, 4*hd*2*nh, 4*hd);
            g = ggml_cont_2d(ctx, g, hd*nh, nt);
            int sections[4] = { 11, 11, 10, 0 };
            q = ggml_rope_multi(ctx, q, tp, nullptr, 64, sections, GGML_ROPE_TYPE_IMROPE, 262144, 10000000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            k = ggml_rope_multi(ctx, k, tp, nullptr, 64, sections, GGML_ROPE_TYPE_IMROPE, 262144, 10000000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            ggml_tensor * k2 = ggml_view_2d(ctx, k, hd*nkv, nt, k->nb[2], 0);
            ggml_tensor * sk = ggml_set_rows(ctx, tkc, k2, ti);
            ggml_tensor * sv = ggml_set_rows(ctx, tvc, tv, ti);
            outs = { q, g, sk, sv };
        };
    };
    run_io rn;
    rn.in = { qf, kk, vv, wq, wk, kc, vc, pos(0, N), idx(0, N) };
    run_graph(mk(N), rn);
    std::vector<run_io> r1(N);
    for (int t = 0; t < N; ++t) {
        r1[t].in = { cols(qf, 2*hd*nh*4, t, 1), cols(kk, hd*nkv*4, t, 1), cols(vv, hd*nkv*4, t, 1), wq, wk, kc, vc, pos(t, 1), idx(t, 1) };
        run_graph(mk(1), r1[t]);
    }
    // q / gate: token-major; the caches: compare the row of each token's cell (written by that token's store)
    run_io rn2 = rn;
    std::vector<run_io> r12 = r1;
    const size_t row = (size_t) hd*nkv*2;
    for (int k = 2; k < 4; ++k) {
        std::vector<uint8_t> a((size_t) N*row);
        for (int t = 0; t < N; ++t) {
            memcpy(a.data() + t*row, rn.out[k].data() + (size_t) (p0 + t)*row, row);
            r12[t].out[k].assign(r1[t].out[k].begin() + (size_t) (p0 + t)*row, r1[t].out[k].begin() + (size_t) (p0 + t + 1)*row);
        }
        rn2.out[k] = a;
    }
    char name[96];
    snprintf(name, sizeof(name), "prep N=%d p0=%d", N, p0);
    check(name, rn2, r12, { (size_t) hd*nh*4, (size_t) hd*nh*4, row, row }, { true, true, true, true });
}

int main(int argc, char ** argv) {
    const std::string filter = argc > 1 ? argv[1] : "";
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        fprintf(stderr, "no GPU device\n");
        return 2;
    }
    g_backend = ggml_backend_dev_init(dev, nullptr);
    printf("device %s\n", ggml_backend_dev_name(dev));
    if (filter == "bench") {
        // standalone time of the attention + gate graph: one N-token graph vs the single-token graph of each token
        g_bench = 200;
        for (int nctx : { 1024, 4096 }) {
            for (int N : { 1, 2, 4, 8, 16 }) {
                case_fa(N, nctx == 1024 ? 700 : 3000, nctx == 1024 ? 1024 : 8192, true);
            }
        }
        g_bench = 0;
        ggml_backend_free(g_backend);
        return 0;
    }
    auto want = [&](const char * c) { return filter.empty() || filter.find(c) != std::string::npos; };
    if (want("fa")) {
        const int ps[][2] = { {2, 0}, {4, 0}, {3, 100}, {4, 250}, {4, 253}, {4, 254}, {2, 255}, {8, 250}, {16, 240}, {16, 500},
                              {5, 509}, {4, 600}, {12, 760}, {16, 1000}, {6, 1018}, {2, 1022}, {16, 1008}, {4, 300} };
        for (const auto & p : ps) {
            case_fa(p[0], p[1]);
        }
        // long KV (n_kv > 1024: the single-token generic path per query), incl. the 256-boundaries at 1024 and above
        const int pl[][2] = { {4, 1021}, {2, 1023}, {16, 1020}, {4, 1100}, {4, 1278}, {3, 1279}, {8, 1530}, {16, 1790},
                              {4, 2046}, {2, 3000}, {16, 3070}, {4, 4090}, {16, 8000} };
        for (const auto & p : pl) {
            case_fa(p[0], p[1], 8192);
        }
    }
    if (want("norm")) {
        for (int N = 2; N <= 6; ++N) {
            case_norm(N);
        }
        case_norm(8);
        case_norm(16);
    }
    if (want("swiglu")) {
        for (int N : { 2, 3, 4, 5, 6, 8, 16 }) {
            case_glu(N, false);
        }
    }
    if (want("sigmul")) {
        for (int N : { 2, 4, 6, 8, 16 }) {
            case_glu(N, true);
        }
    }
    if (want("prep")) {
        for (int N : { 2, 4, 8, 16 }) {
            case_prep(N, 37);
        }
        case_prep(4, 254);
    }
    printf("%d/%d cases bit-identical\n", g_cases - g_fail, g_cases);
    ggml_backend_free(g_backend);
    return g_fail ? 1 : 0;
}
