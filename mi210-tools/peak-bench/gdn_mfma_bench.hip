// Harness for the chunked MFMA gated-delta kernel: correctness vs a CPU reference, then timing.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include "gdn_mfma_kernel.inc"

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP %s @%d\n", hipGetErrorString(e), __LINE__); exit(1);} } while (0)

struct problem {
    int H = 48, Hk = 16; int64_t n; const int D = 128;
    std::vector<float> q, k, v, g, beta, state;
    void init(int64_t n_, unsigned seed) {
        n = n_; std::mt19937 rng(seed); std::normal_distribution<float> nd(0, 1); std::uniform_real_distribution<float> ud(0, 1);
        q.resize(D * Hk * n); k.resize(D * Hk * n); v.resize((size_t) D * H * n); g.resize(H * n); beta.resize(H * n); state.resize((size_t) D * D * H);
        for (int64_t t = 0; t < n; ++t) for (int h = 0; h < Hk; ++h) {
            double nq = 0, nk = 0; float * qq = &q[(t * Hk + h) * D], * kk = &k[(t * Hk + h) * D];
            for (int i = 0; i < D; ++i) { qq[i] = nd(rng); kk[i] = nd(rng); nq += qq[i] * qq[i]; nk += kk[i] * kk[i]; }
            for (int i = 0; i < D; ++i) { qq[i] /= sqrt(nq); kk[i] /= sqrt(nk); }
        }
        for (auto & x : v) x = nd(rng);
        for (auto & x : g) x = -0.2f * ud(rng) * ud(rng);
        for (auto & x : beta) x = ud(rng);
        for (auto & x : state) x = 0.1f * nd(rng);
    }
};

// AR reference in double precision (same math as gated_delta_net_cuda)
static void reference(const problem & p, std::vector<double> & out, std::vector<double> & st) {
    const int D = p.D; const double scale = 1.0 / sqrt((double) D);
    out.assign((size_t) D * p.H * p.n, 0); st.assign(p.state.begin(), p.state.end());
    for (int h = 0; h < p.H; ++h) for (int64_t t = 0; t < p.n; ++t) {
        const float * q = &p.q[(t * p.Hk + h % p.Hk) * D], * k = &p.k[(t * p.Hk + h % p.Hk) * D], * v = &p.v[(t * p.H + h) * D];
        const double gv = exp((double) p.g[t * p.H + h]), bv = p.beta[t * p.H + h];
        for (int col = 0; col < D; ++col) {
            double * s = &st[((size_t) h * D + col) * D];
            double kv = 0; for (int i = 0; i < D; ++i) kv += s[i] * k[i];
            const double delta = (v[col] - gv * kv) * bv;
            double at = 0; for (int i = 0; i < D; ++i) { s[i] = gv * s[i] + k[i] * delta; at += s[i] * q[i]; }
            out[(t * p.H + h) * D + col] = at * scale;
        }
    }
}

int main(int argc, char ** argv) {
    for (int64_t n : {7, 16, 100, 4096}) {
        problem p; p.init(n, 1234);
        const int D = 128;
        float *dq, *dk, *dv, *dg, *db, *ds, *dout, *dso;
        auto up = [&](float *& d, const std::vector<float> & h) { CK(hipMalloc(&d, h.size() * 4)); CK(hipMemcpy(d, h.data(), h.size() * 4, hipMemcpyHostToDevice)); };
        up(dq, p.q); up(dk, p.k); up(dv, p.v); up(dg, p.g); up(db, p.beta); up(ds, p.state);
        CK(hipMalloc(&dout, (size_t) D * p.H * n * 4)); CK(hipMalloc(&dso, (size_t) D * D * p.H * 4));
        gdn_mfma_args a = { dq, dk, dv, dg, db, ds, dout, dso, p.H, p.Hk, n, 1, 1,
                            D, (int64_t) D * p.Hk, (int64_t) D * p.Hk * n, D, (int64_t) D * p.H, (int64_t) D * p.H * n, 1, p.H, (int64_t) p.H * n,
                            1.0f / sqrtf((float) D) };
        const int blocks = p.H * (GDN_D / GDN_V);
        hipLaunchKernelGGL(gdn_mfma_kernel, dim3(blocks), dim3(256), 0, 0, a); CK(hipDeviceSynchronize());
        if (n <= 512) {
            std::vector<double> ro, rs; reference(p, ro, rs);
            std::vector<float> go(ro.size()), gs(rs.size());
            CK(hipMemcpy(go.data(), dout, go.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(gs.data(), dso, gs.size() * 4, hipMemcpyDeviceToHost));
            double eo = 0, no = 0, es = 0, ns = 0, mo = 0;
            for (size_t i = 0; i < ro.size(); ++i) { double d = go[i] - ro[i]; eo += d * d; no += ro[i] * ro[i]; mo = fmax(mo, fabs(d)); }
            for (size_t i = 0; i < rs.size(); ++i) { double d = gs[i] - rs[i]; es += d * d; ns += rs[i] * rs[i]; }
            printf("n=%lld: out rel-L2 err %.3e (max abs %.3e, rms ref %.3e), state rel-L2 err %.3e\n", (long long) n, sqrt(eo / no), mo, sqrt(no / ro.size()), sqrt(es / ns));
        }
        hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
        const int R = n > 1000 ? 300 : 5;
        if (n > 1000) { for (int r = 0; r < 400; ++r) hipLaunchKernelGGL(gdn_mfma_kernel, dim3(blocks), dim3(256), 0, 0, a); CK(hipDeviceSynchronize()); }
        CK(hipEventRecord(e0)); for (int r = 0; r < R; ++r) hipLaunchKernelGGL(gdn_mfma_kernel, dim3(blocks), dim3(256), 0, 0, a); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
        float ms; CK(hipEventElapsedTime(&ms, e0, e1));
        printf("n=%lld: %.3f ms per call (%d blocks)\n", (long long) n, ms / R, blocks);
        for (float * d : {dq, dk, dv, dg, db, ds, dout, dso}) hipFree(d);
    }
    return 0;
}
