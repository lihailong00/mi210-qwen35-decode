// ---- SoA re-layout prototypes (lossless): IQ3_XXS (98 B/blk, unchanged size) and IQ3_S (112 B/blk, +1.8%) ----
// Persistent workgroups: stage q8_1 activation into LDS once (int8 qs + float d), codebook in LDS,
// L lanes per row (one 32-weight sub-block per lane per step), grid-stride over row groups.

// IQ3_XXS SoA: QS[nb][64] | AUX[nb][32] | D[nb] (half)
static __global__ void soa_pack_iq3xxs(const block_iq3_xxs * __restrict__ raw, uint8_t * __restrict__ out, size_t nb) {
    const size_t g = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    if (g >= nb) return;
    const block_iq3_xxs & b = raw[g];
    uint8_t * qs  = out + g*64;
    uint8_t * aux = out + nb*64 + g*32;
    half    * d   = (half *)(out + nb*96);
    for (int i = 0; i < 64; ++i) qs[i] = b.qs[i];
    for (int i = 0; i < 32; ++i) aux[i] = b.qs[64+i];
    d[g] = b.d;
}

// IQ3_S SoA: QS[nb][64] | SIGNS[nb][32] | META[nb][16] = qh[8], scales[4], d(2), pad(2)
static __global__ void soa_pack_iq3s(const block_iq3_s * __restrict__ raw, uint8_t * __restrict__ out, size_t nb) {
    const size_t g = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    if (g >= nb) return;
    const block_iq3_s & b = raw[g];
    uint8_t * qs = out + g*64;
    uint8_t * sg = out + nb*64 + g*32;
    uint8_t * mt = out + nb*96 + g*16;
    for (int i = 0; i < 64; ++i) qs[i] = b.qs[i];
    for (int i = 0; i < 32; ++i) sg[i] = b.signs[i];
    for (int i = 0; i < 8; ++i) mt[i] = b.qh[i];
    for (int i = 0; i < 4; ++i) mt[8+i] = b.scales[i];
    memcpy(mt + 12, &b.d, 2); mt[14] = 0; mt[15] = 0;
}

template <int KMAX>
struct YStage {
    int8_t qs[KMAX];
    float  d[KMAX/32];
};

template <bool XXS, int L, int NW, int KMAX>
__launch_bounds__(NW*64, 1)
__global__ void soa_gemv_iq3(const uint8_t * __restrict__ w, const block_q8_1 * __restrict__ y, float * __restrict__ dst, const int K, const int N, const size_t nb) {
    __shared__ YStage<KMAX> ys;
    __shared__ uint32_t sgrid[XXS ? 256 : 512];
    const int tid = threadIdx.y*64 + threadIdx.x;
    {
        const uint32_t * gsrc = XXS ? (const uint32_t *) iq3xxs_grid : (const uint32_t *) iq3s_grid;
        for (int i = tid; i < (XXS ? 256 : 512); i += NW*64) sgrid[i] = gsrc[i];
        const int nblk = K/32;
        // each thread copies 4 bytes of qs per step
        for (int i = tid; i < K/4; i += NW*64) {
            const int blk = i/8, wi = i%8;
            ((int *) ys.qs)[i] = ((const int *) y[blk].qs)[wi];
        }
        for (int i = tid; i < nblk; i += NW*64) ys.d[i] = __low2float(y[i].ds);
        __syncthreads();
    }
    constexpr int G = 64/L;
    const int lane = threadIdx.x % L, grp = threadIdx.x / L;
    const int bpr = K/256;
    const int steps = K/(32*L);   // K multiple of 32*L assumed (1024 for L=32)
    const int ngroups = N;        // one row per lane-group
    for (int row = (blockIdx.x*NW + threadIdx.y)*G + grp; row < ngroups; row += gridDim.x*NW*G) {
        float acc = 0.0f;
        const size_t gbase = (size_t)row*bpr;
#pragma unroll 2
        for (int st = 0; st < steps; ++st) {
            const int s = st*L + lane;          // sub-block index within row
            const int b = s >> 3, j = s & 7;
            const size_t g = gbase + b;
            const uint2 q = *(const uint2 *)(w + g*64 + 8*j);
            const int4 yv0 = *(const int4 *)(ys.qs + s*32);
            const int4 yv1 = *(const int4 *)(ys.qs + s*32 + 16);
            const float yd = ys.d[s];
            const int yy[8] = {yv0.x, yv0.y, yv0.z, yv0.w, yv1.x, yv1.y, yv1.z, yv1.w};
            int sumi = 0;
            float dw;
            if constexpr (XXS) {
                const uint32_t aux32 = *(const uint32_t *)(w + nb*64 + g*32 + 4*j);
                dw = __half2float(*(const half *)(w + nb*96 + g*2));
                const uint32_t qq[2] = {q.x, q.y};
#pragma unroll
                for (int l0 = 0; l0 < 8; l0 += 2) {
                    const uint32_t i0 = (qq[l0/4] >> (8*(l0%4))) & 0xFF;
                    const uint32_t i1 = (qq[l0/4] >> (8*(l0%4) + 8)) & 0xFF;
                    const uint32_t g0 = sgrid[i0], g1 = sgrid[i1];
                    const uint8_t sign_code = aux32 >> (7*l0/2);
                    const uint32_t signs = sign_code ^ ((__popc(sign_code) & 1u) << 7);
                    sumi = ggml_cuda_dp4a(iq3_apply_signs(g0, signs & 15u), yy[l0+0], sumi);
                    sumi = ggml_cuda_dp4a(iq3_apply_signs(g1, signs >> 4),  yy[l0+1], sumi);
                }
                const int ls = aux32 >> 28;
                sumi = (ls*sumi + sumi/2)/2;
            } else {
                const uint32_t sg = *(const uint32_t *)(w + nb*64 + g*32 + 4*j);
                const uint8_t * mt = w + nb*96 + g*16;
                const uint32_t qh = mt[j];
                const uint32_t sc = mt[8 + j/2];
                dw = __half2float(*(const half *)(mt + 12));
                const uint32_t qq[2] = {q.x, q.y};
#pragma unroll
                for (int l0 = 0; l0 < 8; l0 += 2) {
                    const uint32_t i0 = ((qq[l0/4] >> (8*(l0%4))) & 0xFF)     | ((qh << (8 - l0)) & 0x100);
                    const uint32_t i1 = ((qq[l0/4] >> (8*(l0%4) + 8)) & 0xFF) | ((qh << (7 - l0)) & 0x100);
                    const uint32_t g0 = sgrid[i0], g1 = sgrid[i1];
                    const uint32_t signs = (sg >> (8*(l0/2))) & 0xFF;
                    sumi = ggml_cuda_dp4a(iq3_apply_signs(g0, signs & 15u), yy[l0+0], sumi);
                    sumi = ggml_cuda_dp4a(iq3_apply_signs(g1, signs >> 4),  yy[l0+1], sumi);
                }
                sumi *= 1 + 2*((sc >> (4*(j & 1))) & 0x0F);
            }
            acc += dw*yd*sumi;
        }
        acc = warp_reduce_sum<L>(acc);
        if (lane == 0) dst[row] = acc;
    }
}
