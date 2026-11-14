// ---- LDS-table variants of codebook vec_dots + retained packed path (included by gemv_bench.hip) ----

static __device__ __forceinline__ float dot_iq3_s_lds(const uint32_t * __restrict__ grid, const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int kbx, const int iqs) {
    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;
    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs = (const uint8_t *) &qs_packed;
    const int qh = bq3->qh[iqs/2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8 = (const uint8_t *) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint32_t g0 = grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)];
        const uint32_t g1 = grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)];
        const uint32_t signs = signs_packed_8[l0/2];
        const int grid_l = iq3_apply_signs(g0, signs & 15u);
        const int grid_h = iq3_apply_signs(g1, signs >> 4);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

static __device__ __forceinline__ float dot_iq3_xxs_lds(const uint32_t * __restrict__ grid, const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int kbx, const int iqs) {
    const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vbq + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs+1));
    const uint8_t * q3 = (const uint8_t *) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K/16 + iqs/2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint32_t g0 = grid[q3[l0 + 0]];
        const uint32_t g1 = grid[q3[l0 + 1]];
        const uint8_t sign_code = aux32 >> (7*l0/2);
        const uint32_t signs = sign_code ^ ((__popc(sign_code) & 1u) << 7);
        const int grid_l = iq3_apply_signs(g0, signs & 15u);
        const int grid_h = iq3_apply_signs(g1, signs >> 4);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

static __device__ __forceinline__ float dot_iq2_s_lds(const uint32_t * __restrict__ grid, const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int kbx, const int iqs) {
    const block_iq2_s * bq2 = (const block_iq2_s *) vbq + kbx;
    const int       qs_packed = get_int_b2(bq2->qs, iqs/2);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;
    const int qh = bq2->qh[iqs/2];
    const int       signs_packed_32 = get_int_b2(bq2->qs, QK_K/32 + iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;
    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int gi = (qs[l0/2] | ((qh << (8-l0)) & 0x300));
        const int g0 = grid[2*gi+0], g1 = grid[2*gi+1];
        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(g0 ^ signs0, signs0);
        const int grid_h = __vsub4(g1 ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        if (l0 < 4) { sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0); sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0); }
        else        { sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1); sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1); }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

static __device__ __forceinline__ float dot_iq2_xs_lds(const uint32_t * __restrict__ grid, const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int kbx, const int iqs) {
    const block_iq2_xs * bq2 = (const block_iq2_xs *) vbq + kbx;
    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t * q2 = (const uint16_t *) &q2_packed;
    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int gi = q2[l0/2] & 0x1FF;
        const uint32_t gx = grid[2*gi+0], gy = grid[2*gi+1];
        const uint32_t signs = unpack_ksigns(q2[l0/2] >> 9);
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(gx ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(gy ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        if (l0 < 4) { sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0); sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0); }
        else        { sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1); sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1); }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

static __device__ __forceinline__ float dot_iq2_xxs_lds(const uint32_t * __restrict__ grid, const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int kbx, const int iqs) {
    const block_iq2_xxs * bq2 = (const block_iq2_xxs *) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t * aux8 = (const uint8_t *) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const int gi = aux8[k0/2];
        const uint32_t gx = grid[2*gi+0], gy = grid[2*gi+1];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(gx ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(gy ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

template <ggml_type t> struct LT { static constexpr int words = 0; };
template <> struct LT<GGML_TYPE_IQ3_S>  { static constexpr int words = 512;  static __device__ const uint32_t * tab() { return (const uint32_t *) iq3s_grid; }  static __device__ __forceinline__ float dot(const uint32_t * g, const void * a, const block_q8_1 * b, int k, int i) { return dot_iq3_s_lds(g,a,b,k,i);} };
template <> struct LT<GGML_TYPE_IQ3_XXS>{ static constexpr int words = 256;  static __device__ const uint32_t * tab() { return (const uint32_t *) iq3xxs_grid; } static __device__ __forceinline__ float dot(const uint32_t * g, const void * a, const block_q8_1 * b, int k, int i) { return dot_iq3_xxs_lds(g,a,b,k,i);} };
template <> struct LT<GGML_TYPE_IQ2_S>  { static constexpr int words = 2048; static __device__ const uint32_t * tab() { return (const uint32_t *) iq2s_grid; }  static __device__ __forceinline__ float dot(const uint32_t * g, const void * a, const block_q8_1 * b, int k, int i) { return dot_iq2_s_lds(g,a,b,k,i);} };
template <> struct LT<GGML_TYPE_IQ2_XS> { static constexpr int words = 1024; static __device__ const uint32_t * tab() { return (const uint32_t *) iq2xs_grid; } static __device__ __forceinline__ float dot(const uint32_t * g, const void * a, const block_q8_1 * b, int k, int i) { return dot_iq2_xs_lds(g,a,b,k,i);} };
template <> struct LT<GGML_TYPE_IQ2_XXS>{ static constexpr int words = 512;  static __device__ const uint32_t * tab() { return (const uint32_t *) iq2xxs_grid; } static __device__ __forceinline__ float dot(const uint32_t * g, const void * a, const block_q8_1 * b, int k, int i) { return dot_iq2_xxs_lds(g,a,b,k,i);} };

// V2: like V1 but codebook in LDS
template <ggml_type type, int L, int R, int NW>
__launch_bounds__(NW*64, 1)
__global__ void gemv_v2(const void * __restrict__ vx, const block_q8_1 * __restrict__ y, float * __restrict__ dst, int K, int N) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk, qi = ggml_cuda_type_traits<type>::qi, vdr = TT<type>::vdr;
    __shared__ uint32_t sgrid[LT<type>::words];
    {
        const uint32_t * g = LT<type>::tab();
        for (int i = threadIdx.y*64 + threadIdx.x; i < LT<type>::words; i += NW*64) sgrid[i] = g[i];
        __syncthreads();
    }
    constexpr int G = 64/L;
    const int lane = threadIdx.x % L, grp = threadIdx.x / L;
    const int wave = blockIdx.x*NW + threadIdx.y;
    const int rowbase = (wave*G + grp)*R;
    if (rowbase >= N) return;
    const int bpr = K / qk;
    constexpr int cpb = qi/vdr;
    const int C = bpr*cpb;
    float tmp[R] = {0.0f};
    for (int t = lane; t < C; t += L) {
        const int kbx = t / cpb, kqs = vdr*(t % cpb);
        const int kby = kbx*(qk/QK8_1);
#pragma unroll
        for (int i = 0; i < R; ++i) tmp[i] += LT<type>::dot(sgrid, vx, &y[kby], (rowbase+i)*bpr + kbx, kqs);
    }
#pragma unroll
    for (int i = 0; i < R; ++i) {
        float v = warp_reduce_sum<L>(tmp[i]);
        if (lane == 0 && rowbase + i < N) dst[rowbase+i] = v;
    }
}

// ---- retained packed path (copied from mi210-iq3-retained.patch), non-fused ----
struct pk_meta { ggml_half d; uint8_t scales[4]; };
static __global__ void pk_pack_iq3s(const block_iq3_s * raw, uint32_t * codes, pk_meta * meta, size_t count) {
    const size_t gidx = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (gidx >= count*8) return;
    const size_t block = gidx / 8; const int group = gidx % 8;
    const auto & in = raw[block];
    const uint32_t high = in.qh[group];
    for (int pair = 0; pair < 4; pair++) {
        const int j = 2*pair; const auto signs = in.signs[group*4+pair];
        uint32_t packed = 0;
        for (int side = 0; side < 2; side++) {
            const uint32_t index = in.qs[group*8+j+side] | (((high>>(j+side))&1u)<<8);
            const uint32_t values = iq3s_grid[index];
            for (int lane = 0; lane < 4; lane++) {
                int value = (values>>(8*lane))&255;
                if ((signs>>(4*side+lane))&1) value = -value;
                packed |= uint32_t((value+15)/2)<<(8*lane+4*side);
            }
        }
        codes[gidx*4+pair] = packed;
    }
    if (group == 0) { meta[block].d = in.d; for (int i = 0; i < 4; i++) meta[block].scales[i] = in.scales[i]; }
}
static __global__ void pk_pack_iq3xxs(const block_iq3_xxs * raw, uint32_t * codes, pk_meta * meta, size_t count) {
    const size_t gidx = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (gidx >= count*8) return;
    const size_t block = gidx/8; const int group = gidx%8;
    const auto & in = raw[block];
    const uint8_t * a = in.qs+64+4*group;
    const uint32_t aux = uint32_t(a[0]) | uint32_t(a[1])<<8 | uint32_t(a[2])<<16 | uint32_t(a[3])<<24;
    for (int pair = 0; pair < 4; pair++) {
        const uint8_t s7 = aux>>(7*pair);
        const uint32_t signs = s7 ^ ((__popc(s7)&1u)<<7);
        uint32_t packed = 0;
        for (int side = 0; side < 2; side++) {
            const uint32_t v = iq3xxs_grid[in.qs[group*8+2*pair+side]];
            for (int lane = 0; lane < 4; lane++) {
                const uint32_t magnitude = (v>>(8*lane))&255u;
                const uint32_t index = (magnitude-4u)/8u + (((signs>>(4*side+lane))&1u)<<3);
                packed |= index<<(8*lane+4*side);
            }
        }
        codes[gidx*4+pair] = packed;
    }
    if (group == 0) { meta[block].d = in.d; for (int i = 0; i < 4; i++) meta[block].scales[i] = (in.qs[64+8*i+3]>>4) | (in.qs[64+8*i+7]&0xf0u); }
}
static __global__ void pk_sum_q8(const block_q8_1 * x, int * sums, int groups) {
    const int g = blockIdx.x*blockDim.x+threadIdx.x;
    if (g >= groups) return;
    int sum = 0;
    for (int j = 0; j < 8; j++) sum = ggml_cuda_dp4a(0x01010101,get_int_b4(x[g].qs,j),sum);
    sums[g] = sum;
}
static __device__ __forceinline__ float pk_block_dot(const uint32_t * codes, const pk_meta & m, const block_q8_1 & x, int sum_x, int group) {
    int value = 0;
    for (int j = 0; j < 4; j++) {
        const uint32_t pair = codes[j];
        value = ggml_cuda_dp4a(pair & 0x0f0f0f0fu,get_int_b4(x.qs,2*j),value);
        value = ggml_cuda_dp4a((pair>>4) & 0x0f0f0f0fu,get_int_b4(x.qs,2*j+1),value);
    }
    value = 2*value-15*sum_x;
    value *= 1+2*((m.scales[group/2]>>((group&1)*4))&15);
    const float d = __half2float(m.d) * __low2float(x.ds);
    return d*value;
}
alignas(4) static __device__ __constant__ int8_t pk_xxs_levels[16] = { 4, 12, 20, 28, 36, 44, 52, 62, -4,-12,-20,-28,-36,-44,-52,-62 };
static __device__ __forceinline__ float pk_xxs_block_dot(const uint32_t * codes, const pk_meta & m, const block_q8_1 & x, int group) {
    int value = 0;
    for (int j = 0; j < 4; j++) {
        const int2 v = get_int_from_table_16(codes[j],pk_xxs_levels);
        value = ggml_cuda_dp4a(v.x,get_int_b4(x.qs,2*j),value);
        value = ggml_cuda_dp4a(v.y,get_int_b4(x.qs,2*j+1),value);
    }
    const int ls = (m.scales[group/2]>>((group&1)*4))&15;
    value = (ls*value + value/2)/2;
    const float d = __half2float(m.d) * __low2float(x.ds);
    return d*value;
}
template<bool xxs>
static __global__ void pk_gemv(const uint32_t * up, const pk_meta * up_meta, const block_q8_1 * x, const int * sums, float * out, int groups) {
    const int row = blockIdx.x, lane = threadIdx.x;
    float result = 0.f;
    for (int g = lane; g < groups; g += 128) {
        const size_t pos = size_t(row)*groups+g;
        if constexpr (xxs) result += pk_xxs_block_dot(up+pos*4,up_meta[pos/8],x[g],g&7);
        else               result += pk_block_dot(up+pos*4,up_meta[pos/8],x[g],sums[g],g&7);
    }
    __shared__ float upper[64];
    if (lane >= 64) upper[lane-64] = result;
    __syncthreads();
    if (lane < 64) {
        result = warp_reduce_sum<64>(result+upper[lane]);
        if (lane == 0) out[row] = result;
    }
}
