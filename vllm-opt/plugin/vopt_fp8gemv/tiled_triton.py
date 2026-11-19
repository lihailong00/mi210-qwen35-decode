"""Large-M fallback on the tiled weight layout [N/64][K/16][64][16].

Same math as the reference overlay's _mi210_exact_fp8_block_scaled_mm (dynamic fp8 activation
quant, exact e4m3 -> fp16 decode, fp16 tl.dot, fp32 accumulation with block scales); only the
weight addressing differs. Tile configs: the reference table (_mi210_exact_fp8_config) when the
overlay is present, else the same defaults.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _e4m3_to_fp16(u):
    u16 = u.to(tl.uint16)
    au = u16 & 0x7F
    normal_bits = ((au << 7) + 0x2000).to(tl.uint16)
    sub = au.to(tl.float16) * tl.full((), 0.001953125, tl.float16)
    bits = tl.where(au < 8, sub.to(tl.uint16, bitcast=True), normal_bits).to(tl.uint16)
    bits = tl.where(au == 0x7F, tl.full((), 0x7E00, tl.uint16), bits).to(tl.uint16)
    bits = bits | ((u16 & 0x80) << 8)
    return bits.to(tl.float16, bitcast=True)


@triton.jit
def _tiled_fp8_block_scaled_mm(
    A, B, C, As, Bs,
    M, N, K,
    group_n, group_k,
    stride_am, stride_ak,
    stride_cm, stride_cn,
    stride_As_m, stride_As_k,
    stride_Bs_k, stride_Bs_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    PERMUTE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # tiled layout: the [BK x 64] slab of 64-row group g at k0 is BK*64 contiguous bytes
    # ((k0/16 .. ) tiles of 1 KB); load G = BN/64 such slabs as a [G, BK*64] block and
    # permute [G, BK/16, 64, 16] -> [BK/16, 16, G, 64] -> [BK, BN] in registers.
    G: tl.constexpr = BLOCK_SIZE_N // 64
    if PERMUTE:
        offs_g = (pid_n * G + tl.arange(0, G)) % (N // 64)
        b_ptrs = B + (offs_g.to(tl.int64)[:, None] * (64 * K) + tl.arange(0, BLOCK_SIZE_K * 64)[None, :])
    else:
        # direct [BK, BN] gather: 16 contiguous bytes per (n, 16-k chunk)
        b_k = (offs_k // 16) * 1024 + (offs_k % 16)
        b_k = tl.max_contiguous(tl.multiple_of(b_k, 16), 16)
        b_n = (offs_bn // 64).to(tl.int64) * (64 * K) + (offs_bn % 64) * 16
        b_n = tl.multiple_of(b_n, 16)
        b_ptrs = B + (b_k[:, None] + b_n[None, :])

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_u = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)
        if PERMUTE:
            b_raw = tl.load(b_ptrs)
            b_u = tl.reshape(tl.permute(tl.reshape(b_raw, (G, BLOCK_SIZE_K // 16, 64, 16)), (1, 3, 0, 2)),
                             (BLOCK_SIZE_K, BLOCK_SIZE_N))
        else:
            b_u = tl.load(b_ptrs)
        a = _e4m3_to_fp16(a_u)
        b = _e4m3_to_fp16(b_u)
        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        if PERMUTE:
            b_ptrs += BLOCK_SIZE_K * 64
        else:
            b_ptrs += (BLOCK_SIZE_K // 16) * 1024

    c = accumulator.to(C.dtype.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def _c(bm, bn, bk, nw=4, st=2, gm=32, perm=0, **kw):
    d = {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": gm,
         "num_warps": nw, "num_stages": st}
    d.update(kw)
    return d, perm


# (N, K) -> [(max M, (config, permute)), ...], tuned with fallback_tune.py (runs/ft*.log)
TABLE = {
    (34816, 5120): [(16, _c(16, 128, 128, st=1, gm=1, perm=1)), (64, _c(32, 256, 64)), (1 << 30, _c(64, 256, 64))],
    (16384, 5120): [(16, _c(16, 64, 64)), (64, _c(32, 128, 128)), (1 << 30, _c(64, 256, 64))],
    (14336, 5120): [(16, _c(16, 64, 64)), (64, _c(32, 128, 128)), (1 << 30, _c(64, 256, 64))],
    (5120, 6144): [(16, _c(16, 64, 64, waves_per_eu=2)), (64, _c(32, 128, 128, perm=1)), (1 << 30, _c(128, 256, 64, nw=8, perm=1))],
    (5120, 17408): [(16, _c(16, 64, 64, waves_per_eu=1)), (64, _c(32, 128, 128, perm=1)), (1 << 30, _c(128, 256, 64, nw=8, perm=1))],
}


def _config(M, N, K):
    t = TABLE.get((N, K))
    if t:
        for mmax, (cfg, perm) in t:
            if M <= mmax:
                return dict(cfg), perm
    try:
        from vllm.model_executor.layers.quantization.utils import fp8_utils
        f = getattr(fp8_utils, "_mi210_exact_fp8_config", None)
        if f is not None:
            return f(M, N, K), 0
    except Exception:
        pass
    return {"BLOCK_SIZE_M": 16 if M <= 32 else 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 32, "num_warps": 4, "num_stages": 2}, 0


def block_scaled_mm_tiled(A, Bt, As, Bs, output_dtype=torch.bfloat16, cfg=None, permute=None):
    """A fp8 [.., K] (dynamic per-token-group quantized), As fp32 [.., K/128],
    Bt tiled weight [N, K] (fp8 or uint8 view), Bs fp32 [N/128, K/128]."""
    assert A.is_contiguous()
    K = A.shape[-1]
    M = A.numel() // K
    N = Bt.shape[0]
    assert Bt.shape[1] == K and K % 128 == 0 and N % 64 == 0
    C = A.new_empty(A.shape[:-1] + (N,), dtype=output_dtype)
    if cfg is None:
        cfg, perm0 = _config(M, N, K)
        if permute is None:
            permute = perm0
    cfg = dict(cfg)
    assert cfg["BLOCK_SIZE_K"] % 16 == 0 and cfg["BLOCK_SIZE_N"] % 64 == 0 and K % cfg["BLOCK_SIZE_K"] == 0
    As2 = As.reshape(-1, As.shape[-1])

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    _tiled_fp8_block_scaled_mm[grid](
        A.view(torch.uint8), Bt.view(torch.uint8), C, As2, Bs,
        M, N, K, 128, 128,
        A.stride(-2) if A.dim() > 1 else K, A.stride(-1),
        C.stride(-2) if C.dim() > 1 else N, C.stride(-1),
        As2.stride(-2), As2.stride(-1), Bs.stride(1), Bs.stride(0),
        PERMUTE=int(permute or 0),
        **cfg,
    )
    return C
