"""Draft-only lm_head copies for the vopt_spec draft head (built once at load time, on the GPU)."""
import torch

GQ = 128  # int4 group (columns)


def quant_int4(w, rows, chunk=8192):
    """w: bf16 [V, 5120] (the lm_head); rows: int64 [N] vocab ids of the subset.
    Returns (q int32 [ceil(N/2), 320, 4], ab int32 [ceil(N/2), 40, 2]) in the i4_head_argmax layout:
    asymmetric 4-bit codes per 128-column group, code n -> f16 1 + n/16, a = 16*scale and
    b = min - 16*scale as half2 (so w ~= a * (1 + n/16) + b = scale * n + min)."""
    N = rows.numel()
    K = w.shape[1]
    assert K % GQ == 0 and K % 16 == 0
    P = (N + 1) // 2
    q_out = torch.empty(P, K // 16, 4, dtype=torch.int32, device=w.device)
    ab_out = torch.empty(P, K // GQ, 2, dtype=torch.int32, device=w.device)
    sh = (torch.arange(8, device=w.device, dtype=torch.int64) * 4)
    for s in range(0, P, chunk // 2):
        e = min(P, s + chunk // 2)
        r = rows[2 * s: min(N, 2 * e)]
        wf = w[r].float()
        if wf.shape[0] < 2 * (e - s):  # odd N: zero pad row
            wf = torch.cat([wf, wf.new_zeros(2 * (e - s) - wf.shape[0], K)])
        n = wf.shape[0]
        wg = wf.view(n, K // GQ, GQ)
        lo = wg.amin(-1, keepdim=True)
        hi = wg.amax(-1, keepdim=True)
        sc = ((hi - lo) / 15).clamp_min(1e-10)
        code = ((wg - lo) / sc).round_().clamp_(0, 15).to(torch.int64).view(n, K // 8, 8)
        words = (code << sh).sum(-1)                       # [n, K/8] uint32 values in int64
        words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
        a = (16 * sc).squeeze(-1).half().view(torch.int16).to(torch.int32) & 0xFFFF
        b = (lo - 16 * sc).squeeze(-1).half().view(torch.int16).to(torch.int32) & 0xFFFF
        abw = (a.to(torch.int64) | (b.to(torch.int64) << 16))
        abw = torch.where(abw >= 2 ** 31, abw - 2 ** 32, abw).to(torch.int32)  # [n, 40]
        words = words.view(e - s, 2, K // 16, 2)           # pair, row-in-pair, chunk, word-in-chunk
        q_out[s:e] = words.permute(0, 2, 1, 3).reshape(e - s, K // 16, 4)
        ab_out[s:e] = abw.view(e - s, 2, K // GQ).permute(0, 2, 1)
    return q_out, ab_out


def dequant_int4(q, ab, N, K=5120):
    """Reference dequantization (fp64) of the int4 layout -> [N, K] (for tests)."""
    P = q.shape[0]
    words = q.view(P, K // 16, 2, 2).permute(0, 2, 1, 3).reshape(2 * P, K // 8).to(torch.int64) & 0xFFFFFFFF
    sh = torch.arange(8, device=q.device, dtype=torch.int64) * 4
    code = ((words[..., None] >> sh) & 15).reshape(2 * P, K).double()
    abw = ab.permute(0, 2, 1).reshape(2 * P, K // GQ).to(torch.int64) & 0xFFFFFFFF
    a = (abw & 0xFFFF).to(torch.int16).view(torch.float16).double()
    b = ((abw >> 16) & 0xFFFF).to(torch.int16).view(torch.float16).double()
    h = 1 + code / 16
    w = a.repeat_interleave(GQ, 1) * h + b.repeat_interleave(GQ, 1)
    return w[:N]
