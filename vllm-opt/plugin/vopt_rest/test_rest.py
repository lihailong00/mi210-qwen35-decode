"""Unit tests + graph-replay micro-benchmarks for torch.ops.vopt_rest.* (run in the vLLM image
with the GPU lock held).  usage: python3 -m vopt_rest.test_rest [--bench] [--only norms,act,...]

Correctness: each op against (a) the stock vLLM implementation it replaces (same inputs,
same states) and (b) an fp64 reference written from the vLLM code. Reports max |diff|,
bf16-ulp mismatch counts. Bench: CUDA-graph replay of NREP back-to-back calls, us/call."""
import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vopt_rest.ext import load_ext  # noqa: E402

DEV = "cuda"
RES = {}


def ulp_mismatch(a, b):
    """number of elements differing, and max difference in bf16 ulps"""
    ai = a.contiguous().view(torch.int16).to(torch.int32)
    bi = b.contiguous().view(torch.int16).to(torch.int32)
    # map sign-magnitude to monotonic ints
    ai = torch.where(ai < 0, -32768 - ai, ai)
    bi = torch.where(bi < 0, -32768 - bi, bi)
    d = (ai - bi).abs()
    return int((d > 0).sum()), int(d.max())


def graph_time(fn, nrep=50, warm=3):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warm):
            fn(0)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(nrep):
            fn(i)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(10):
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / nrep)
    ts.sort()
    return ts[len(ts) // 2]


# ------------------------------------------------------------------------------ norms
def native_gemma(x, w, eps, r=None):
    from vllm import ir
    wf = w.float() + 1.0
    if r is None:
        return ir.ops.rms_norm(x, wf, eps)
    return ir.ops.fused_add_rms_norm(x, r, wf, eps)


def test_norms(bench):
    out = {}
    eps = 1e-6
    for D in (5120, 256):
        for M in (1, 4, 8, 64, 24):
            x = (torch.randn(M, D, device=DEV) * 3).bfloat16()
            r = (torch.randn(M, D, device=DEV) * 10).bfloat16()
            w = (torch.randn(D, device=DEV) * 0.3).bfloat16()
            y = torch.ops.vopt_rest.gemma_rms_norm(x, w, eps)
            yn = native_gemma(x, w, eps)
            y2, r2 = torch.ops.vopt_rest.gemma_add_rms_norm(x, r, w, eps)
            y2n, r2n = native_gemma(x.clone(), w, eps, r.clone())
            # fp64 reference
            xd = x.double(); s = xd + r.double()
            ref1 = xd * torch.rsqrt((xd * xd).mean(-1, keepdim=True) + eps) * (w.double() + 1)
            ref2 = s * torch.rsqrt((s * s).mean(-1, keepdim=True) + eps) * (w.double() + 1)
            m1 = ulp_mismatch(y, yn); m2 = ulp_mismatch(y2, y2n); m3 = ulp_mismatch(r2, r2n)
            out[f"D{D}_M{M}"] = dict(
                rms_vs_native_mismatch=m1, add_vs_native_mismatch=m2, resid_vs_native_mismatch=m3,
                rms_maxerr_fp64=float((y.double() - ref1).abs().max()),
                native_maxerr_fp64=float((yn.double() - ref1).abs().max()),
                add_maxerr_fp64=float((y2.double() - ref2).abs().max()),
                add_native_maxerr_fp64=float((y2n.double() - ref2).abs().max()))
    if bench:
        for D, M in ((5120, 1), (5120, 4), (5120, 8)):
            xs = [(torch.randn(M, D, device=DEV)).bfloat16() for _ in range(8)]
            rs = [(torch.randn(M, D, device=DEV)).bfloat16() for _ in range(8)]
            w = torch.randn(D, device=DEV).bfloat16()
            t_new = graph_time(lambda i: torch.ops.vopt_rest.gemma_add_rms_norm(xs[i % 8], rs[i % 8], w, 1e-6))
            t_old = graph_time(lambda i: native_gemma(xs[i % 8], w, 1e-6, rs[i % 8]))
            out[f"bench_add_D{D}_M{M}_us"] = dict(new=round(t_new, 2), stock=round(t_old, 2))
    RES["norms"] = out


# ------------------------------------------------------------------------------ act
def test_act(bench):
    import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)
    out = {}
    for M in (1, 4, 8, 64):
        x = (torch.randn(M, 2 * 17408, device=DEV) * 3).bfloat16()
        y = torch.ops.vopt_rest.silu_mul(x)
        yn = F.silu(x[:, :17408]) * x[:, 17408:]
        torch_c = torch.empty_like(yn)
        torch.ops._C.silu_and_mul(torch_c, x)
        out[f"M{M}"] = dict(vs_native=ulp_mismatch(y, yn), vs_vllm_C=ulp_mismatch(y, torch_c))
    for M in (1, 8):
        a = torch.randn(M, 6144, device=DEV).bfloat16(); g = (torch.randn(M, 6144, device=DEV) * 4).bfloat16()
        out[f"sigmul_M{M}"] = ulp_mismatch(torch.ops.vopt_rest.sigmoid_mul(a, g), a * torch.sigmoid(g))
    if bench:
        xs = [torch.randn(1, 2 * 17408, device=DEV).bfloat16() for _ in range(8)]
        out["bench_silu_M1_us"] = dict(
            new=round(graph_time(lambda i: torch.ops.vopt_rest.silu_mul(xs[i % 8])), 2),
            stock=round(graph_time(lambda i: torch.ops._C.silu_and_mul(torch.empty(1, 17408, device=DEV, dtype=torch.bfloat16), xs[i % 8])), 2))
        a = [torch.randn(1, 6144, device=DEV).bfloat16() for _ in range(8)]
        out["bench_sigmul_M1_us"] = dict(
            new=round(graph_time(lambda i: torch.ops.vopt_rest.sigmoid_mul(a[i % 8], a[(i + 1) % 8])), 2),
            stock=round(graph_time(lambda i: a[i % 8] * torch.sigmoid(a[(i + 1) % 8])), 2))
    RES["act"] = out


# ------------------------------------------------------------------------------ gdn
class GdnCase:
    def __init__(self, M, slots=None, nlayers=1, null_rows=()):
        slots = slots or max(64, M + 2)
        g = torch.Generator(device=DEV).manual_seed(1234 + M)
        self.M = M
        self.qkvz = (torch.randn(M, 16384, device=DEV, generator=g) * 2).bfloat16()
        self.ba = (torch.randn(M, 96, device=DEV, generator=g) * 2).bfloat16()
        # vLLM SD layout: (slots, state_len, dim)
        self.conv = [(torch.randn(slots, 3, 10240, device=DEV, generator=g)).bfloat16() for _ in range(nlayers)]
        self.ssm = [(torch.randn(slots, 48, 128, 128, device=DEV, generator=g) * 0.05) for _ in range(nlayers)]
        self.cw = (torch.randn(10240, 1, 4, device=DEV, generator=g) * 0.5).bfloat16()
        self.A_log = (torch.rand(48, device=DEV, generator=g) * 2 - 1).float()
        self.dt_bias = (torch.randn(48, device=DEV, generator=g)).bfloat16()
        self.norm_w = (torch.rand(128, device=DEV, generator=g) + 0.5).bfloat16()
        perm = torch.randperm(slots - 1, device=DEV, generator=g)[:M] + 1
        for i in null_rows:
            perm[i] = 0
        self.idx = perm.to(torch.int32)


def gdn_stock(c, conv, ssm):
    """stock vLLM path: zeros + causal_conv1d_update + packed decode + RMSNormGated (Triton)"""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import \
        fused_recurrent_gated_delta_rule_packed_decode
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn
    M = c.M
    qkvz = c.qkvz.clone()
    mixed_qkv, z = qkvz.split([10240, 6144], dim=-1)
    z = z.reshape(M, -1, 128)
    b, a = c.ba.chunk(2, dim=-1)
    core = torch.zeros(M, 48, 128, dtype=torch.bfloat16, device=DEV)
    mq = causal_conv1d_update(mixed_qkv, conv.transpose(-1, -2), c.cw.view(10240, 4), None, "silu",
                              conv_state_indices=c.idx, validate_data=False)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mq, a=a.contiguous(), b=b.contiguous(), A_log=c.A_log, dt_bias=c.dt_bias,
        scale=128 ** -0.5, initial_state=ssm, out=core.unsqueeze(1), ssm_state_indices=c.idx,
        use_qk_l2norm_in_kernel=True)
    y = rmsnorm_fn(core.reshape(-1, 128), c.norm_w, None, z=z.reshape(-1, 128), eps=1e-6,
                   group_size=None, norm_before_gate=True, activation="silu")
    return y.reshape(M, 6144)


_SCR = None


def gdn_new(c, conv, ssm, out=None, dbg=0):
    global _SCR
    if _SCR is None:
        _SCR = (torch.zeros(512 * 16, dtype=torch.int32, device=DEV),)
    M = c.M
    if out is None:
        out = torch.empty(M, 6144, dtype=torch.bfloat16, device=DEV)
    qkvz = c.qkvz
    torch.ops.vopt_rest.gdn_decode(qkvz, qkvz[:, 10240:], c.ba[:, :48], c.ba[:, 48:], conv,
                                   conv.stride(0), conv.stride(2), conv.stride(1),
                                   c.cw.view(10240, 4), c.A_log, c.dt_bias, c.norm_w, ssm, c.idx, out,
                                   *_SCR, 128 ** -0.5, 1e-6, dbg)
    return out


def gdn_ref64(c, conv, ssm):
    """fp64 reference from the vLLM semantics (bf16 rounding at the same storage points)"""
    M = c.M
    rb = lambda t: t.to(torch.bfloat16).double()
    outs = []
    conv = conv.clone(); ssm = ssm.clone()
    for t in range(M):
        i = int(c.idx[t])
        if i <= 0:
            outs.append(torch.zeros(6144, dtype=torch.float64, device=DEV)); continue
        x = c.qkvz[t, :10240].double()
        win = torch.cat([conv[i].double(), x[None]], 0)  # [4, 10240]
        w = c.cw.view(10240, 4).double().T  # [4, 10240]
        acc = rb(win * w).sum(0)
        acc = rb(acc / (1 + torch.exp(-acc)))
        q = acc[:2048].view(16, 128); k = acc[2048:4096].view(16, 128); v = acc[4096:].view(48, 128)
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * 128 ** -0.5
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        q = q.repeat_interleave(3, 0); k = k.repeat_interleave(3, 0)
        a = c.ba[t, 48:].double(); b = c.ba[t, :48].double()
        xg = a + c.dt_bias.double()
        sp = torch.where(xg <= 20, torch.log1p(torch.exp(xg)), xg)
        g = -torch.exp(c.A_log.double()) * sp
        beta = torch.sigmoid(b)
        h = ssm[i].double() * torch.exp(g)[:, None, None]  # [48, V, K]
        vv = (v - (h * k[:, None, :]).sum(-1)) * beta[:, None]
        h = h + vv[:, :, None] * k[:, None, :]
        o = rb((h * q[:, None, :]).sum(-1))  # [48, 128]
        rstd = torch.rsqrt((o * o).mean(-1, keepdim=True) + 1e-6)
        zz = c.qkvz[t, 10240:].double().view(48, 128)
        y = o * rstd * c.norm_w.double() * zz * torch.sigmoid(zz)
        outs.append(y.reshape(-1))
        ssm[i] = h.float()
        conv[i] = torch.cat([conv[i][1:], c.qkvz[t, :10240][None]], 0)
    return torch.stack(outs), conv, ssm


def test_gdn(bench):
    out = {}
    for M, nulls in ((1, ()), (2, ()), (4, (3,)), (8, (0, 5)), (64, (7,))):
        c = GdnCase(M, null_rows=nulls)
        conv_a, ssm_a = c.conv[0].clone(), c.ssm[0].clone()
        conv_b, ssm_b = c.conv[0].clone(), c.ssm[0].clone()
        y_s = gdn_stock(c, conv_a, ssm_a)
        y_n = gdn_new(c, conv_b, ssm_b)
        torch.cuda.synchronize()
        r = dict(out_vs_stock=ulp_mismatch(y_n, y_s),
                 out_maxabs_vs_stock=float((y_n.float() - y_s.float()).abs().max()),
                 conv_state_equal=bool(torch.equal(conv_a, conv_b)),
                 ssm_maxabs_vs_stock=float((ssm_a - ssm_b).abs().max()),
                 ssm_absmax=float(ssm_a.abs().max()))
        if M <= 8:
            y_r, conv_r, ssm_r = gdn_ref64(c, c.conv[0], c.ssm[0])
            r.update(new_maxabs_fp64=float((y_n.double() - y_r).abs().max()),
                     stock_maxabs_fp64=float((y_s.double() - y_r).abs().max()),
                     new_ssm_maxabs_fp64=float((ssm_b.double() - ssm_r.double()).abs().max()),
                     stock_ssm_maxabs_fp64=float((ssm_a.double() - ssm_r.double()).abs().max()),
                     conv_vs_ref_equal=bool(torch.equal(conv_b, conv_r)),
                     out_absmax=float(y_r.abs().max()))
        # second step on the updated states (counters must have reset)
        y_s2 = gdn_stock(c, conv_a, ssm_a); y_n2 = gdn_new(c, conv_b, ssm_b)
        r["step2_out_maxabs_vs_stock"] = float((y_n2.float() - y_s2.float()).abs().max())
        r["step2_ssm_maxabs_vs_stock"] = float((ssm_a - ssm_b).abs().max())
        out[f"M{M}"] = r
    if bench:
        for M in (1, 4, 8):
            NL = 12  # rotate over 12 layers' states (37 MB) to defeat the 8 MB L2
            c = GdnCase(M, slots=M + 2, nlayers=NL)
            t_new = graph_time(lambda i: gdn_new(c, c.conv[i % NL], c.ssm[i % NL]), nrep=48)
            t_old = graph_time(lambda i: gdn_stock(c, c.conv[i % NL], c.ssm[i % NL]), nrep=48)
            t_nold = graph_time(lambda i: gdn_new(c, c.conv[i % NL], c.ssm[i % NL], dbg=1), nrep=48)
            t_nost = graph_time(lambda i: gdn_new(c, c.conv[i % NL], c.ssm[i % NL], dbg=2), nrep=48)
            t_none = graph_time(lambda i: gdn_new(c, c.conv[i % NL], c.ssm[i % NL], dbg=3), nrep=48)
            t_same = graph_time(lambda i: gdn_new(c, c.conv[0], c.ssm[0]), nrep=48)
            out[f"bench_M{M}_dbg_us"] = dict(no_state_load=round(t_nold, 2), no_state_store=round(t_nost, 2),
                                            no_state_io=round(t_none, 2), same_layer_L2=round(t_same, 2))
            out[f"bench_M{M}_us"] = dict(new=round(t_new, 2), stock_4kernels=round(t_old, 2),
                                         floor_us=round(M * 48 * 128 * 128 * 4 * 2 / 1.446e6, 2))
    RES["gdn"] = out


# ------------------------------------------------------------------------------ fused qk rope
def test_qk(bench):
    from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
    out = {}
    hd, nq, nk, rd = 256, 24, 4, 64
    base = 1e7
    inv = 1.0 / (base ** (torch.arange(0, rd, 2, device=DEV, dtype=torch.float32) / rd))
    pos_all = torch.arange(8192, device=DEV, dtype=torch.float32)
    fr = torch.outer(pos_all, inv)
    cache = torch.cat([fr.cos(), fr.sin()], -1).bfloat16()  # vLLM keeps cos_sin_cache in model dtype
    for M in (1, 4):
        qkv = (torch.randn(M, nq * 2 * hd + 2 * nk * hd, device=DEV) * 3).bfloat16()
        qw = (torch.randn(hd, device=DEV) * 0.2).bfloat16(); kw = (torch.randn(hd, device=DEV) * 0.2).bfloat16()
        pos = torch.randint(0, 8000, (M,), device=DEV)
        q_gate, k, v = qkv.split([nq * 2 * hd, nk * hd, nk * hd], dim=-1)
        q, kk, gate = fused_qk_rmsnorm_rope_gate(q_gate, k, qw, kw, cache, pos, 1e-6, nq, nk, hd, rd,
                                                 norm_beta=1.0)
        # reference: native split + GemmaRMSNorm + neox partial rope (bf16 storage between)
        qg = q_gate.view(M, nq, 2 * hd)
        qr, gr = qg[..., :hd], qg[..., hd:]
        qn = native_gemma(qr.reshape(-1, hd).contiguous(), qw, 1e-6).view(M, nq, hd)
        kn = native_gemma(k.reshape(-1, hd).contiguous(), kw, 1e-6).view(M, nk, hd)

        def rope(x):
            cs = cache[pos].float()
            cos, sin = cs[:, None, :rd // 2], cs[:, None, rd // 2:]
            x1, x2 = x[..., :rd // 2].float(), x[..., rd // 2:rd].float()
            o1 = x1 * cos - x2 * sin; o2 = x2 * cos + x1 * sin
            return torch.cat([o1.bfloat16(), o2.bfloat16(), x[..., rd:]], -1)
        out[f"M{M}"] = dict(q=ulp_mismatch(q.view(M, nq, hd), rope(qn)), k=ulp_mismatch(kk.view(M, nk, hd), rope(kn)),
                            gate_equal=bool(torch.equal(gate.view(M, nq, hd), gr)))
    RES["qk"] = out


def test_noop(bench):
    x = torch.zeros(1, device=DEV)
    out = {}
    for nb, nt in ((1, 64), (1, 1024), (48, 1024), (104, 1024), (208, 256), (1024, 256)):
        out[f"noop_{nb}x{nt}_us"] = round(graph_time(lambda i: torch.ops.vopt_rest.noop(x, nb, nt), nrep=100), 2)
    xs = [torch.randn(1, 5120, device=DEV).bfloat16() for _ in range(8)]
    out["copy_5120_us"] = round(graph_time(lambda i: xs[(i + 1) % 8].copy_(xs[i % 8]), nrep=100), 2)
    RES["noop"] = out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--only", default="norms,act,gdn,qk")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    load_ext()
    torch.manual_seed(0)
    for n in a.only.split(","):
        globals()[f"test_{n}"](a.bench)
        print(n, json.dumps(RES[n], indent=1), flush=True)
    if a.out:
        json.dump(RES, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
