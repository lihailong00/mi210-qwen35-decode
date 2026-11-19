"""MTP verify-path tests for vopt_rest (run in the vLLM image with the GPU lock held).
usage: python3 -m vopt_rest.test_spec [--bench] [--only gdn,attn,ops,graph]

gdn  : torch.ops.vopt_rest.gdn_spec (R requests x T tokens, vLLM spec slot contract) vs the deployed
       1-token kernel gdn_decode run token by token from the same committed state, over several
       rounds with random acceptance (rollback). Bitwise: every output row, every written state slot,
       the conv-state slot. Also vs an fp64 reference and vs the stock Triton spec path (info).
attn : pseudo-sequence verify attention (attn_pseudo_meta + stock unified_attention, 3D path) vs
       the 1-token decode attention per query, bitwise, contexts 40..6000, block sizes 16 / 800.
ops  : per-token ops at T = 1..8 rows vs 1-row calls, bitwise: gemma_rms(+add), silu_mul,
       sigmoid_mul, fused qk-norm/rope/gate (Triton), reshape_and_cache (Triton).
graph: gdn_spec + attention metadata under CUDA-graph capture/replay with changing metadata.
bench: graph-replay timing of gdn_spec vs the stock spec path and the 1-token kernel; verify
       attention 2D (stock) vs pseudo-sequence 3D.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vopt_rest.ext import load_ext  # noqa: E402

DEV = "cuda"
RES = {}
H, HV, D = 16, 48, 128


def nmis(a, b):
    return int((a.contiguous().view(torch.int16) != b.contiguous().view(torch.int16)).sum()) \
        if a.dtype == torch.bfloat16 else int((a != b).sum())


def graph_time(fn, nrep=48, warm=3):
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


# ------------------------------------------------------------------------------ gdn
class Layer:
    def __init__(self, g, slots, K):
        self.cw = (torch.randn(10240, 1, 4, device=DEV, generator=g) * 0.5).bfloat16()
        self.A_log = (torch.rand(48, device=DEV, generator=g) * 2 - 1).float()
        self.dt_bias = torch.randn(48, device=DEV, generator=g).bfloat16()
        self.norm_w = (torch.rand(128, device=DEV, generator=g) + 0.5).bfloat16()
        # vLLM SD conv layout (slots, state_len, dim); MTP: state_len = 3 + K
        self.conv = torch.randn(slots, 3 + K, 10240, device=DEV, generator=g).bfloat16()
        self.ssm = torch.randn(slots, 48, 128, 128, device=DEV, generator=g) * 0.05


_SCR = {}


def scratch():
    if "c" not in _SCR:
        _SCR["c"] = torch.zeros(512 * 16, dtype=torch.int32, device=DEV)
    return _SCR["c"]


def run_spec(L, qkvz, ba, sidx, nacc, qsl, out=None, dbg=0):
    n = qkvz.size(0)
    if out is None:
        out = torch.empty(n, 6144, dtype=torch.bfloat16, device=DEV)
    cs = L.conv
    torch.ops.vopt_rest.gdn_spec(qkvz, qkvz[:, 10240:], ba[:, :48], ba[:, 48:], cs, cs.stride(0),
                                 cs.stride(2), cs.stride(1), cs.size(1), L.cw.view(10240, 4),
                                 L.A_log, L.dt_bias, L.norm_w, L.ssm, sidx, nacc, qsl, out, scratch(),
                                 128 ** -0.5, 1e-6, dbg)
    return out


def run_m1(L, conv3, ssm, slot, x, bax):
    """deployed 1-token kernel on one token; conv3 is a 3-column (K=0) conv state tensor"""
    out = torch.empty(1, 6144, dtype=torch.bfloat16, device=DEV)
    idx = torch.tensor([slot], dtype=torch.int32, device=DEV)
    torch.ops.vopt_rest.gdn_decode(x, x[:, 10240:], bax[:, :48], bax[:, 48:], conv3, conv3.stride(0),
                                   conv3.stride(2), conv3.stride(1), L.cw.view(10240, 4), L.A_log,
                                   L.dt_bias, L.norm_w, ssm, idx, out, scratch(), 128 ** -0.5, 1e-6)
    return out


def ref64_token(L, win, h, x, bax):
    """fp64 reference for one token: win [3,10240] conv window (bf16), h [48,128,128] state"""
    rb = lambda t: t.to(torch.bfloat16).double()
    xx = x[0, :10240]
    w = L.cw.view(10240, 4).double().T
    acc = rb(torch.cat([win.double(), xx.double()[None]], 0) * w).sum(0)
    acc = rb(acc / (1 + torch.exp(-acc)))
    q = acc[:2048].view(16, 128); k = acc[2048:4096].view(16, 128); v = acc[4096:].view(48, 128)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * 128 ** -0.5
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(3, 0); k = k.repeat_interleave(3, 0)
    a = bax[0, 48:].double(); b = bax[0, :48].double()
    xg = a + L.dt_bias.double()
    sp = torch.where(xg <= 20, torch.log1p(torch.exp(xg)), xg)
    g = -torch.exp(L.A_log.double()) * sp
    hh = h.double() * torch.exp(g)[:, None, None]
    vv = (v - (hh * k[:, None, :]).sum(-1)) * torch.sigmoid(b)[:, None]
    hh = hh + vv[:, :, None] * k[:, None, :]
    o = rb((hh * q[:, None, :]).sum(-1))
    rstd = torch.rsqrt((o * o).mean(-1, keepdim=True) + 1e-6)
    zz = x[0, 10240:].double().view(48, 128)
    return (o * rstd * L.norm_w.double() * zz * torch.sigmoid(zz)).reshape(-1), hh


def stock_spec(L, qkvz, ba, sidx, nacc, qsl):
    """stock vLLM spec path (causal_conv1d_update spec + fused_sigmoid_gating + RMSNormGated)"""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import \
        fused_sigmoid_gating_delta_rule_update
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn
    n = qkvz.size(0)
    mixed_qkv, z = qkvz.clone().split([10240, 6144], dim=-1)
    b, a = ba.chunk(2, dim=-1)
    nb = sidx.size(0)
    mq = causal_conv1d_update(mixed_qkv, L.conv.transpose(-1, -2), L.cw.view(10240, 4), None, "silu",
                              conv_state_indices=sidx[:, 0], num_accepted_tokens=nacc,
                              query_start_loc=qsl, max_query_len=sidx.size(-1), validate_data=False)
    q, k, v = torch.split(mq, [2048, 2048, 6144], dim=-1)
    q = q.view(1, n, 16, 128); k = k.view(1, n, 16, 128); v = v.view(1, n, 48, 128)
    core, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=L.A_log, a=a.contiguous(), b=b.contiguous(), dt_bias=L.dt_bias, q=q, k=k, v=v,
        initial_state=L.ssm, inplace_final_state=True, cu_seqlens=qsl[:nb + 1], ssm_state_indices=sidx,
        num_accepted_tokens=nacc, use_qk_l2norm_in_kernel=True, scale=128 ** -0.5)
    y = rmsnorm_fn(core.reshape(-1, 128), L.norm_w, None, z=z.reshape(-1, 128), eps=1e-6,
                   group_size=None, norm_before_gate=True, activation="silu")
    return y.reshape(n, 6144)


def gdn_rounds(R, T, rounds, seed, pad_req=0, pad_tok=0, check64=False):
    """R real requests (+pad_req padded ones with NULL slots, + pad_tok tail tokens), T = 1+K tokens.
    Returns mismatch counts accumulated over the rounds."""
    K = T - 1
    g = torch.Generator(device=DEV).manual_seed(seed)
    slots = 1 + (R + pad_req) * T + 4
    L = Layer(g, slots, K)
    nb = R + pad_req
    sidx = torch.zeros(nb, T, dtype=torch.int32, device=DEV)
    perm = torch.randperm(slots - 1, generator=torch.Generator().manual_seed(seed))[:R * T] + 1
    sidx[:R] = perm.view(R, T).to(torch.int32).to(DEV)
    # reference state per request (K=0 deployment: 3-column conv, one ssm slot)
    ref_conv = torch.zeros(R + 1, 3, 10240, dtype=torch.bfloat16, device=DEV)
    ref_ssm = torch.zeros(R + 1, 48, 128, 128, dtype=torch.float32, device=DEV)
    nacc_h = [1] * nb
    for i in range(R):
        ref_conv[i + 1] = L.conv[sidx[i, 0], 0:3]
        ref_ssm[i + 1] = L.ssm[sidx[i, 0]]
    st = dict(out=0, ssm=0, conv=0, pad_nonzero=0, rows=0, max64=0.0, ssm_max64=0.0,
              stock_out_mis=0, stock_out_maxabs=0.0)
    for rd in range(rounds):
        n = nb * T + pad_tok
        qkvz = (torch.randn(n, 16384, device=DEV, generator=g) * 2).bfloat16()
        ba = (torch.randn(n, 96, device=DEV, generator=g) * 2).bfloat16()
        qsl_l = [min(i, R) * T for i in range(nb + 1)]  # padded requests: zero-length
        qsl = torch.tensor(qsl_l, dtype=torch.int32, device=DEV)
        nacc = torch.tensor(nacc_h, dtype=torch.int32, device=DEV)
        conv_before = L.conv.clone()
        ssm_snap = {int(s): L.ssm[int(s)].clone() for s in sidx[:R].flatten().tolist()}
        if rd == 0 and check64:
            ssm_keep = L.ssm.clone()
        out = torch.full((n, 6144), 7.0, dtype=torch.bfloat16, device=DEV)
        run_spec(L, qkvz, ba, sidx, nacc, qsl, out)
        torch.cuda.synchronize()
        st["pad_nonzero"] += int((out[R * T:] != 0).sum())
        for i in range(R):
            c3 = ref_conv.clone(); s3 = ref_ssm.clone()
            o = nacc_h[i] - 1
            for t in range(T):
                row = i * T + t
                y = run_m1(L, c3, s3, i + 1, qkvz[row:row + 1], ba[row:row + 1])
                st["out"] += nmis(y[0], out[row]); st["rows"] += 1
                st["ssm"] += nmis(s3[i + 1], L.ssm[sidx[i, t]])
            # conv slot: [old[o+1], old[o+2], x_0..x_{T-1}]
            exp = torch.cat([conv_before[sidx[i, 0], o + 1:o + 3], qkvz[i * T:(i + 1) * T, :10240]], 0)
            st["conv"] += nmis(exp, L.conv[sidx[i, 0], :T + 2])
            if check64 and rd == 0:
                win = conv_before[sidx[i, 0], o:o + 3]
                h = ssm_keep[sidx[i, o]]
                for t in range(T):
                    row = i * T + t
                    y64, h = ref64_token(L, win, h, qkvz[row:row + 1], ba[row:row + 1])
                    st["max64"] = max(st["max64"], float((out[row].double() - y64).abs().max()))
                    st["ssm_max64"] = max(st["ssm_max64"], float((L.ssm[sidx[i, t]].double() - h).abs().max()))
                    win = torch.cat([win[1:], qkvz[row:row + 1, :10240]], 0)
        if rd == 0 and pad_req == 0 and pad_tok == 0:
            # stock Triton spec path on the same pre-round state (info: it is NOT exact)
            Ls = Layer.__new__(Layer); Ls.__dict__.update(L.__dict__)
            Ls.conv = conv_before.clone(); Ls.ssm = L.ssm.clone()
            for s_, v_ in ssm_snap.items():
                Ls.ssm[s_] = v_
            ys = stock_spec(Ls, qkvz, ba, sidx, nacc, qsl)
            st["stock_out_mis"] += nmis(ys[:R * T], out[:R * T])
            st["stock_out_maxabs"] = max(st["stock_out_maxabs"], float((ys[:R * T].float() - out[:R * T].float()).abs().max()))
        # commit a random number of accepted tokens per request; advance the reference by it
        for i in range(R):
            acc = int(torch.randint(1, T + 1, (1,), generator=torch.Generator().manual_seed(seed * 100 + rd * 10 + i)))
            for t in range(acc):
                row = i * T + t
                run_m1(L, ref_conv, ref_ssm, i + 1, qkvz[row:row + 1], ba[row:row + 1])
            nacc_h[i] = acc
    torch.cuda.synchronize()
    return st


def test_gdn(bench):
    out = {}
    _test_gdn(bench, out, "spec")
    RES["gdn"] = out


def _test_gdn(bench, out, tag):
    for R, T, pr, pt in ((1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (1, 4, 0, 0), (1, 5, 0, 0),
                         (1, 6, 0, 0), (1, 7, 0, 0), (1, 8, 0, 0), (2, 4, 0, 0), (3, 4, 1, 3),
                         (4, 2, 2, 2), (8, 4, 0, 0)):
        st = gdn_rounds(R, T, rounds=4, seed=17 + 7 * R + T, pad_req=pr, pad_tok=pt,
                        check64=(R <= 2 and T in (1, 4, 8)))
        out[f"{tag}_R{R}_T{T}_padreq{pr}_padtok{pt}"] = st
        print("gdn", tag, R, T, pr, pt, st, flush=True)
    if bench:
        NL = 12
        for R, T in ((1, 1), (1, 2), (1, 4), (2, 4), (1, 8)):
            g = torch.Generator(device=DEV).manual_seed(5)
            Ls = [Layer(g, 1 + R * T, T - 1) for _ in range(NL)]
            n = R * T
            qkvz = (torch.randn(n, 16384, device=DEV, generator=g) * 2).bfloat16()
            ba = (torch.randn(n, 96, device=DEV, generator=g) * 2).bfloat16()
            sidx = (torch.arange(R * T, dtype=torch.int32, device=DEV) + 1).view(R, T)
            nacc = torch.ones(R, dtype=torch.int32, device=DEV)
            qsl = torch.arange(R + 1, dtype=torch.int32, device=DEV) * T
            o = torch.empty(n, 6144, dtype=torch.bfloat16, device=DEV)
            r = {}
            r["spec"] = graph_time(lambda i: run_spec(Ls[i % NL], qkvz, ba, sidx, nacc, qsl, o))
            r["spec_no_state_store"] = graph_time(lambda i: run_spec(Ls[i % NL], qkvz, ba, sidx, nacc, qsl, o, dbg=2))
            r["spec_no_state_io"] = graph_time(lambda i: run_spec(Ls[i % NL], qkvz, ba, sidx, nacc, qsl, o, dbg=3))
            r["spec_nt"] = graph_time(lambda i: run_spec(Ls[i % NL], qkvz, ba, sidx, nacc, qsl, o, dbg=4))
            big = torch.ones(15_728_640, dtype=torch.bfloat16, device=DEV)  # 31 MB ~ gdn out_proj weights
            r["read31MB_alone"] = graph_time(lambda i: torch.sum(big))
            r["spec+read31MB"] = graph_time(lambda i: (run_spec(Ls[i % NL], qkvz, ba, sidx, nacc, qsl, o), torch.sum(big)))
            r["spec_nt+read31MB"] = graph_time(lambda i: (run_spec(Ls[i % NL], qkvz, ba, sidx, nacc, qsl, o, dbg=4), torch.sum(big)))
            if T > 1:
                r["stock_spec_path"] = graph_time(lambda i: stock_spec(Ls[i % NL], qkvz, ba, sidx, nacc, qsl))
            # deployed 1-token kernel on R rows (K=0 decode reference cost)
            c3 = [torch.randn(R + 1, 3, 10240, device=DEV).bfloat16() for _ in range(NL)]
            idx = torch.arange(1, R + 1, dtype=torch.int32, device=DEV)
            o1 = torch.empty(R, 6144, dtype=torch.bfloat16, device=DEV)

            def m1(i):
                L = Ls[i % NL]; c = c3[i % NL]
                torch.ops.vopt_rest.gdn_decode(qkvz[:R], qkvz[:R, 10240:], ba[:R, :48], ba[:R, 48:], c,
                                               c.stride(0), c.stride(2), c.stride(1), L.cw.view(10240, 4),
                                               L.A_log, L.dt_bias, L.norm_w, L.ssm, idx, o1, scratch(),
                                               128 ** -0.5, 1e-6)
            r["m1_kernel_R_rows"] = graph_time(m1)
            r["m1+read31MB"] = graph_time(lambda i: (m1(i), torch.sum(big)))

            def m1nt(i):
                L = Ls[i % NL]; c = c3[i % NL]
                torch.ops.vopt_rest.gdn_decode(qkvz[:R], qkvz[:R, 10240:], ba[:R, :48], ba[:R, 48:], c,
                                               c.stride(0), c.stride(2), c.stride(1), L.cw.view(10240, 4),
                                               L.A_log, L.dt_bias, L.norm_w, L.ssm, idx, o1, scratch(),
                                               128 ** -0.5, 1e-6, 4)
            r["m1_nt"] = graph_time(m1nt)
            r["m1_nt+read31MB"] = graph_time(lambda i: (m1nt(i), torch.sum(big)))
            r["floor_state_io"] = (1 + T) * R * 48 * 128 * 128 * 4 / 1.446e6
            out[f"bench_R{R}_T{T}_us_per_layer"] = {k: round(v, 2) for k, v in r.items()}
            print("bench gdn", R, T, out[f"bench_R{R}_T{T}_us_per_layer"], flush=True)


# ------------------------------------------------------------------------------ attention
def attn_case(L_ctx, R, T, BS, seed, use_meta=True):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    HQ, HK, HD = 24, 4, 256
    seg, thr = 16, 32
    g = torch.Generator(device=DEV).manual_seed(seed)
    so = torch.empty(thr, HQ, seg, HD, device=DEV, dtype=torch.float32)
    sm = torch.empty(thr, HQ, seg, device=DEV, dtype=torch.float32)
    se = torch.empty(thr, HQ, seg, device=DEV, dtype=torch.float32)
    maxlen = 8192
    nbc = (maxlen + BS - 1) // BS
    nblk = R * ((max(L_ctx) + T + BS - 1) // BS) + 4
    # model layout: kv_cache (blocks, H, BS, 2*hd) -> transpose(1,2).split
    kv = torch.randn(nblk, HK, BS, 2 * HD, device=DEV, generator=g).bfloat16()
    kc, vc = kv.transpose(1, 2).split(HD, dim=-1)
    bt = torch.zeros(R, nbc, dtype=torch.int32, device=DEV)
    perm = torch.randperm(nblk, generator=torch.Generator().manual_seed(seed))
    need = [(L_ctx[i] + T + BS - 1) // BS for i in range(R)]
    p = 0
    for i in range(R):
        bt[i, :need[i]] = perm[p:p + need[i]].to(torch.int32).to(DEV); p += need[i]
    n = R * T
    q = torch.randn(n, HQ, HD, device=DEV, generator=g).bfloat16()

    def run(qq, o, cu, mq, su, btab):
        unified_attention(qq, kc, vc, o, cu, mq, su, maxlen, HD ** -0.5, True, (-1, -1), btab, 0, None,
                          None, None, seq_threshold_3D=thr, num_par_softmax_segments=seg,
                          softmax_segm_output=so, softmax_segm_max=sm, softmax_segm_expsum=se)
    # 1-token decode reference per query (as K=0 graph decode: 3D path)
    ref = torch.empty(n, HQ, HD, dtype=torch.bfloat16, device=DEV)
    cu1 = torch.tensor([0, 1], dtype=torch.int32, device=DEV)
    for i in range(R):
        for t in range(T):
            s = i * T + t
            run(q[s:s + 1], ref[s:s + 1], cu1, 1, torch.tensor([L_ctx[i] + t + 1], dtype=torch.int32, device=DEV),
                bt[i:i + 1])
    qsl = torch.arange(R + 1, dtype=torch.int32, device=DEV) * T
    sl = torch.tensor([L_ctx[i] + T for i in range(R)], dtype=torch.int32, device=DEV)
    cu = torch.empty(n + 1, dtype=torch.int32, device=DEV)
    su = torch.empty(n, dtype=torch.int32, device=DEV)
    bto = torch.empty(n, nbc, dtype=torch.int32, device=DEV)
    torch.ops.vopt_rest.attn_pseudo_meta(qsl, sl, bt, R, cu, su, bto)
    o4 = torch.empty(n, HQ, HD, dtype=torch.bfloat16, device=DEV)
    run(q, o4, cu, 1, su, bto)
    o2 = torch.empty(n, HQ, HD, dtype=torch.bfloat16, device=DEV)
    run(q, o2, qsl, T, sl, bt)  # stock 2D multi-query path
    torch.cuda.synchronize()
    r = dict(pseudo_mis=nmis(o4, ref), stock2d_mis=nmis(o2, ref),
             stock2d_maxabs=float((o2.float() - ref.float()).abs().max()), n=ref.numel())
    timing = dict(
        stock2d=graph_time(lambda i: run(q, o2, qsl, T, sl, bt), nrep=16),
        pseudo3d=graph_time(lambda i: (torch.ops.vopt_rest.attn_pseudo_meta(qsl, sl, bt, R, cu, su, bto)
                                       if i == 0 else None, run(q, o4, cu, 1, su, bto)), nrep=16),
        m1_R=graph_time(lambda i: run(q[:R], ref[:R], torch.arange(R + 1, dtype=torch.int32, device=DEV),
                                      1, sl - T + 1, bt), nrep=16))
    return r, timing


def test_attn(bench):
    out = {}
    for BS in (16, 800):
        for L_ctx, R, T in (([40], 1, 4), ([300], 1, 4), ([1500], 1, 4), ([6000], 1, 4), ([700], 1, 8),
                            ([797], 1, 4), ([50, 3000], 2, 4), ([10, 900, 2000, 5], 4, 2), ([123] * 8, 8, 4)):
            r, tm = attn_case(L_ctx, R, T, BS, seed=BS + sum(L_ctx) + T)
            if bench:
                r["us_graph"] = {k: round(v, 2) for k, v in tm.items()}
            out[f"BS{BS}_ctx{'-'.join(map(str, L_ctx))}_R{R}_T{T}"] = r
            print("attn", BS, L_ctx, R, T, r, flush=True)
    RES["attn"] = out


# ------------------------------------------------------------------------------ per-token ops
def test_ops(bench):
    out = {}
    g = torch.Generator(device=DEV).manual_seed(3)
    w = (torch.randn(5120, device=DEV, generator=g) * 0.1).bfloat16()
    for T in range(1, 9):
        x = torch.randn(T, 5120, device=DEV, generator=g).bfloat16()
        rr = torch.randn(T, 5120, device=DEV, generator=g).bfloat16()
        y = torch.ops.vopt_rest.gemma_rms_norm(x, w, 1e-6)
        y2, r2 = torch.ops.vopt_rest.gemma_add_rms_norm(x, rr, w, 1e-6)
        gu = torch.randn(T, 2 * 17408, device=DEV, generator=g).bfloat16()
        sm_ = torch.ops.vopt_rest.silu_mul(gu)
        a = torch.randn(T, 6144, device=DEV, generator=g).bfloat16()
        gt = torch.randn(T, 6144, device=DEV, generator=g).bfloat16()
        sg = torch.ops.vopt_rest.sigmoid_mul(a, gt)
        mis = dict(rms=0, add_rms=0, add_res=0, silu_mul=0, sigmoid_mul=0)
        for t in range(T):
            mis["rms"] += nmis(torch.ops.vopt_rest.gemma_rms_norm(x[t:t + 1], w, 1e-6), y[t:t + 1])
            a1, b1 = torch.ops.vopt_rest.gemma_add_rms_norm(x[t:t + 1], rr[t:t + 1], w, 1e-6)
            mis["add_rms"] += nmis(a1, y2[t:t + 1]); mis["add_res"] += nmis(b1, r2[t:t + 1])
            mis["silu_mul"] += nmis(torch.ops.vopt_rest.silu_mul(gu[t:t + 1]), sm_[t:t + 1])
            mis["sigmoid_mul"] += nmis(torch.ops.vopt_rest.sigmoid_mul(a[t:t + 1], gt[t:t + 1]), sg[t:t + 1])
        # fused qk-norm / rope / gate (Triton, enabled on ROCm by the attn patch)
        from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
        HQ, HK, HD, RD = 24, 4, 256, 64
        qg = torch.randn(T, HQ * 2 * HD, device=DEV, generator=g).bfloat16()
        kk = torch.randn(T, HK * HD, device=DEV, generator=g).bfloat16()
        qw = (torch.randn(HD, device=DEV, generator=g) * 0.1).bfloat16()
        kw = (torch.randn(HD, device=DEV, generator=g) * 0.1).bfloat16()
        cos_sin = torch.randn(8192, RD, device=DEV, generator=g).bfloat16()
        pos = torch.randint(0, 8000, (T,), device=DEV, generator=g)
        qo, ko, go = fused_qk_rmsnorm_rope_gate(qg, kk, qw, kw, cos_sin, pos, 1e-6, HQ, HK, HD, RD, norm_beta=1.0)
        mis["qk_rope_gate"] = 0
        for t in range(T):
            q1, k1, g1 = fused_qk_rmsnorm_rope_gate(qg[t:t + 1], kk[t:t + 1], qw, kw, cos_sin, pos[t:t + 1],
                                                    1e-6, HQ, HK, HD, RD, norm_beta=1.0)
            mis["qk_rope_gate"] += nmis(q1, qo[t:t + 1]) + nmis(k1, ko[t:t + 1]) + nmis(g1, go[t:t + 1])
        # reshape_and_cache (Triton, used by TRITON_ATTN)
        from vllm.v1.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash
        BS = 800
        kv_a = torch.zeros(3, HK, BS, 2 * HD, device=DEV).bfloat16()
        kv_b = torch.zeros_like(kv_a)
        kn = torch.randn(T, HK, HD, device=DEV, generator=g).bfloat16()
        vn = torch.randn(T, HK, HD, device=DEV, generator=g).bfloat16()
        slots = torch.arange(T, device=DEV, dtype=torch.int64) + BS + 790
        one = torch.ones((), device=DEV)
        ka, va = kv_a.transpose(1, 2).split(HD, dim=-1)
        kb, vb = kv_b.transpose(1, 2).split(HD, dim=-1)
        triton_reshape_and_cache_flash(kn, vn, ka, va, slots, "auto", one, one)
        for t in range(T):
            triton_reshape_and_cache_flash(kn[t:t + 1], vn[t:t + 1], kb, vb, slots[t:t + 1], "auto", one, one)
        mis["reshape_and_cache"] = nmis(kv_a, kv_b)
        out[f"T{T}"] = mis
        print("ops", T, mis, flush=True)
    RES["ops"] = out


# ------------------------------------------------------------------------------ graph capture
def test_graph(bench):
    """gdn_spec captured with persistent metadata buffers, replayed with new num_accepted /
    inputs; must equal the eager run on the same states"""
    out = {}
    for R, T in ((1, 2), (1, 4), (2, 4), (3, 8)):
        g = torch.Generator(device=DEV).manual_seed(11 + R)
        L = Layer(g, 1 + R * T + 2, T - 1)
        n = R * T
        sidx = (torch.arange(R * T, dtype=torch.int32, device=DEV) + 1).view(R, T)
        nacc = torch.ones(R, dtype=torch.int32, device=DEV)
        qsl = torch.arange(R + 1, dtype=torch.int32, device=DEV) * T
        qkvz = torch.zeros(n, 16384, device=DEV).bfloat16()
        ba = torch.zeros(n, 96, device=DEV).bfloat16()
        o = torch.empty(n, 6144, dtype=torch.bfloat16, device=DEV)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        conv0, ssm0 = L.conv.clone(), L.ssm.clone()
        with torch.cuda.stream(s):
            run_spec(L, qkvz, ba, sidx, nacc, qsl, o)
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            run_spec(L, qkvz, ba, sidx, nacc, qsl, o)
        mis = 0
        for rd in range(4):
            L.conv.copy_(conv0); L.ssm.copy_(ssm0)
            qkvz.copy_((torch.randn(n, 16384, device=DEV, generator=g) * 2).bfloat16())
            ba.copy_((torch.randn(n, 96, device=DEV, generator=g) * 2).bfloat16())
            nacc.copy_(torch.randint(1, T + 1, (R,), device=DEV, generator=g).to(torch.int32))
            gr.replay(); torch.cuda.synchronize()
            yg, cg, sg = o.clone(), L.conv.clone(), L.ssm.clone()
            L.conv.copy_(conv0); L.ssm.copy_(ssm0)
            ye = run_spec(L, qkvz, ba, sidx, nacc, qsl); torch.cuda.synchronize()
            mis += nmis(yg, ye) + nmis(cg, L.conv) + nmis(sg, L.ssm)
        out[f"gdn_R{R}_T{T}_replay_vs_eager_mis"] = mis
        print("graph gdn", R, T, mis, flush=True)
    RES["graph"] = out


def test_prof(bench):
    """phase timestamps (s_memtime cycles from kernel entry) of block (0,0), dbg bit 3"""
    out = {}
    for T in (1, 2, 4, 8):
        g = torch.Generator(device=DEV).manual_seed(5)
        L = Layer(g, 1 + T, T - 1)
        qkvz = (torch.randn(T, 16384, device=DEV, generator=g) * 2).bfloat16()
        ba = (torch.randn(T, 96, device=DEV, generator=g) * 2).bfloat16()
        sidx = (torch.arange(T, dtype=torch.int32, device=DEV) + 1).view(1, T)
        nacc = torch.ones(1, dtype=torch.int32, device=DEV)
        qsl = torch.tensor([0, T], dtype=torch.int32, device=DEV)
        for dbg in (8, 8 | 3):
            res = []
            for rep in range(5):
                run_spec(L, qkvz, ba, sidx, nacc, qsl, dbg=dbg)
                torch.cuda.synchronize()
                res.append(scratch()[4096:4104].tolist())
            out[f"T{T}_dbg{dbg}"] = res[-1]
            print("prof", T, dbg, res[-1], flush=True)
    RES["prof"] = out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--only", default="gdn,attn,ops,graph")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    load_ext()
    torch.backends.cuda.matmul.allow_tf32 = False
    for name in args.only.split(","):
        {"gdn": test_gdn, "attn": test_attn, "ops": test_ops, "graph": test_graph, "prof": test_prof}[name](args.bench)
    print(json.dumps(RES, indent=1))
    if args.json:
        json.dump(RES, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
