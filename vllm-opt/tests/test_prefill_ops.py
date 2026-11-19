"""Prefill-size (M=601) check of the two vopt_rest patches that are active in prefill:
GemmaRMSNorm (+residual) and the fused qk-norm/RoPE/gate path on ROCm.
Each is compared with the stock vLLM code it replaces and with an fp64 reference built from the
same bf16 inputs. Run in the vLLM image (GPU lock):  python3 /vopt/deploy/tests/test_prefill_ops.py"""
import json
import sys

import torch

sys.path.insert(0, "/vopt/deploy/plugin")
from vopt_rest.ext import load_ext  # noqa: E402

load_ext()
DEV = "cuda"
torch.manual_seed(0)


def ulp(a, b):
    ai = a.contiguous().view(torch.int16).to(torch.int32)
    bi = b.contiguous().view(torch.int16).to(torch.int32)
    ai = torch.where(ai < 0, -32768 - ai, ai)
    bi = torch.where(bi < 0, -32768 - bi, bi)
    d = (ai - bi).abs()
    return dict(n_diff=int((d > 0).sum()), frac=round(float((d > 0).float().mean()), 5), max_ulp=int(d.max()))


def err(y, ref):
    d = (y.double() - ref).abs()
    return dict(max=float(d.max()), mean=float(d.mean()),
                rel_l2=float(d.norm() / ref.norm()))


def native_gemma(x, w, eps, r=None):
    from vllm import ir
    wf = w.float() + 1.0
    if r is None:
        return ir.ops.rms_norm(x, wf, eps)
    return ir.ops.fused_add_rms_norm(x, r, wf, eps)


RES = {}
eps = 1e-6
for M, xs, rs in ((601, 3.0, 50.0), (601, 1.0, 400.0), (4096, 3.0, 50.0)):
    D = 5120
    x = (torch.randn(M, D, device=DEV) * xs).bfloat16()
    r = (torch.randn(M, D, device=DEV) * rs).bfloat16()
    w = (torch.randn(D, device=DEV) * 0.3).bfloat16()
    y, r2 = torch.ops.vopt_rest.gemma_add_rms_norm(x, r, w, eps)
    yn, r2n = native_gemma(x.clone(), w, eps, r.clone())
    s = x.double() + r.double()
    # reference as the stock code defines it: residual rounded to bf16 first, then fp32 norm
    sb = (x.float() + r.float()).bfloat16().double()
    ref = sb * torch.rsqrt((sb * sb).mean(-1, keepdim=True) + eps) * (w.double() + 1)
    RES[f"addnorm_M{M}_x{xs}_r{rs}"] = dict(new_vs_stock=ulp(y, yn), resid_vs_stock=ulp(r2, r2n),
                                            new_vs_fp64=err(y, ref), stock_vs_fp64=err(yn, ref))
    y1 = torch.ops.vopt_rest.gemma_rms_norm(x, w, eps)
    y1n = native_gemma(x, w, eps)
    xd = x.double()
    ref1 = xd * torch.rsqrt((xd * xd).mean(-1, keepdim=True) + eps) * (w.double() + 1)
    RES[f"norm_M{M}_x{xs}"] = dict(new_vs_stock=ulp(y1, y1n), new_vs_fp64=err(y1, ref1),
                                   stock_vs_fp64=err(y1n, ref1))

# ---- qk norm + rope + gate with the model's real rope object
from vllm.model_executor.layers.rotary_embedding import get_rope  # noqa: E402
from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate  # noqa: E402

cfg = json.load(open("/model/config.json"))
tc = cfg.get("text_config", cfg)
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
torch.set_default_dtype(torch.bfloat16)
with set_current_vllm_config(VllmConfig()):
  rope = get_rope(head_size=256, max_position=tc["max_position_embeddings"],
                rope_parameters=tc["rope_parameters"]).to(DEV)
torch.set_default_dtype(torch.float32)
hd, nq, nk = 256, 24, 4
rd = rope.rotary_dim
for M in (601, 4096):
    qkv = (torch.randn(M, nq * 2 * hd + 2 * nk * hd, device=DEV) * 3).bfloat16()
    qw = (torch.randn(hd, device=DEV) * 0.2).bfloat16()
    kw = (torch.randn(hd, device=DEV) * 0.2).bfloat16()
    pos1 = torch.arange(M, device=DEV)
    pos3 = pos1.unsqueeze(0).expand(3, M).contiguous()  # text-only mrope positions
    q_gate, k, v = qkv.split([nq * 2 * hd, nk * hd, nk * hd], dim=-1)
    # fused, exactly as the patched model calls it
    p = pos3
    if p.ndim == 2 and not getattr(rope, "mrope_section", None):
        p = p[0]
    qf, kf, gf = fused_qk_rmsnorm_rope_gate(
        q_gate, k, qw, kw, rope.cos_sin_cache, p, eps, nq, nk, hd, rd, norm_beta=1.0,
        mrope_section=(getattr(rope, "mrope_section", None) if p.ndim == 2 else None))
    # stock: split/chunk + GemmaRMSNorm (native) + rotary_emb.forward_cuda
    qg = q_gate.reshape(M, nq, 2 * hd)
    qs, gs = torch.chunk(qg, 2, dim=-1)
    qs = qs.reshape(M, -1); gs = gs.reshape(M, -1)
    qn = native_gemma(qs.reshape(-1, hd).contiguous(), qw, eps).view(M, -1)
    kn = native_gemma(k.reshape(-1, hd).contiguous(), kw, eps).view(M, -1)
    qr, kr = rope.forward_cuda(pos3, qn.clone(), kn.clone())

    # fp64 reference: norm in fp64, rope with an fp64 cos/sin (theta from config)
    def ref_qk(xb, w, nh):
        xd = xb.reshape(M, nh, hd).double()
        xn = xd * torch.rsqrt((xd * xd).mean(-1, keepdim=True) + eps) * (w.double() + 1)
        theta = float(tc["rope_parameters"]["rope_theta"])
        inv = 1.0 / (theta ** (torch.arange(0, rd, 2, device=DEV, dtype=torch.float64) / rd))
        fr = torch.outer(pos1.double(), inv)
        cos, sin = fr.cos()[:, None, :], fr.sin()[:, None, :]
        x1, x2 = xn[..., :rd // 2], xn[..., rd // 2:rd]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, xn[..., rd:]], -1).reshape(M, -1)
    rq = ref_qk(qs, qw, nq); rk = ref_qk(k, kw, nk)
    RES[f"qk_M{M}"] = dict(
        rotary_dim=rd, cache_dtype=str(rope.cos_sin_cache.dtype),
        q_new_vs_stock=ulp(qf.reshape(M, -1), qr), k_new_vs_stock=ulp(kf.reshape(M, -1), kr),
        gate_equal=bool(torch.equal(gf.reshape(M, -1), gs)),
        q_new_vs_fp64=err(qf.reshape(M, -1), rq), q_stock_vs_fp64=err(qr, rq),
        k_new_vs_fp64=err(kf.reshape(M, -1), rk), k_stock_vs_fp64=err(kr, rk))
print(json.dumps(RES, indent=1))
