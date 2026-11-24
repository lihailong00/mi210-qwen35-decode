#!/usr/bin/env python3
# usage: ppsum.py <run_kernel_trace.csv> [top=40] -- per-kernel time summary of a prefill trace
import csv, sys, re, collections

rows = []
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"],
                     int(r["Grid_Size_X"]) * int(r["Grid_Size_Y"]) * int(r["Grid_Size_Z"])))
rows.sort()
top = int(sys.argv[2]) if len(sys.argv) > 2 else 40

def short(n):
    n = re.sub(r"\(.*", "", n)
    n = n.replace("void ", "")
    return n[:110]

CATS = [
    ("moe GEMM (mul_mat_q / mmq, expert)", r"mul_mat_q|mmq"),
    ("dense GEMM (rocBLAS/Tensile/hipBLASLt)", r"Cijk_|gemm|Tensile|hipblaslt|rocblas"),
    ("mul_mat_vec / mmvq / mmvf", r"mul_mat_vec"),
    ("flash attention", r"flash_attn|fattn"),
    ("quantize activations", r"quantize"),
    ("convert / cpy / cont", r"convert|cpy|copy|cont"),
    ("moe sort / ids / get_rows", r"get_rows|argsort|topk|mmid|ids|scatter|gather"),
    ("norms", r"norm"),
    ("binary / unary elementwise", r"bin_bcast|unary|scale|clamp|sigmoid|silu|gelu|soft|op_|fill|sqr|sqrt|div|mul_kernel|add"),
    ("concat / pad / permute / repeat", r"concat|pad|permute|repeat|transpose"),
    ("ink_* (our decode kernels)", r"ink"),
    ("ssm conv", r"ssm"),
    ("sum_rows / reduce", r"sum|reduce"),
]

tot = sum(e - s for s, e, _, _ in rows)
span = rows[-1][1] - rows[0][0]
print(f"kernels {len(rows)}  busy {tot/1e6:.1f} ms  span {span/1e6:.1f} ms")
by = collections.defaultdict(lambda: [0, 0])
for s, e, n, g in rows:
    k = short(n)
    by[k][0] += 1
    by[k][1] += e - s
cat = collections.defaultdict(lambda: [0, 0])
for k, (c, d) in by.items():
    for name, rx in CATS:
        if re.search(rx, k, re.I):
            break
    else:
        name = "other"
    cat[name][0] += c
    cat[name][1] += d
print("\nby category:")
for k, (c, d) in sorted(cat.items(), key=lambda x: -x[1][1]):
    print(f"  {d/1e6:8.1f} ms {100*d/tot:5.1f}%  n={c:6d}  {k}")
print(f"\ntop {top} kernels:")
for k, (c, d) in sorted(by.items(), key=lambda x: -x[1][1])[:top]:
    print(f"  {d/1e6:8.2f} ms {100*d/tot:5.1f}%  n={c:5d}  {d/c/1e3:8.1f} us  {k}")
