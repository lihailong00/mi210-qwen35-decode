"""Build the MI210 Bonsai Q2_0 projection manifest in inference order."""
import argparse
from pathlib import Path
from gguf import GGUFReader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("groups", type=Path)
    args = parser.parse_args()
    tensors = {t.name: t for t in GGUFReader(str(args.model)).tensors}
    groups = []
    for layer in range(64):
        prefix = f"blk.{layer}."
        if prefix + "attn_qkv.weight" in tensors:
            stages = [("attn_qkv", "attn_gate"), ("ssm_out",)]
        else:
            stages = [("attn_q", "attn_k", "attn_v"), ("attn_output",)]
        stages.extend([("ffn_gate", "ffn_up"), ("ffn_down",)])
        for stage in stages:
            groups.append([prefix + item + ".weight" for item in stage])
    groups.append(["output.weight"])
    weights, shapes = [], []
    for group in groups:
        width = int(tensors[group[0]].shape[0])
        shapes.append(f"{width} {len(group)}")
        for name in group:
            t = tensors[name]
            k, n = map(int, t.shape)
            if k != width or int(t.tensor_type) != 42 or t.n_bytes != k*n//64*18:
                raise ValueError(f"Unexpected Q2_0 tensor shape or format for {name}")
            weights.append(f"{name} {k} {n} {int(t.data_offset)} {int(t.n_bytes)}")
    if len(weights) != 401 or len(groups) != 257:
        raise ValueError("Unexpected Bonsai projection count")
    args.weights.write_text("\n".join(weights) + "\n")
    args.groups.write_text("\n".join(shapes) + "\n")

if __name__ == "__main__":
    main()
