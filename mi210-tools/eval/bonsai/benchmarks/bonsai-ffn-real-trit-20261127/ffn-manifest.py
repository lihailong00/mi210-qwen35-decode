import argparse
from pathlib import Path
from gguf import GGUFReader
parser = argparse.ArgumentParser(description='Extract the 192 Bonsai FFN tensor offsets for the MI210 replay.')
parser.add_argument('model', type=Path)
parser.add_argument('manifest', type=Path)
args = parser.parse_args()
model = args.model
tensors={t.name:t for t in GGUFReader(str(model)).tensors}
rows=[]
for l in range(64):
 for kind in ('gate','up','down'):
  name=f'blk.{l}.ffn_{kind}.weight';t=tensors[name]; k,n=map(int,t.shape)
  assert int(t.tensor_type)==42 and t.n_bytes==k*n//64*18 and (k,n)==((17408,5120) if kind=='down' else (5120,17408))
  rows.append(f'{name} {k} {n} {int(t.data_offset)} {int(t.n_bytes)}')
args.manifest.write_text('\n'.join(rows)+'\n')
