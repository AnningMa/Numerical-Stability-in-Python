import os
import yaml
import numpy as np

with open("params.yaml") as _f:
    P = yaml.safe_load(_f)

SRC = P["paths"]["emb_float64"]
OUT = {
    "float32": _P["paths"]["emb_float32"],
    "float16": _P["paths"]["emb_float16"],
}


def file_size_mb(path):
    return os.path.getsize(path) / (1024 ** 2)

emb64 = np.load(SRC)

for dtype_name, out_path in OUT.items():
    converted = emb64.astype(dtype_name)
    np.save(out_path, converted)
    size_mb = file_size_mb(out_path)
    ratio   = file_size_mb(SRC) / size_mb
    print(f"{dtype_name:>8}  →  {out_path}")
    print(f"           size : {size_mb:.2f} MB  (×{ratio:.1f} smaller than float64)")

    # numerical error vs float64 reference
    diff = np.abs(emb64 - converted.astype("float64"))
    print(f"           max |error| : {diff.max():.6e}")
    print(f"           mean|error| : {diff.mean():.6e}\n")
