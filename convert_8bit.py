
import os
import yaml
import numpy as np

with open("params.yaml") as _f:
    P = yaml.safe_load(_f)

SRC         = P["paths"]["emb_float64"]
OUT_Q       = P["paths"]["emb_int8"]
OUT_SCALE   = P["paths"]["emb_int8_scales"]
OUT_ZP      = P["paths"]["emb_int8_zp"]
INT8_LEVELS = P["quantize"]["int8_levels"]

def file_size_mb(path):
    return os.path.getsize(path) / (1024 ** 2)

X = np.load(SRC)                              # (N, 768), float64
N, D = X.shape
print(f"Loaded {SRC}  shape={X.shape}  dtype={X.dtype}")
print(f"  size : {file_size_mb(SRC):.2f} MB\n")

x_min = X.min(axis=1, keepdims=True)          # (N, 1)
x_max = X.max(axis=1, keepdims=True)          # (N, 1)

scales = (x_max - x_min) / INT8_LEVELS             # (N, 1)
scales = np.where(scales == 0, 1.0, scales)      

zero_points = np.round(-x_min / scales)            # (N, 1), float
zero_points = np.clip(zero_points, 0, INT8_LEVELS).astype(np.uint8)

quantized = np.round(X / scales + zero_points)
quantized = np.clip(quantized, 0, INT8_LEVELS).astype(np.uint8)   # (N, 768)

scales_1d = scales.squeeze(1)                 # (N,)
zp_1d     = zero_points.squeeze(1)            # (N,)

X_reconstructed = (quantized.astype(np.float64) - zp_1d[:, None]) * scales_1d[:, None]

diff = np.abs(X - X_reconstructed)
print(f"  max  |error| : {diff.max():.6e}")
print(f"  mean |error| : {diff.mean():.6e}")
print(f"  max  scale   : {scales_1d.max():.6e}  (= worst-case ½ LSB error per row)\n")

np.save(OUT_Q,     quantized)
np.save(OUT_SCALE, scales_1d)
np.save(OUT_ZP,    zp_1d)

total_mb = sum(file_size_mb(p) for p in [OUT_Q, OUT_SCALE, OUT_ZP])
