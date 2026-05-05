#!/usr/bin/env python3
"""
Step 3b: Asymmetric per-row 8-bit quantization of float64 embeddings.

Scheme:
  For each row i (one embedding vector of dim 768):
    scale_i     = (max(xᵢ) − min(xᵢ)) / 255
    zero_point_i = round(−min(xᵢ) / scale_i)        ∈ [0, 255]
    q_i          = clip(round(xᵢ / scale_i + zero_point_i), 0, 255)  → uint8

  Reconstruction:
    x̂_i = (q_i − zero_point_i) × scale_i

Input:  features/embeddings_float64.npy
Output: features/embeddings_int8.npy          uint8  (N, 768)
        features/embeddings_int8_scales.npy   float64 (N,)
        features/embeddings_int8_zp.npy       uint8   (N,)
"""

import os
import yaml
import numpy as np

with open("params.yaml") as _f:
    _P = yaml.safe_load(_f)

SRC         = _P["paths"]["emb_float64"]
OUT_Q       = _P["paths"]["emb_int8"]
OUT_SCALE   = _P["paths"]["emb_int8_scales"]
OUT_ZP      = _P["paths"]["emb_int8_zp"]
INT8_LEVELS = _P["quantize"]["int8_levels"]


def file_size_mb(path):
    return os.path.getsize(path) / (1024 ** 2)


# ── load reference ───────────────────────────────────────────────────────────
X = np.load(SRC)                              # (N, 768), float64
N, D = X.shape
print(f"Loaded {SRC}  shape={X.shape}  dtype={X.dtype}")
print(f"  size : {file_size_mb(SRC):.2f} MB\n")

# ── per-row quantization (vectorised) ────────────────────────────────────────
x_min = X.min(axis=1, keepdims=True)          # (N, 1)
x_max = X.max(axis=1, keepdims=True)          # (N, 1)

scales = (x_max - x_min) / INT8_LEVELS             # (N, 1)
scales = np.where(scales == 0, 1.0, scales)        # avoid division by zero for constant rows

zero_points = np.round(-x_min / scales)            # (N, 1), float
zero_points = np.clip(zero_points, 0, INT8_LEVELS).astype(np.uint8)

quantized = np.round(X / scales + zero_points)
quantized = np.clip(quantized, 0, INT8_LEVELS).astype(np.uint8)   # (N, 768)

# ── verify reconstruction error ──────────────────────────────────────────────
scales_1d = scales.squeeze(1)                 # (N,)
zp_1d     = zero_points.squeeze(1)            # (N,)

X_reconstructed = (quantized.astype(np.float64) - zp_1d[:, None]) * scales_1d[:, None]

diff = np.abs(X - X_reconstructed)
print("Reconstruction error (vs float64 reference):")
print(f"  max  |error| : {diff.max():.6e}")
print(f"  mean |error| : {diff.mean():.6e}")
print(f"  max  scale   : {scales_1d.max():.6e}  (= worst-case ½ LSB error per row)\n")

# ── save ─────────────────────────────────────────────────────────────────────
np.save(OUT_Q,     quantized)
np.save(OUT_SCALE, scales_1d)
np.save(OUT_ZP,    zp_1d)

total_mb = sum(file_size_mb(p) for p in [OUT_Q, OUT_SCALE, OUT_ZP])
print(f"Saved:")
print(f"  {OUT_Q:<45}  {file_size_mb(OUT_Q):.2f} MB  (uint8 embeddings)")
print(f"  {OUT_SCALE:<45}  {file_size_mb(OUT_SCALE):.2f} MB  (per-row scale)")
print(f"  {OUT_ZP:<45}  {file_size_mb(OUT_ZP):.2f} MB  (per-row zero_point)")
print(f"  {'total':<45}  {total_mb:.2f} MB")
print(f"  compression vs float64 : ×{file_size_mb(SRC) / total_mb:.1f}")
