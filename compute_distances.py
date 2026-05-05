import os
import time
import yaml
import numpy as np
import pandas as pd

with open("params.yaml") as _f:
    P = yaml.safe_load(_f)

SEG_CSV     = P["paths"]["segments_csv"]
RESULTS_OUT = P["paths"]["results_csv"]

PRECISIONS = {
    "float64": {
        "emb": P["paths"]["emb_float64"],
    },
    "float32": {
        "emb": P["paths"]["emb_float32"],
    },
    "float16": {
        "emb": P["paths"]["emb_float16"],
    },
    "int8": {
        "emb":    P["paths"]["emb_int8"],
        "scales": P["paths"]["emb_int8_scales"],
        "zp":     P["paths"]["emb_int8_zp"],
    },
}

def file_size_mb(path):
    return os.path.getsize(path) / (1024 ** 2)


def load_embeddings(cfg):
    if "scales" in cfg:
        q      = np.load(cfg["emb"])
        scales = np.load(cfg["scales"])
        zp     = np.load(cfg["zp"])
        X = (q.astype(np.float64) - zp[:, None]) * scales[:, None]
        size_mb = sum(file_size_mb(cfg[k]) for k in ["emb", "scales", "zp"])
    else:
        X = np.load(cfg["emb"])
        size_mb = file_size_mb(cfg["emb"])
    return X, size_mb


def cosine_distance_matrix(X):
    compute_dtype = np.float32 if X.dtype == np.float16 else X.dtype
    Xc = X.astype(compute_dtype)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    X_norm = Xc / norms
    sim = X_norm @ X_norm.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return (1.0 - sim).astype(X.dtype)


def build_pair_masks(seg_df):
    spk  = seg_df["speaker_id"].values
    word = seg_df["word"].values
    sent = seg_df["sentence_id"].values
    N    = len(seg_df)

    ii, jj = np.triu_indices(N, k=1)

    same_word = word[ii] == word[jj]
    same_spk  = spk[ii]  == spk[jj]
    diff_sent = sent[ii] != sent[jj]

    intra_mask = same_word & same_spk  & diff_sent
    inter_mask = same_word & ~same_spk

    print(f"Pair counts — intra: {intra_mask.sum():,}   inter: {inter_mask.sum():,}")
    return ii, jj, intra_mask, inter_mask


def compute_stats(D, ii, jj, intra_mask, inter_mask):
    d_flat = D[ii, jj].astype(np.float64)   # always compare in float64
    intra  = float(d_flat[intra_mask].mean()) if intra_mask.any() else float("nan")
    inter  = float(d_flat[inter_mask].mean()) if inter_mask.any() else float("nan")
    ratio  = inter / intra if intra > 0 else float("nan")
    return intra, inter, ratio

seg_df = pd.read_csv(SEG_CSV)
print(f"Segments loaded: {len(seg_df)}\n")

ii, jj, intra_mask, inter_mask = build_pair_masks(seg_df)

rows = []

for precision, cfg in PRECISIONS.items():
    print(f"── {precision} " + "─" * 40)

    X, size_mb = load_embeddings(cfg)
    print(f"  shape={X.shape}  dtype={X.dtype}  size={size_mb:.2f} MB")

    t0 = time.perf_counter()
    D  = cosine_distance_matrix(X)
    elapsed = time.perf_counter() - t0
    print(f"  distance matrix: {elapsed:.3f}s")

    intra, inter, ratio = compute_stats(D, ii, jj, intra_mask, inter_mask)
    print(f"  intra={intra:.6f}  inter={inter:.6f}  ratio={ratio:.4f}\n")

    rows.append({
        "precision":      precision,
        "intra_mean":     round(intra,    6),
        "inter_mean":     round(inter,    6),
        "ratio":          round(ratio,    4),
        "compute_time_s": round(elapsed,  3),
        "file_size_mb":   round(size_mb,  2),
    })

results_df = pd.DataFrame(rows)
results_df.to_csv(RESULTS_OUT, index=False)
