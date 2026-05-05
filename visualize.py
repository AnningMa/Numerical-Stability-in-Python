#!/usr/bin/env python3
"""
Step 5: Visualise distance distributions and efficiency tradeoffs.

Figures saved to figures/:
  fig1_kde.png         KDE of intra/inter distance distributions per precision
  fig2_bars.png        Grouped bar chart: intra vs inter mean distances
  fig3_ratio.png       inter/intra ratio across precisions
  fig5_efficiency.png  File size and compute time across precisions
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ── config ───────────────────────────────────────────────────────────────────
with open("params.yaml") as _f:
    _P = yaml.safe_load(_f)

RESULTS_CSV = _P["paths"]["results_csv"]
SEG_CSV     = _P["paths"]["segments_csv"]
MAX_SAMPLES = _P["visualize"]["max_samples"]
SEED        = _P["visualize"]["seed"]
FIGURES_DIR = _P["paths"]["figures_dir"]

PRECISIONS  = ["float64", "float32", "float16", "int8"]
PREC_LABELS = ["float64\n(ref)", "float32", "float16", "int8"]
EMB_CFG     = {
    "float64": {"emb": _P["paths"]["emb_float64"]},
    "float32": {"emb": _P["paths"]["emb_float32"]},
    "float16": {"emb": _P["paths"]["emb_float16"]},
    "int8": {
        "emb":    _P["paths"]["emb_int8"],
        "scales": _P["paths"]["emb_int8_scales"],
        "zp":     _P["paths"]["emb_int8_zp"],
    },
}
COLOR_INTRA = "#2196F3"
COLOR_INTER = "#F44336"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(FIGURES_DIR, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


# ── helpers ──────────────────────────────────────────────────────────────────

def load_embeddings(cfg):
    """Load embedding matrix in native dtype (float16/32/64 or reconstructed int8)."""
    if "scales" in cfg:
        q      = np.load(cfg["emb"])
        scales = np.load(cfg["scales"])
        zp     = np.load(cfg["zp"])
        return (q.astype(np.float64) - zp[:, None]) * scales[:, None]
    return np.load(cfg["emb"])


def build_pair_samples(seg_df, max_samples, seed):
    """
    Build sampled (ii, jj) index arrays for intra and inter pairs.
    Intra: same word, same speaker, different sentence_id.
    Inter: same word, different speaker.
    """
    rng = np.random.default_rng(seed)

    intra_ii, intra_jj = [], []
    for (_, _), grp in seg_df.groupby(["word", "speaker_id"]):
        idxs  = grp.index.values
        sents = grp["sentence_id"].values
        n = len(idxs)
        for a in range(n):
            for b in range(a + 1, n):
                if sents[a] != sents[b]:
                    intra_ii.append(idxs[a])
                    intra_jj.append(idxs[b])

    inter_ii, inter_jj = [], []
    for _, grp in seg_df.groupby("word"):
        speakers = grp["speaker_id"].unique()
        spk_idx  = {s: grp[grp["speaker_id"] == s].index.values for s in speakers}
        for p in range(len(speakers)):
            for q_idx in range(p + 1, len(speakers)):
                a_idx = spk_idx[speakers[p]]
                b_idx = spk_idx[speakers[q_idx]]
                aa, bb = np.meshgrid(a_idx, b_idx)
                inter_ii.append(aa.ravel())
                inter_jj.append(bb.ravel())

    intra_ii = np.array(intra_ii)
    intra_jj = np.array(intra_jj)
    inter_ii  = np.concatenate(inter_ii)
    inter_jj  = np.concatenate(inter_jj)

    if len(intra_ii) > max_samples:
        sel = rng.choice(len(intra_ii), max_samples, replace=False)
        intra_ii, intra_jj = intra_ii[sel], intra_jj[sel]
    if len(inter_ii) > max_samples:
        sel = rng.choice(len(inter_ii), max_samples, replace=False)
        inter_ii, inter_jj = inter_ii[sel], inter_jj[sel]

    print(f"Pair samples — intra: {len(intra_ii):,}   inter: {len(inter_ii):,}")
    return intra_ii, intra_jj, inter_ii, inter_jj


def cosine_distances_for_pairs(X, ii, jj):
    """Compute cosine distances for given index pairs in native dtype."""
    X = X.astype(np.float32) if X.dtype == np.float16 else X  # avoid float16 norm overflow
    norms = np.linalg.norm(X, axis=1)
    norms = np.where(norms == 0, 1.0, norms)
    sim = (X[ii] * X[jj]).sum(axis=1) / (norms[ii] * norms[jj])
    return (1.0 - np.clip(sim, -1.0, 1.0)).astype(np.float64)


# ── load data ────────────────────────────────────────────────────────────────
results = pd.read_csv(RESULTS_CSV)
seg_df  = pd.read_csv(SEG_CSV)

print("Building pair sample indices …")
intra_ii, intra_jj, inter_ii, inter_jj = build_pair_samples(seg_df, MAX_SAMPLES, SEED)

print("Computing sampled distances per precision …")
kde_data = {}
for p in PRECISIONS:
    X = load_embeddings(EMB_CFG[p])
    kde_data[p] = {
        "intra": cosine_distances_for_pairs(X, intra_ii, intra_jj),
        "inter": cosine_distances_for_pairs(X, inter_ii, inter_jj),
    }
    print(f"  {p} done")


# ── Figure 1: KDE distance distributions ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=False)
fig.suptitle("Cosine Distance Distributions by Precision", fontsize=14, fontweight="bold")

for ax, p, label in zip(axes.flat, PRECISIONS, PREC_LABELS):
    for category, color, ls in [("intra", COLOR_INTRA, "-"), ("inter", COLOR_INTER, "--")]:
        d = kde_data[p][category]
        xs = np.linspace(d.min(), d.max(), 500)
        kde = gaussian_kde(d, bw_method="scott")
        ax.plot(xs, kde(xs), color=color, ls=ls, lw=2,
                label=f"{category}-speaker")
        ax.axvline(d.mean(), color=color, lw=0.8, alpha=0.5)

    ax.set_title(label.replace("\n", " "), fontsize=11)
    ax.set_xlabel("Cosine distance")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig1_kde.png", dpi=150)
plt.close()
print("Saved figures/fig1_kde.png")


# ── Figure 2: Grouped bar chart intra vs inter ────────────────────────────────
x    = np.arange(len(PRECISIONS))
w    = 0.35
fig, ax = plt.subplots(figsize=(8, 5))

bars_intra = ax.bar(x - w / 2, results["intra_mean"], w,
                    label="intra-speaker", color=COLOR_INTRA, alpha=0.85)
bars_inter = ax.bar(x + w / 2, results["inter_mean"], w,
                    label="inter-speaker", color=COLOR_INTER, alpha=0.85)

for bar in list(bars_intra) + list(bars_inter):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(PREC_LABELS)
ax.set_ylabel("Mean cosine distance")
ax.set_title("Intra- vs Inter-Speaker Mean Distances", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig2_bars.png", dpi=150)
plt.close()
print("Saved figures/fig2_bars.png")


# ── Figure 3: inter/intra ratio line plot ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(PREC_LABELS, results["ratio"], marker="o", color="#4CAF50",
        lw=2, ms=8, label="inter / intra ratio")
ax.axhline(1.0, color="gray", lw=1.2, ls="--", label="ratio = 1 (no separation)")

for i, (label, val) in enumerate(zip(PREC_LABELS, results["ratio"])):
    ax.annotate(f"{val:.3f}", (label, val),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9)

ax.set_ylabel("inter / intra ratio")
ax.set_title("Speaker Separability Ratio Across Precisions", fontweight="bold")
ax.set_ylim(bottom=0)
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig3_ratio.png", dpi=150)
plt.close()
print("Saved figures/fig3_ratio.png")


# ── Figure 5: efficiency — file size and compute time ─────────────────────────
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

x = np.arange(len(PRECISIONS))
w = 0.35

b1 = ax1.bar(x - w / 2, results["file_size_mb"],   w,
             color="#9C27B0", alpha=0.8, label="File size (MB)")
b2 = ax2.bar(x + w / 2, results["compute_time_s"], w,
             color="#FF9800", alpha=0.8, label="Compute time (s)")

for bar in b1:
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{bar.get_height():.1f}", ha="center", fontsize=8, color="#9C27B0")
for bar in b2:
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
             f"{bar.get_height():.2f}s", ha="center", fontsize=8, color="#FF9800")

ax1.set_xticks(x)
ax1.set_xticklabels(PREC_LABELS)
ax1.set_ylabel("File size (MB)", color="#9C27B0")
ax2.set_ylabel("Compute time (s)", color="#FF9800")
ax1.tick_params(axis="y", colors="#9C27B0")
ax2.tick_params(axis="y", colors="#FF9800")
ax1.set_title("Storage and Computation Cost by Precision", fontweight="bold")

lines = [b1, b2]
labels = ["File size (MB)", "Compute time (s)"]
ax1.legend(lines, labels, loc="upper right")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig5_efficiency.png", dpi=150)
plt.close()
print("Saved figures/fig5_efficiency.png")

print("\nAll figures saved to figures/")
