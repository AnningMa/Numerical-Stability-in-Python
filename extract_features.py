#!/usr/bin/env python3
"""
Step 2: Extract wav2vec2 word-level embeddings (float64 reference).
Inputs:  features/words.csv  +  data/corpus/**/*.wav
Outputs: features/embeddings_float64.npy   shape: (N, 768)
         features/segments.csv             metadata aligned with embedding rows
"""

import os
import re
import yaml
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from math import gcd
from scipy.signal import resample_poly
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# ── config ──────────────────────────────────────────────────────────────────
with open("params.yaml") as _f:
    _P = yaml.safe_load(_f)

CORPUS_DIR          = _P["paths"]["corpus_dir"]
WORDS_CSV           = _P["paths"]["words_csv"]
EMB_OUT             = _P["paths"]["emb_float64"]
SEG_OUT             = _P["paths"]["segments_csv"]
MODEL_NAME          = _P["extract"]["model_name"]
TARGET_SR           = _P["extract"]["target_sr"]
MIN_SEGMENT_SAMPLES = _P["extract"]["min_segment_samples"]
MIN_SPEAKERS        = _P["filter"]["min_speakers"]
MIN_OCC             = _P["filter"]["min_occ"]
# ────────────────────────────────────────────────────────────────────────────

L1_TO_TAG = {"fr": "fra", "ru": "rus"}


def find_wav(spk_id, l1_status, repetition, sentence_id):
    """Return WAV path, tolerating zero-padding in FRcorp number."""
    spk_dir = os.path.join(CORPUS_DIR, spk_id)
    lang_tag = L1_TO_TAG.get(l1_status, "*")
    wav_re = re.compile(
        rf"^{re.escape(spk_id)}_{lang_tag}_list{repetition}_FRcorp0*{sentence_id}\.wav$",
        re.IGNORECASE,
    )
    for fname in os.listdir(spk_dir):
        if wav_re.match(fname):
            return os.path.join(spk_dir, fname)
    return None


def load_segment(wav_path, onset, offset):
    """Load audio slice [onset, offset] and resample to TARGET_SR."""
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:          # stereo → mono
        audio = audio.mean(axis=1)
    segment = audio[int(onset * sr) : int(offset * sr)]
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        segment = resample_poly(segment, TARGET_SR // g, sr // g).astype("float32")
    return segment


def mean_pool_last_hidden(last_hidden_state):
    """Mean-pool over time, return float64 numpy vector."""
    return last_hidden_state.squeeze(0).mean(dim=0).double().numpy() #.double() converts to float64


# ── load model (once) ────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME} …")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model     = Wav2Vec2Model.from_pretrained(MODEL_NAME)
model.eval()

# ── filter words.csv to usable tokens ───────────────────────────────────────
df = pd.read_csv(WORDS_CSV)

word_spk_count  = df.groupby("word")["speaker_id"].nunique()
word_occ_count  = df.groupby(["word", "speaker_id"])["sentence_id"].nunique()
min_occ_per_spk = word_occ_count.groupby("word").min()
usable_words    = word_spk_count.index[
    (word_spk_count >= MIN_SPEAKERS) & (min_occ_per_spk >= MIN_OCC)
]
df = df[df["word"].isin(usable_words)].reset_index(drop=True)
print(f"Tokens to process: {len(df)}  ({df['word'].nunique()} words, "
      f"{df['speaker_id'].nunique()} speakers)")

# ── extract ──────────────────────────────────────────────────────────────────
embeddings = []
seg_rows   = []
skipped    = 0

for i, row in df.iterrows():
    wav_path = find_wav(row.speaker_id, row.l1_status, row.repetition, row.sentence_id)
    if wav_path is None:
        print(f"  [warn] not found: {row.speaker_id} sent={row.sentence_id} rep={row.repetition}")
        skipped += 1
        continue

    segment = load_segment(wav_path, row.onset, row.offset)
    if len(segment) < MIN_SEGMENT_SAMPLES:
        skipped += 1
        continue

    inputs = processor(segment, sampling_rate=TARGET_SR, return_tensors="pt",
                       return_attention_mask=True)

    with torch.no_grad():
        outputs = model(**inputs)

    vec = mean_pool_last_hidden(outputs.last_hidden_state)
    embeddings.append(vec)
    seg_rows.append({
        "idx":         len(embeddings) - 1,
        "speaker_id":  row.speaker_id,
        "sentence_id": row.sentence_id,
        "repetition":  row.repetition,
        "word":        row.word,
        "onset":       row.onset,
        "offset":      row.offset,
        "l1_status":   row.l1_status,
        "gender":      row.gender,
    })

    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(df)} processed …")

# ── save ─────────────────────────────────────────────────────────────────────
os.makedirs("features", exist_ok=True)
emb_array = np.stack(embeddings)      # (N, 768), float64
np.save(EMB_OUT, emb_array)

pd.DataFrame(seg_rows).to_csv(SEG_OUT, index=False)

print(f"\nDone: {len(embeddings)} embeddings → {EMB_OUT}")
print(f"  Shape  : {emb_array.shape}   dtype: {emb_array.dtype}")
print(f"  Skipped: {skipped}")
