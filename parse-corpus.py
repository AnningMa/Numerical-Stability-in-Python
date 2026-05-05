#!/usr/bin/env python3
"""
Step 1: Parse corpus TextGrids → features/words.csv
Each row = one word token.
Columns: speaker_id, sentence_id, repetition, word, onset, offset, duration_ms, l1_status, gender
"""

import os
import re
import yaml
import pandas as pd
from praatio import textgrid

with open("params.yaml") as _f:
    _P = yaml.safe_load(_f)

CORPUS_DIR           = _P["paths"]["corpus_dir"]
METADATA             = _P["paths"]["metadata"]
OUTPUT               = _P["paths"]["words_csv"]
WORD_TIER_CANDIDATES = _P["parse"]["word_tier_candidates"]
MIN_SPEAKERS         = _P["filter"]["min_speakers"]
MIN_OCC              = _P["filter"]["min_occ"]

# Filename pattern: {spk}_{fra|rus}_{list}_{FRcorpN}.TextGrid
FNAME_RE = re.compile(r"^.+_(fra|rus)_(list\d+)_(FRcorp\d+)\.TextGrid$", re.IGNORECASE)

# metadata — drop duplicate speaker rows to avoid .loc returning a Series
meta = pd.read_csv(METADATA, sep=";")
meta["spk"] = meta["spk"].str.upper()
meta = meta.set_index("spk")
meta = meta[~meta.index.duplicated(keep="first")]

# parse
rows = []
skipped_no_wav       = 0
skipped_no_meta      = 0
skipped_no_word_tier = 0

for spk_dir in sorted(os.listdir(CORPUS_DIR)):
    spk_path = os.path.join(CORPUS_DIR, spk_dir)
    if not os.path.isdir(spk_path):
        continue

    spk_id = spk_dir.upper()
    if spk_id not in meta.index:
        print(f"[skip] {spk_id}: not in metadata")
        skipped_no_meta += 1
        continue

    l1_status = meta.loc[spk_id, "L1"]
    gender    = meta.loc[spk_id, "Gender"]

    for fname in sorted(os.listdir(spk_path)):
        m = FNAME_RE.match(fname)
        if not m:
            continue

        repetition  = int(re.search(r"\d+", m.group(2)).group())
        sentence_id = int(re.search(r"\d+", m.group(3)).group())

        # Bug fix: use re.sub for case-insensitive extension replacement
        wav_name = re.sub(r"\.TextGrid$", ".wav", fname, flags=re.IGNORECASE)
        if not os.path.exists(os.path.join(spk_path, wav_name)):
            skipped_no_wav += 1
            continue

        tg = textgrid.openTextgrid(
            os.path.join(spk_path, fname),
            includeEmptyIntervals=False,
        )

        # Find word tier by trying candidate names
        available_tiers = tg.tierNames
        words_tier = None
        for candidate in WORD_TIER_CANDIDATES:
            if candidate in available_tiers:
                words_tier = tg.getTier(candidate)
                break

        if words_tier is None:
            print(f"[warn] {fname}: no word tier found (available: {available_tiers})")
            skipped_no_word_tier += 1
            continue

        for entry in words_tier.entries:
            onset, offset, word = entry.start, entry.end, entry.label
            if not word.strip():
                continue
            rows.append({
                "speaker_id":  spk_id,
                "sentence_id": sentence_id,
                "repetition":  repetition,
                "word":        word.strip().lower(),
                "onset":       round(onset, 6),
                "offset":      round(offset, 6),
                "duration_ms": round((offset - onset) * 1000, 2),
                "l1_status":   l1_status,
                "gender":      gender,
            })

# save
os.makedirs("features", exist_ok=True)
cols = ["speaker_id", "sentence_id", "repetition", "word",
        "onset", "offset", "duration_ms", "l1_status", "gender"]
df = pd.DataFrame(rows, columns=cols)
df.to_csv(OUTPUT, index=False)

print(f"Done: {len(df)} word tokens → {OUTPUT}")
print(f"  Speakers parsed       : {df['speaker_id'].nunique()}")
print(f"  Unique words          : {df['word'].nunique()}")
print(f"  Skipped (no WAV)      : {skipped_no_wav}")
print(f"  Skipped (no meta)     : {skipped_no_meta}")
print(f"  Skipped (no word tier): {skipped_no_word_tier}")

# Coverage analysis: words usable for the experiment
# "repetitions" = distinct sentence_ids where the word appears for a given speaker
word_spk_count  = df.groupby("word")["speaker_id"].nunique()
word_occ_count  = df.groupby(["word", "speaker_id"])["sentence_id"].nunique()
min_occ_per_spk = word_occ_count.groupby("word").min()

usable = word_spk_count.index[
    (word_spk_count >= MIN_SPEAKERS) & (min_occ_per_spk >= MIN_OCC)
]
print(f"\nWords usable for experiment (≥2 speakers, each ≥2 sentence occurrences): {len(usable)}")
if len(usable):
    print(f"  {'Word':<20} {'#Speakers':>10} {'Min occurrences/speaker':>24}")
    for w in usable:
        print(f"  {w:<20} {word_spk_count[w]:>10} {min_occ_per_spk[w]:>24}")
