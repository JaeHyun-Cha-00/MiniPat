# CSV -> deduped, filtered train/val/test JSONL

import csv
import hashlib
import json
import os
import random
import re

csv.field_size_limit(10**9)

CSV_IN = "data/raw/patent_pairs.csv"
OUT_DIR = "data/processed"

VAL_SIZE, TEST_SIZE = 1000, 500
SEED = 42

# Tuned once against CSV_IN; the per-reason drop counts print on every run.
MIN_KO, MAX_KO = 20, 2000
MIN_EN, MAX_EN = 20, 4000
MIN_RATIO, MAX_RATIO = 0.8, 6.0  # len(en)/len(ko); the corpus median is ~2.15


def norm(s):
    """Collapse whitespace; the BigQuery abstracts are full of newlines and runs of spaces."""
    return re.sub(r"\s+", " ", (s or "")).strip()


def key(s):
    return hashlib.sha1(s.lower().encode("utf-8")).hexdigest()


def main():
    with open(CSV_IN, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Read {len(rows)} rows from {CSV_IN}")

    drops = {"empty": 0, "length": 0, "ratio": 0, "dup_family": 0, "dup_pub": 0, "dup_text": 0}
    seen_family, seen_pub, seen_text = set(), set(), set()
    pairs = []

    for r in rows:
        ko, en = norm(r["ko_abstract"]), norm(r["en_abstract"])
        if not ko or not en:
            drops["empty"] += 1
            continue
        if not (MIN_KO <= len(ko) <= MAX_KO) or not (MIN_EN <= len(en) <= MAX_EN):
            drops["length"] += 1
            continue
        if not (MIN_RATIO <= len(en) / len(ko) <= MAX_RATIO):
            drops["ratio"] += 1
            continue
        # SQL already dedupes by family_id; these run as verification and print 0 when it worked.
        family = r.get("family_id") or ""
        if family and family in seen_family:
            drops["dup_family"] += 1
            continue
        pubs = (r["kr_pub"], r["en_pub"])
        if pubs[0] in seen_pub or pubs[1] in seen_pub:
            drops["dup_pub"] += 1
            continue
        texts = (key(ko), key(en))
        if texts[0] in seen_text or texts[1] in seen_text:
            drops["dup_text"] += 1
            continue
        if family:
            seen_family.add(family)
        seen_pub.update(pubs)
        seen_text.update(texts)
        pairs.append({
            "family_id": family,
            "kr_pub": r["kr_pub"],
            "en_pub": r["en_pub"],
            "en_source": r["en_source"],
            "ko": ko,
            "en": en,
        })

    print("Dropped: " + ", ".join(f"{k}={v}" for k, v in drops.items()))
    print(f"Kept {len(pairs)} pairs")

    random.Random(SEED).shuffle(pairs)
    splits = {
        "test": pairs[:TEST_SIZE],
        "val": pairs[TEST_SIZE:TEST_SIZE + VAL_SIZE],
        "train": pairs[TEST_SIZE + VAL_SIZE:],
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    for name in ("train", "val", "test"):
        path = os.path.join(OUT_DIR, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for p in splits[name]:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        wo = sum(1 for p in splits[name] if p["en_source"] == "WO")
        print(f"Wrote {path}: {len(splits[name])} pairs, WO {100 * wo / len(splits[name]):.1f}%")


if __name__ == "__main__":
    main()
