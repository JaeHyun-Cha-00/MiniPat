# How close each Korean/English abstract pair is to being a translation, by LaBSE similarity.
#
# Pairs come from the same patent family, not from a translation. WO abstracts are often
# translations of each other, but US abstracts are frequently rewritten from scratch, so some
# "references" say different things than the source. LaBSE cosine similarity flags those:
# a random (unrelated) pair scores ~0.44, a close translation ~0.9, a rewrite of the same
# invention in between.
#
#   pip install -r requirements/align.txt
#   python data_pipeline/score_alignment.py      # -> data/processed/labse_{train,val,test}.jsonl
#
# Output lines are {"family_id", "labse_sim"}; join on family_id. LaBSE reads only the first
# 256 tokens of each abstract, so the score reflects the opening of long abstracts.

import argparse
import json
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

IN_DIR = "data/processed"
SPLITS = ("train", "val", "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/LaBSE", device=device)

    for split in args.splits:
        with open(os.path.join(IN_DIR, f"{split}.jsonl")) as f:
            rows = [json.loads(line) for line in f]
        enc = lambda texts: model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                                         show_progress_bar=True)
        sim = (enc([r["ko"] for r in rows]) * enc([r["en"] for r in rows])).sum(axis=1)

        out_path = os.path.join(IN_DIR, f"labse_{split}.jsonl")
        with open(out_path, "w") as f:
            for r, s in zip(rows, sim):
                f.write(json.dumps({"family_id": r["family_id"], "labse_sim": round(float(s), 4)}) + "\n")
        print(f"{split}: {len(rows)} pairs, median {np.median(sim):.3f}, "
              f"<0.80: {(sim < 0.80).mean():.1%} -> {out_path}")


if __name__ == "__main__":
    main()
