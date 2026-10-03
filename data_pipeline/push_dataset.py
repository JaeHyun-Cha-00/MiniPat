# data/processed/*.jsonl -> private Hugging Face dataset, so RunPod pods can load the
# exact splits without copying files around (data/ is gitignored).
#
#   hf auth login        # or export HF_TOKEN=...
#   python data_pipeline/push_dataset.py --repo <user>/minipat-ko-en
#
# Then set data.hf_dataset in training/configs/base.yaml to the same repo id.
# If data_pipeline/score_alignment.py has been run, its labse_sim scores are added as a column.

import argparse
import hashlib
import json
import os

from datasets import load_dataset

IN_DIR = "data/processed"
SPLITS = ("train", "val", "test")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF dataset repo id, e.g. user/minipat-ko-en")
    args = ap.parse_args()

    files = {s: os.path.join(IN_DIR, f"{s}.jsonl") for s in SPLITS}
    ds = load_dataset("json", data_files=files)
    for s in SPLITS:
        print(f"{s}: {len(ds[s])} pairs, sha256 {sha256(files[s])[:16]}")

    labse_files = {s: os.path.join(IN_DIR, f"labse_{s}.jsonl") for s in SPLITS}
    if all(os.path.exists(p) for p in labse_files.values()):
        for s, path in labse_files.items():
            with open(path) as f:
                sim = {r["family_id"]: r["labse_sim"] for r in map(json.loads, f)}
            ds[s] = ds[s].add_column("labse_sim", [sim[fid] for fid in ds[s]["family_id"]])
        print("Added labse_sim column")
    else:
        print("No labse_*.jsonl for every split; pushing without labse_sim")

    # private=True: the abstracts are public patent data, but the curated pairs are this project's work.
    ds.push_to_hub(args.repo, private=True)
    print(f"Pushed to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
