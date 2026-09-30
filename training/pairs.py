# Loads the KR<->EN pairs for one split. Shared by training and evaluation so both read
# the same data source; imports nothing heavy, so API-only evaluation runs stay light.

import json
import os


def load_pairs(data_cfg, split, limit=None):
    """Raw {family_id, ko, en, en_source, ...} rows, from the HF Hub dataset or local JSONL.

    data_cfg is the data: section of training/configs/*.yaml.
    """
    if data_cfg["hf_dataset"]:
        from datasets import load_dataset
        rows = list(load_dataset(data_cfg["hf_dataset"], split=split))
    else:
        path = os.path.join(data_cfg["local_dir"], f"{split}.jsonl")
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
    return rows[:limit] if limit else rows
