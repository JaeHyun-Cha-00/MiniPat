# Saved translations -> quality, cost, and latency per system.
# Runs in its own environment: unbabel-comet needs transformers<5, the training image has 5.x.
#
#   python -m venv .venv-score && .venv-score/bin/pip install -r requirements/score.txt
#   .venv-score/bin/python evaluation/score.py evaluation/outputs/*.jsonl [--no-comet]
#
# Quality (BLEU, chrF++, COMET) is reported overall, per direction, and per en_source (WO/US).
# Latency comes only from rows translated one request at a time (batch_size == 1): the API
# systems, and the local systems' "--tag latency" runs. Cost comes from the full runs.

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import yaml
from sacrebleu.metrics import BLEU, CHRF

COMET_MODEL = "Unbabel/wmt22-comet-da"

# flores200 is a SentencePiece tokenizer covering Korean and English alike, so BLEU is computed
# the same way in both directions instead of depending on whitespace in Korean output.
bleu = BLEU(tokenize="flores200")
chrf = CHRF(word_order=2)  # word_order=2 -> chrF++


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def comet_scores(rows, batch_size=16):
    """Segment-level COMET for every row; aggregated per group later."""
    import torch
    from comet import download_model, load_from_checkpoint

    model = load_from_checkpoint(download_model(COMET_MODEL))
    data = [{"src": r["source"], "mt": r["hypothesis"], "ref": r["reference"]} for r in rows]
    out = model.predict(data, batch_size=batch_size, gpus=1 if torch.cuda.is_available() else 0)
    return out.scores


def quality(rows, comet=None):
    hyps = [r["hypothesis"] for r in rows]
    refs = [[r["reference"] for r in rows]]
    res = {
        "n": len(rows),
        "bleu": round(bleu.corpus_score(hyps, refs).score, 2),
        "chrf++": round(chrf.corpus_score(hyps, refs).score, 2),
        "truncated": sum(bool(r.get("truncated")) for r in rows),
    }
    if comet is not None:
        res["comet"] = round(float(np.mean(comet)), 4)
    return res


def cost_per_1k(rows, prices):
    """USD per 1,000 translations, or None when the needed price isn't filled in."""
    r0 = rows[0]
    if "input_tokens" in r0:  # OpenAI
        p = (prices.get("openai") or {}).get(r0["model"]) or {}
        if p.get("input") is None or p.get("output") is None:
            return None
        usd = sum(r["input_tokens"] * p["input"] + r["output_tokens"] * p["output"] for r in rows) / 1e6
    elif "billed_characters" in r0:  # DeepL
        if prices.get("deepl_per_million_chars") is None:
            return None
        usd = sum(r["billed_characters"] for r in rows) * prices["deepl_per_million_chars"] / 1e6
    else:  # local: GPU time. latency_s is per batch, so each row carries 1/batch_size of it.
        if prices.get("gpu_per_hour") is None:
            return None
        usd = sum(r["latency_s"] / r["batch_size"] for r in rows) / 3600 * prices["gpu_per_hour"]
    return round(1000 * usd / len(rows), 4)


def latency(rows):
    single = [r["latency_s"] for r in rows if r["batch_size"] == 1]
    if not single:
        return None
    return {"n": len(single), "p50_s": round(float(np.percentile(single, 50)), 3),
            "p95_s": round(float(np.percentile(single, 95)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--config", default="evaluation/config.yaml")
    ap.add_argument("--no-comet", action="store_true", help="skip COMET (slow without a GPU)")
    ap.add_argument("--out", default="evaluation/results/scores.json")
    args = ap.parse_args()

    with open(args.config) as f:
        prices = yaml.safe_load(f)["prices"]
    if not prices.get("checked"):
        print("NOTE: prices.checked is unset in the config; costs are n/a until prices are filled in.")

    # "base__latency.jsonl" is a latency-only run of "base": merge by the name before "__".
    by_system = defaultdict(lambda: {"full": [], "latency": []})
    for path in args.files:
        name = os.path.basename(path).removesuffix(".jsonl")
        system, _, tag = name.partition("__")
        by_system[system]["latency" if tag == "latency" else "full"].extend(read_jsonl(path))

    results = {}
    for system, parts in sorted(by_system.items()):
        rows = parts["full"]
        if not rows:
            print(f"{system}: only a latency run, skipping quality/cost")
            continue
        comet = None if args.no_comet else comet_scores(rows)
        groups = {"all": list(range(len(rows)))}
        for i, r in enumerate(rows):
            groups.setdefault(r["direction"], []).append(i)
            groups.setdefault(f"{r['direction']}/{r['en_source']}", []).append(i)
        results[system] = {
            "quality": {
                g: quality([rows[i] for i in idx], None if comet is None else [comet[i] for i in idx])
                for g, idx in sorted(groups.items())
            },
            "cost_usd_per_1k": cost_per_1k(rows, prices),
            "latency": latency(parts["latency"] + rows),
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"prices_checked": prices.get("checked"), "comet_model": None if args.no_comet else COMET_MODEL,
                   "systems": results}, f, indent=2)

    cols = ["bleu", "chrf++"] + ([] if args.no_comet else ["comet"])
    print(f"\n| system | direction | n | {' | '.join(cols)} | $/1k | latency p50 s |")
    print("|" + "---|" * (len(cols) + 5))
    for system, r in results.items():
        lat = r["latency"]["p50_s"] if r["latency"] else "n/a"
        cost = r["cost_usd_per_1k"] if r["cost_usd_per_1k"] is not None else "n/a"
        for d in ("ko-en", "en-ko"):
            q = r["quality"].get(d)
            if q:
                print(f"| {system} | {d} | {q['n']} | " + " | ".join(str(q[c]) for c in cols) + f" | {cost} | {lat} |")
    print(f"\nWrote {args.out} (includes per-WO/US breakdown)")


if __name__ == "__main__":
    main()
