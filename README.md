# MiniPat

LoRA fine-tuning of **small open models (0.8B–2B)** for Korean ↔ English translation of patent abstracts. Each model is compared before and after fine-tuning, and against an OpenAI GPT-6 model and DeepL, on quality, cost, and latency.

> **Status: in progress.** The data pipeline is done. The training and evaluation code is written and tested on CPU with tiny stand-ins of each model family, but has not yet been run on a GPU. **There are no results yet**, and none will be reported here until the runs are done.

## Repository layout

```
data_pipeline/   BigQuery -> CSV -> deduplicated train/val/test JSONL -> private HF dataset
training/        LoRA/QLoRA fine-tuning (train_lora.py), prompt format, per-model configs
evaluation/      translate.py (one system per run), score.py (metrics, cost, latency), plot.py (figures)
requirements/    data.txt, train.txt, eval.txt, score.txt (pinned per environment)
Dockerfile       training + inference environment (CUDA 12.6, PyTorch 2.14)
```

## Data

Pairs come from [Google Patents Public Data](https://console.cloud.google.com/marketplace/product/google_patents_public_datasets/google-patents-public-data) (`patents-public-data.patents.publications`). A Korean publication with a Korean abstract is joined to a US or WO publication in the same **patent family** that has an English abstract.

- **One pair per family.** A Korean application's publication and registration documents share a family, so without this rule the same abstract could appear twice. The SQL keeps one row per `family_id`, and `prep_dataset.py` checks family, publication number, and text duplicates again.
- **Filters:** length limits for each language, and an EN/KO character-length ratio between 0.8 and 6.0 to drop misaligned pairs.
- **Splits:** 28,146 train / 1,000 val / 500 test pairs. No family appears in more than one split. Each split is about 62% WO and 38% US.
- **Caveat:** for WO families the Korean abstract is often a translation of the English one (or of a third language), so KO→EN test references can be closer to the source than a naturally written English abstract would be. `score.py` reports WO and US separately for this reason.
- **Not every pair is a translation.** US abstracts are often written separately from the Korean one: same invention, different content and structure. `data_pipeline/score_alignment.py` scores each pair by LaBSE similarity (unrelated pairs ≈ 0.44, close translations ≈ 0.9). Below 0.80 are 11.4% of train pairs (WO 4.1%, US 23.8%). `score.py` also reports an "aligned" test subset (LaBSE ≥ 0.80, 442 of 500 pairs), and `qwen35_0.8b_aligned` trains on the 24,949 aligned train pairs only.

```bash
pip install -r requirements/data.txt
python data_pipeline/fetch_patent_pairs.py --project <gcp-project> --dry-run   # estimate bytes scanned
python data_pipeline/fetch_patent_pairs.py --project <gcp-project>            # -> data/raw/patent_pairs.csv
python data_pipeline/prep_dataset.py                                          # -> data/processed/{train,val,test}.jsonl
pip install -r requirements/align.txt
python data_pipeline/score_alignment.py                                       # -> data/processed/labse_{train,val,test}.jsonl
python data_pipeline/push_dataset.py --repo <user>/minipat-ko-en              # private HF dataset
```

The published splits come from an earlier `ORDER BY RAND()` sample. The query now orders by a hash of `family_id`, so reruns give identical samples, but they won't match the published splits. The HF dataset is the canonical copy.

## Training

`training/train_lora.py` fine-tunes one model on both directions (KO→EN and EN→KO) from each pair. Shared hyperparameters are in `training/configs/base.yaml`; each model has a small file in `training/configs/models/` with only what differs: its Hugging Face id, LoRA target layers, and `max_length`.

| Config | Model | Size | Why it's in the comparison |
|---|---|---|---|
| `qwen35_0.8b` | `Qwen/Qwen3.5-0.8B` | 0.8B | same family at two sizes: how quality scales with size |
| `qwen35_2b` | `Qwen/Qwen3.5-2B` | 2B | |
| `exaone4_1.2b` | `LGAI-EXAONE/EXAONE-4.0-1.2B` | 1.2B | Korean-focused model (LG AI Research) |
| `gemma4_e2b` | `google/gemma-4-E2B-it` | 2B effective | a different family at a similar size |

- **Chat models, thinking off.** Each is the instruction-tuned version, so the "before fine-tuning" baseline can already follow a translation prompt.
- **LoRA targets:** attention and MLP projections of the text decoder only. Qwen3.5 and Gemma-4 ship with vision (and, for Gemma, audio) encoders; those stay frozen. For Qwen3.5's Gated DeltaNet linear-attention layers, `in_proj_qkv` / `in_proj_z` / `out_proj` are trained and the decay/gating parameters stay frozen.
- **Loss masking:** loss is computed only on the assistant turn (the translation plus the model's end-of-turn token). Examples longer than `max_length` are dropped, never truncated.
- **Licenses:** Qwen3.5 is Apache-2.0. EXAONE-4.0 allows research and educational use only, so its fine-tuned adapter must not be used commercially. Gemma-4 is gated: accept its terms on Hugging Face before downloading.
- **Not included:** Kakao's Kanana models. Kanana-2 needs custom model code written for an older `transformers`, and Kanana-1.5's config is rejected by `transformers` 5.x (hidden size not divisible by head count), so neither loads with the pinned versions.
- **Precision:** bf16 LoRA by default; set `model.qlora: true` for 4-bit QLoRA on smaller GPUs.
- **Traceability:** each run folder records the config, git SHA, CLI arguments, GPU, and package versions (`run_info.json`).

```bash
pip install -r requirements/train.txt
# smoke test first
python training/train_lora.py --model qwen35_0.8b --limit 256 --max-steps 20 --output-dir /workspace/checkpoints/smoke
# full runs -> /workspace/checkpoints/<model>/final
for m in qwen35_0.8b qwen35_2b exaone4_1.2b gemma4_e2b; do
    python training/train_lora.py --model $m
done
```

## Evaluation

Every system translates the 500-pair test split in both directions:

| System | How it runs |
|---|---|
| Each model, fine-tuned | base model + merged LoRA adapter, HF `generate`, greedy |
| Each model, before fine-tuning | same prompt, no adapter |
| OpenAI GPT-6 (`gpt-6.1-sol` by default) | Responses API, same prompt |
| DeepL | `quality_optimized` model; no prompt |

- **Quality:** BLEU (sacrebleu, `flores200` tokenization, so Korean output is scored the same way as English), chrF++, and COMET (`Unbabel/wmt22-comet-da`). Each is reported overall, per direction, and per WO/US.
- **Cost:** USD per 1,000 translations, from token counts (OpenAI), billed characters (DeepL), or GPU time × hourly price (local models). Prices are entered by hand in `evaluation/config.yaml` together with the date they were checked.
- **Latency:** single-request p50/p95. Local models are batched for the full run, so their latency comes from a separate batch-size-1 run.

```bash
pip install -r requirements/train.txt -r requirements/eval.txt
python evaluation/translate.py --system finetuned --model qwen35_0.8b --limit 5   # pre-flight
for m in qwen35_0.8b qwen35_2b exaone4_1.2b gemma4_e2b; do
    python evaluation/translate.py --system base --model $m
    python evaluation/translate.py --system finetuned --model $m
    python evaluation/translate.py --system finetuned --model $m --batch-size 1 --limit 50 --tag latency
done
python evaluation/translate.py --system openai                 # OPENAI_API_KEY
python evaluation/translate.py --system deepl                  # DEEPL_AUTH_KEY

# COMET needs transformers<5, so scoring and plotting use their own environment
python -m venv .venv-score && .venv-score/bin/pip install -r requirements/score.txt
.venv-score/bin/python evaluation/score.py evaluation/outputs/*.jsonl    # -> evaluation/results/scores.json
.venv-score/bin/python evaluation/plot.py                                 # -> evaluation/results/figures/
```

`plot.py` draws two figures: quality vs. model size (before and after fine-tuning, with the APIs as reference lines), and cost vs. quality for every system with a price.

`translate.py` appends each result as it goes and skips finished items when rerun, so an interrupted run never pays for the same API call twice.

## Environment

```bash
docker build --build-arg GIT_SHA=$(git rev-parse HEAD) -t minipat .
docker run --gpus all -e HF_TOKEN -v /workspace:/workspace minipat \
    python training/train_lora.py --model qwen35_2b
```

The image targets RunPod: the network volume at `/workspace` holds the Hugging Face cache and checkpoints. RunPod pods can't build images, so build and push the image from another machine; alternatively, install `requirements/train.txt` on RunPod's PyTorch template, which uses the same pinned versions.

## Results

Pending: no training or evaluation runs yet.
