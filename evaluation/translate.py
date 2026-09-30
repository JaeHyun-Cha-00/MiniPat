# Test split -> translations from one system, with per-example latency and usage.
#
#   python evaluation/translate.py --system base --model qwen35_4b       # GPU pod; --model is a
#   python evaluation/translate.py --system finetuned --model qwen35_4b  #   training/configs/models/ name
#   python evaluation/translate.py --system openai [--model ID]    # needs OPENAI_API_KEY
#   python evaluation/translate.py --system deepl                  # needs DEEPL_AUTH_KEY
#
#   --limit 5                            pre-flight check before paying for a full run
#   --batch-size 1 --limit 50 --tag latency   single-request latency run for the local models
#                     (-> qwen35_4b-base__latency.jsonl; "__" because model names contain dots)
#
# Output: <output_dir>/<name>.jsonl, one line per (pair, direction), appended as it goes.
# Rerunning skips ids already in the file, so a crash never pays for the same call twice.
# Scoring is separate (score.py) so paid calls happen once and metrics can be rerun for free.

import argparse
import json
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
from config import load_config, model_config_path  # noqa: E402
from pairs import load_pairs  # noqa: E402
from prompts import build_messages  # noqa: E402

# DeepL wants its own language codes; English output must name a variant.
DEEPL_SOURCE = {"ko": "KO", "en": "EN"}
DEEPL_TARGET = {"ko": "KO", "en": "EN-US"}


def make_jobs(pairs, directions):
    jobs = []
    for p in pairs:
        for direction in directions:
            src, tgt = direction.split("-")
            jobs.append({
                "id": f"{p['family_id']}:{direction}",
                "family_id": p["family_id"],
                "direction": direction,
                "en_source": p["en_source"],   # WO vs US, for the split breakdown in score.py
                "source": p[src],
                "reference": p[tgt],
            })
    return jobs


class LocalModel:
    """A training/configs/models/ model via HF generate, optionally with its LoRA adapter merged in."""

    def __init__(self, model_cfg, gen_cfg, adapter=None):
        import torch
        from models import load_base_model, stop_token_ids
        from transformers import AutoTokenizer

        self.torch = torch
        name = model_cfg["model"]["name"]
        self.template_kwargs = model_cfg["model"]["chat_template_kwargs"]
        self.tok = AutoTokenizer.from_pretrained(name)
        # Left padding so every row's generation starts right after its own prompt. Attention
        # layers mask the pads; Qwen3.5's Gated DeltaNet layers zero them via attention_mask, so
        # they don't enter the recurrent state either (checked: same logits alone vs. batched).
        self.tok.padding_side = "left"
        model = load_base_model(name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()  # merged: no LoRA overhead
        self.model = model.eval()
        self.max_new_tokens = gen_cfg["max_new_tokens"]
        self.stop_ids = stop_token_ids(self.tok, self.model)

    def translate(self, batch):
        prompts = [
            self.tok.apply_chat_template(
                build_messages(j["source"], j["direction"]),
                add_generation_prompt=True, tokenize=False, **self.template_kwargs,
            )
            for j in batch
        ]
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        self._sync()
        t0 = time.perf_counter()
        with self.torch.inference_mode():
            out = self.model.generate(
                **enc, do_sample=False, max_new_tokens=self.max_new_tokens,
                eos_token_id=self.stop_ids, pad_token_id=self.tok.pad_token_id,
            )
        self._sync()
        elapsed = time.perf_counter() - t0

        results = []
        for row in out[:, enc["input_ids"].shape[1]:].tolist():
            n = next((i + 1 for i, t in enumerate(row) if t in self.stop_ids), len(row))
            results.append({
                "hypothesis": self.tok.decode(row[:n], skip_special_tokens=True).strip(),
                "output_tokens": n,
                "truncated": n == self.max_new_tokens and row[n - 1] not in self.stop_ids,
                "latency_s": elapsed,  # wall time of the whole batch; score.py divides by batch_size
            })
        return results

    def _sync(self):
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()


class OpenAIModel:
    def __init__(self, cfg, model_id):
        from openai import OpenAI
        self.client = OpenAI(max_retries=5)  # reads OPENAI_API_KEY
        self.model = model_id
        self.cfg = cfg

    def translate(self, batch):
        (job,) = batch
        kwargs = {}
        if self.cfg["reasoning_effort"]:
            kwargs["reasoning"] = {"effort": self.cfg["reasoning_effort"]}
        t0 = time.perf_counter()
        resp = self.client.responses.create(
            model=self.model,
            input=build_messages(job["source"], job["direction"]),
            max_output_tokens=self.cfg["max_output_tokens"],
            store=False,  # don't keep test data on OpenAI's side
            **kwargs,
        )
        elapsed = time.perf_counter() - t0  # includes network round trip, as a real client would see
        u = resp.usage
        return [{
            "hypothesis": resp.output_text.strip(),
            "model": self.model,  # score.py looks up the price by this id
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,  # billed; includes reasoning tokens
            "reasoning_tokens": u.output_tokens_details.reasoning_tokens,
            "truncated": resp.status == "incomplete",
            "latency_s": elapsed,
        }]


class DeepLModel:
    def __init__(self, cfg):
        import deepl
        self.client = deepl.DeepLClient(os.environ["DEEPL_AUTH_KEY"])
        self.cfg = cfg

    def translate(self, batch):
        (job,) = batch
        src, tgt = job["direction"].split("-")
        t0 = time.perf_counter()
        r = self.client.translate_text(
            job["source"], source_lang=DEEPL_SOURCE[src], target_lang=DEEPL_TARGET[tgt],
            model_type=self.cfg["model_type"],
        )
        elapsed = time.perf_counter() - t0
        return [{
            "hypothesis": r.text.strip(),
            "billed_characters": r.billed_characters,
            "model_type_used": r.model_type_used,
            "truncated": False,
            "latency_s": elapsed,
        }]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["base", "finetuned", "openai", "deepl"])
    ap.add_argument("--config", default="evaluation/config.yaml")
    ap.add_argument("--model", help="base/finetuned: training/configs/models/ name (required); "
                                    "openai: model id (default from config)")
    ap.add_argument("--adapter", help="finetuned: adapter dir (default <checkpoint_root>/<model>/final)")
    ap.add_argument("--limit", type=int, help="first N test pairs only")
    ap.add_argument("--batch-size", type=int, help="local systems only; API systems always send 1")
    ap.add_argument("--tag", help="suffix for the output file, e.g. 'latency'")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_cfg = load_config(cfg["data_config"])["data"]

    batch_size = 1
    if args.system in ("base", "finetuned"):
        if not args.model:
            ap.error("--model is required for base/finetuned, e.g. --model qwen35_4b")
        model_cfg = load_config(model_config_path(args.model))
        batch_size = args.batch_size or cfg["local"]["batch_size"]
        adapter = None
        if args.system == "finetuned":
            adapter = args.adapter or os.path.join(model_cfg["checkpoint_root"], model_cfg["run_name"], "final")
        model, name = LocalModel(model_cfg, cfg["local"], adapter), f"{args.model}-{args.system}"
    elif args.system == "openai":
        model_id = args.model or cfg["openai"]["model"]
        model, name = OpenAIModel(cfg["openai"], model_id), f"openai-{model_id}"
    else:
        model, name = DeepLModel(cfg["deepl"]), "deepl"
    if args.tag:
        name += f"__{args.tag}"

    os.makedirs(cfg["output_dir"], exist_ok=True)
    out_path = os.path.join(cfg["output_dir"], f"{name}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f}

    jobs = make_jobs(load_pairs(data_cfg, cfg["split"], args.limit), cfg["directions"])
    todo = [j for j in jobs if j["id"] not in done]
    # Similar lengths per batch -> less padding and fewer rows idling while the longest one finishes.
    todo.sort(key=lambda j: len(j["source"]))
    print(f"{name}: {len(jobs)} jobs, {len(done)} already done, {len(todo)} to run (batch size {batch_size})")

    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            for job, res in zip(batch, model.translate(batch)):
                f.write(json.dumps({**job, **res, "system": name, "batch_size": len(batch)}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {min(i + batch_size, len(todo))}/{len(todo)}", end="\r", flush=True)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
