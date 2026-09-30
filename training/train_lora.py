# JSONL splits -> LoRA/QLoRA adapter for one model (KO<->EN patent abstracts)
#
#   python training/train_lora.py --model qwen35_4b          # = training/configs/models/qwen35_4b.yaml
#   python training/train_lora.py --model qwen35_4b --limit 256 --max-steps 20   # smoke test
#   python training/train_lora.py --config path/to/any.yaml  # a config outside configs/models/
#
# With data.max_length unset, the script prints token-length percentiles and exits,
# so the limit is chosen from measured lengths rather than guessed.

import argparse
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import load_config, model_config_path  # noqa: E402
from models import load_base_model  # noqa: E402
from pairs import load_pairs  # noqa: E402
from prompts import to_examples  # noqa: E402

IGNORE_INDEX = -100  # label value the loss skips; used for prompt and padding tokens


def tokenize(example, tokenizer, template_kwargs):
    """Render with the chat template; loss only on the assistant turn.

    The prompt is rendered with add_generation_prompt=True, which is exactly what the
    model sees at inference. The completion is whatever the full conversation adds after
    it (translation + the family's end-of-turn token, e.g. <|im_end|>), so the model
    learns to stop on its own.
    """
    kw = {"tokenize": False, **template_kwargs}
    prompt = tokenizer.apply_chat_template(example["prompt"], add_generation_prompt=True, **kw)
    full = tokenizer.apply_chat_template(example["prompt"] + example["completion"], **kw)
    if not full.startswith(prompt):
        raise ValueError("Chat template renders the prompt differently with and without the answer; "
                         "prefix-based loss masking would be wrong.")
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(full[len(prompt):], add_special_tokens=False)["input_ids"]
    return {
        "input_ids": prompt_ids + completion_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(completion_ids)),
        "labels": [IGNORE_INDEX] * len(prompt_ids) + completion_ids,
    }


def build_dataset(pairs, cfg, tokenizer):
    examples = [ex for p in pairs for ex in to_examples(p, cfg["data"]["directions"])]
    ds = Dataset.from_list(examples)
    return ds.map(
        tokenize,
        fn_kwargs={"tokenizer": tokenizer, "template_kwargs": cfg["model"]["chat_template_kwargs"]},
        remove_columns=ds.column_names,  # keep only input_ids/attention_mask/labels
        desc="tokenize",
    )


def report_lengths(ds, name):
    lengths = np.array([len(x) for x in ds["input_ids"]])
    pcts = {p: int(np.percentile(lengths, p)) for p in (50, 90, 95, 99, 99.9)}
    print(f"[{name}] {len(lengths)} examples, tokens: "
          + ", ".join(f"p{p}={v}" for p, v in pcts.items()) + f", max={lengths.max()}")


def drop_long(ds, max_length, name):
    """Drop, never truncate: a truncated target teaches the model to stop mid-sentence."""
    kept = ds.filter(lambda x: len(x["input_ids"]) <= max_length, desc=f"filter {name}")
    print(f"[{name}] dropped {len(ds) - len(kept)} / {len(ds)} examples over {max_length} tokens")
    return kept


def load_model(cfg):
    m = cfg["model"]
    quant = None
    if m["qlora"]:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            # Unused vision/audio encoders (Qwen3.5 "visual", Gemma-4 "*_tower") stay unquantized.
            llm_int8_skip_modules=["visual", "vision_tower", "audio_tower", "lm_head"],
        )
    model = load_base_model(
        m["name"], dtype=torch.bfloat16, quantization_config=quant, attn_implementation="sdpa",
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    if m["qlora"]:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
        )
    elif cfg["train"]["gradient_checkpointing"]:
        model.enable_input_require_grads()  # frozen embeddings + checkpointing need this for LoRA grads

    lora = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()
    return model


def write_run_info(out_dir, cfg, args):
    """Everything needed to trace a checkpoint back to code, config, and environment."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)  # merged base + model settings, as actually used

    def pkg(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Inside the Docker image there is no .git; the SHA is baked in at build time instead.
        git_sha, git_dirty = os.environ.get("MINIPAT_GIT_SHA"), None
    info = {
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "cli_args": vars(args),
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": {p: pkg(p) for p in (
            "torch", "transformers", "peft", "datasets", "accelerate",
            "bitsandbytes", "flash-linear-attention", "causal-conv1d",
        )},
    }
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="name of a file in training/configs/models/, e.g. qwen35_4b")
    ap.add_argument("--config", help="path to a model config (alternative to --model)")
    ap.add_argument("--limit", type=int, help="use only the first N train/val pairs (smoke tests)")
    ap.add_argument("--max-steps", type=int, help="override: stop after N optimizer steps")
    ap.add_argument("--output-dir", help="override <checkpoint_root>/<model name>")
    args = ap.parse_args()
    if bool(args.model) == bool(args.config):
        ap.error("pass exactly one of --model or --config")

    cfg = load_config(args.config or model_config_path(args.model))
    cfg["train"]["output_dir"] = args.output_dir or os.path.join(cfg["checkpoint_root"], cfg["run_name"])
    set_seed(cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    val_n = min(cfg["data"]["val_subset"], args.limit or cfg["data"]["val_subset"])
    train_ds = build_dataset(load_pairs(cfg["data"], "train", args.limit), cfg, tokenizer)
    val_ds = build_dataset(load_pairs(cfg["data"], "val", val_n), cfg, tokenizer)
    report_lengths(train_ds, "train")
    report_lengths(val_ds, "val")

    max_length = cfg["data"]["max_length"]
    if max_length is None:
        sys.exit(f"data.max_length is unset for {cfg['run_name']}: pick it from the percentiles above, "
                 "set it in the model config, rerun.")
    train_ds = drop_long(train_ds, max_length, "train")
    val_ds = drop_long(val_ds, max_length, "val")

    # Without flash-linear-attention / causal-conv1d, transformers logs a "falling back to its
    # reference PyTorch implementation" warning on the first forward pass: correct but slow.
    model = load_model(cfg)

    out_dir = cfg["train"]["output_dir"]
    write_run_info(out_dir, cfg, args)
    training_args = TrainingArguments(
        **cfg["train"],  # keys in the YAML train: section are TrainingArguments names
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        bf16=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        train_sampling_strategy="group_by_length",  # batches of similar length -> less padding
        remove_unused_columns=False,  # dataset already holds only model inputs
        max_steps=args.max_steps or -1,
    )
    # Right padding: padded positions come after the real tokens, so they cannot leak into
    # the causal/recurrent state of any real token, and their labels are IGNORE_INDEX.
    tokenizer.padding_side = "right"
    collator = DataCollatorForSeq2Seq(
        tokenizer, padding=True, label_pad_token_id=IGNORE_INDEX, pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.train()

    # load_best_model_at_end restored the lowest-eval_loss adapter; save that one as "final".
    final_dir = os.path.join(out_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved adapter to {final_dir}")


if __name__ == "__main__":
    main()
