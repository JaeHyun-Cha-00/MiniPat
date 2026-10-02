#!/usr/bin/env bash
# Full GPU pipeline on one pod: for each model, train the LoRA adapter, then translate the test
# split with the base model, the fine-tuned model, and a batch-size-1 latency run.
#
#   nohup bash run_all.sh > /workspace/logs/run_all.log 2>&1 &
#   tail -f /workspace/logs/run_all.log
#
# Smallest models first, so a time limit cuts off the most expensive runs, not the cheap ones.
# Rerunning is safe: models with a final adapter are not retrained, and translate.py skips
# items already in its output file. A failing step is logged and the script moves on.

cd "$(dirname "$0")"
MODELS=${MODELS:-"qwen35_0.8b exaone4_1.2b qwen35_2b gemma4_e2b qwen35_4b"}
LOG_DIR=/workspace/logs
mkdir -p "$LOG_DIR"

step() {  # step <log name> <command...>
    local name=$1; shift
    echo "[$(date '+%F %T')] start $name"
    if "$@" > "$LOG_DIR/$name.log" 2>&1; then
        echo "[$(date '+%F %T')] done  $name"
    else
        echo "[$(date '+%F %T')] FAIL  $name (see $LOG_DIR/$name.log)"
        return 1
    fi
}

for m in $MODELS; do
    if [ -d "/workspace/checkpoints/$m/final" ]; then
        echo "[$(date '+%F %T')] skip  train_$m (final adapter exists)"
    else
        step "train_$m" python training/train_lora.py --model "$m" || continue
    fi
    step "eval_base_$m"      python evaluation/translate.py --system base --model "$m"
    step "eval_finetuned_$m" python evaluation/translate.py --system finetuned --model "$m"
    step "eval_latency_$m"   python evaluation/translate.py --system finetuned --model "$m" \
                                 --batch-size 1 --limit 50 --tag latency
done
echo "[$(date '+%F %T')] all done"
