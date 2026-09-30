# Loads a model config (training/configs/models/*.yaml) merged onto its base (configs/base.yaml).

import os

import yaml


def _merge(base, override):
    """Recursive dict merge; values in override win, nested sections merge key by key."""
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path):
    """Merged config, plus "run_name" = the model file's stem (e.g. qwen35_0.8b).

    run_name keys everything downstream: the checkpoint folder and the evaluation file names.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("base", None)
    if base:
        cfg = _merge(load_config(os.path.join(os.path.dirname(path), base)), cfg)
    cfg["run_name"] = os.path.splitext(os.path.basename(path))[0]
    return cfg


def model_config_path(run_name):
    """training/configs/models/<run_name>.yaml, so scripts can take a short name like qwen35_0.8b."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "models", f"{run_name}.yaml")
