# Model-family differences that can be read from the checkpoint itself, so the per-model
# configs don't have to spell them out.

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText


def load_base_model(name, **kwargs):
    """Load a checkpoint with the class it was saved as.

    Qwen3.5 and Gemma-4 ship as vision(-audio)-language models; loading them as such keeps
    module names (model.language_model.layers...) matching the LoRA target regexes. Their
    vision/audio encoders stay frozen and are never fed anything. Text-only models such as
    EXAONE load as plain causal LMs.
    """
    multimodal = hasattr(AutoConfig.from_pretrained(name), "vision_config")
    cls = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
    return cls.from_pretrained(name, **kwargs)


def stop_token_ids(tokenizer, model):
    """Every token that ends a reply for this model family.

    The end-of-turn token differs by family (Qwen <|im_end|>, EXAONE [|endofturn|],
    Gemma <turn|>), and a checkpoint lists it in either the tokenizer's eos or the
    generation config's eos list, not always both. The union covers every case.
    """
    ids = {tokenizer.eos_token_id}
    gen_eos = model.generation_config.eos_token_id
    ids.update(gen_eos if isinstance(gen_eos, list) else [gen_eos])
    return sorted(i for i in ids if i is not None)
