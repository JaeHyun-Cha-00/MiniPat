# Prompt format shared by training and evaluation.
# The fine-tuned model, the base model, and the OpenAI model all get these exact messages;
# DeepL takes no prompt, only the source text and target language.

LANG_NAMES = {"ko": "Korean", "en": "English"}

SYSTEM_PROMPT = (
    "You are a professional patent translator. Translate the patent abstract faithfully, "
    "preserving technical terms, numbers, units, and reference numerals. "
    "Output only the translation."
)


def build_messages(text, direction):
    """Chat messages for one translation request, without the assistant turn."""
    src, tgt = direction.split("-")
    user = f"Translate the following {LANG_NAMES[src]} patent abstract into {LANG_NAMES[tgt]}.\n\n{text}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def to_examples(pair, directions):
    """One row of data/processed/*.jsonl -> one prompt/completion example per direction.

    Loss is taken on "completion" only (see train_lora.tokenize).
    """
    examples = []
    for direction in directions:
        src, tgt = direction.split("-")
        examples.append({
            "prompt": build_messages(pair[src], direction),
            "completion": [{"role": "assistant", "content": pair[tgt]}],
        })
    return examples
