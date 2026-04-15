"""Training-time loss masking for reasoning blocks."""

import re
from enum import IntFlag

from datasets import Dataset

from thinkpack._model import TemplateStyle, _Tokenizer, detect_model


# pytorch's CrossEntropyLoss uses ignore_index=-100 by default, and all major
# training frameworks (transformers Trainer, trl SFTTrainer, unsloth) inherit
# this default — so -100 is the correct value unless the trainer is configured
# otherwise. exposed as a parameter on mask() for the rare case where it differs.
_DEFAULT_IGNORE_INDEX = -100


class Mask(IntFlag):
    """
    Sections of the training sequence to mask from the loss.

    Combine sections with | to mask multiple parts at once:
        Mask.THINK              — mask only the think block (most common)
        Mask.PROMPT | Mask.THINK — mask prompt and think (train on response only)

    PROMPT covers the user instruction. THINK covers the full reasoning block
    including its opening and closing tags. RESPONSE covers the model's answer.
    Masking RESPONSE is unusual (nothing useful remains to train on) but valid.
    """

    PROMPT = 1
    THINK = 2
    RESPONSE = 4


def _build_assistant_message(
    record: dict[str, str],
    style: TemplateStyle,
    open_tag: str,
) -> dict[str, str]:
    """
    Build the assistant message dict from a training record.

    For NATIVE templates, passes reasoning as a separate reasoning_content field.
    For INLINE and PREFIXED templates, wraps reasoning in inline reasoning tags.
    The presence of a "reasoning" key (even if empty) controls whether the think
    block appears in the sequence — required when masking so the model sees the
    same context at training time as at inference time.

    Returns an assistant message dict ready to pass to apply_chat_template.
    """
    # use None sentinel to distinguish "key absent" from "key present but empty"
    reasoning_raw = record.get("reasoning", None)
    response = record["response"]

    message: dict[str, str] = {"role": "assistant"}

    if reasoning_raw is not None and style == TemplateStyle.NATIVE:
        # template natively handles reasoning via a dedicated field (e.g. Qwen3)
        message["content"] = response
        message["reasoning_content"] = reasoning_raw.strip()
    elif reasoning_raw is not None:
        # derive the closing tag from the opening tag, e.g. <think> -> </think>
        close_tag = open_tag.replace("<", "</", 1)
        reasoning = reasoning_raw.strip()
        message["content"] = f"{open_tag}\n{reasoning}\n{close_tag}\n{response}"
    else:
        message["content"] = response

    return message


def _tokenize_prefix(
    tokenizer: _Tokenizer,
    text: str,
    max_seq_length: int,
) -> int:
    """
    Tokenize a text prefix and return its token count.

    Used to locate section boundaries within the full token sequence by
    tokenizing the text up to a known character position.

    Returns the number of tokens in the prefix.
    """
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_length,
        )
    )


def _tokenize_record(
    record: dict[str, str],
    tokenizer: _Tokenizer,
    style: TemplateStyle,
    open_tag: str,
    max_seq_length: int,
    masked: Mask,
    ignore_index: int,
) -> dict[str, list[int]]:
    """
    Tokenize a single training record and apply label masking.

    Locates the PROMPT / THINK / RESPONSE boundaries in the token sequence by
    tokenizing text prefixes (rather than using add_generation_prompt=True). This
    avoids a subtle issue with PREFIXED templates: the generation prompt already
    ends with <think>, so using it as a prefix boundary would leave the opening
    tag trainable while masking the closing tag — teaching the model to "open but
    never close" the reasoning block.

    Each section flagged in `masked` has its labels set to _IGNORE_INDEX so
    PyTorch's cross-entropy ignores those tokens during loss computation.

    Returns a dict with input_ids, labels, and attention_mask.
    """
    messages = [
        {"role": "user", "content": record["instruction"]},
        _build_assistant_message(
            record=record,
            style=style,
            open_tag=open_tag,
        ),
    ]
    full_text_raw: str | list[int] = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    # some tokenizers return token ids despite tokenize=False — decode them
    full_text = (
        tokenizer.decode(full_text_raw)
        if isinstance(full_text_raw, list)
        else full_text_raw
    )
    input_ids = tokenizer.encode(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
    )

    # default: all tokens contribute to the loss
    labels = list(input_ids)

    if not masked:
        # no sections to mask — return labels unchanged
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }

    # find the opening reasoning tag to locate the think block boundary
    open_match = re.search(re.escape(open_tag), full_text)
    think_start = (
        _tokenize_prefix(
            tokenizer=tokenizer,
            text=full_text[: open_match.start()],
            max_seq_length=max_seq_length,
        )
        if open_match is not None
        else None  # no think block present in this record
    )

    # locate the response boundary (rfind to handle response text in the instruction)
    response_start_char = full_text.rfind(record["response"])
    response_start = _tokenize_prefix(
        tokenizer=tokenizer,
        text=full_text[:response_start_char],
        max_seq_length=max_seq_length,
    )

    # mask each requested section independently
    if Mask.PROMPT in masked:
        # mask everything from the start up to the think block (or response if no think)
        prompt_end = think_start if think_start is not None else response_start
        for i in range(prompt_end):
            labels[i] = ignore_index

    if Mask.THINK in masked and think_start is not None:
        # mask the full reasoning block including its opening and closing tags
        for i in range(think_start, response_start):
            labels[i] = ignore_index

    if Mask.RESPONSE in masked:
        # mask the response tokens
        for i in range(response_start, len(labels)):
            labels[i] = ignore_index

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def mask(
    records: list[dict[str, str]],
    tokenizer: _Tokenizer,
    masked: Mask | None = Mask.THINK,
    max_seq_length: int = 32768,
    ignore_index: int = _DEFAULT_IGNORE_INDEX,
    tag: str | None = None,
) -> Dataset:
    """
    Format training records into a pretokenized dataset with selected sections masked.

    Each record must have "instruction" and "response" keys. An optional "reasoning"
    key provides think block content — if absent when masking is applied, an empty
    reasoning block is injected so training context matches inference time.

    Template style (INLINE, NATIVE, PREFIXED) is detected automatically from the
    tokenizer. Combine Mask flags with | to mask multiple sections at once (see
    the Mask class for details). Pass masked=None to train on all tokens.

    Returns a HuggingFace Dataset with input_ids, labels, and attention_mask columns.
    """
    model_info = detect_model(tokenizer=tokenizer)
    # user-supplied tag overrides the detected default (useful for INLINE models
    # whose tag differs from the <think> default, e.g. <reasoning>)
    open_tag = f"<{tag}>" if tag is not None else model_info.open_tag

    # normalise None to an empty Mask so downstream logic is consistent
    effective_masked = masked if masked is not None else Mask(0)

    # when masking is active, inject an empty "reasoning" key for records that lack one
    # so the think block appears in the sequence — required for training/inference context
    # alignment on PREFIXED models that always emit think blocks at inference time
    if effective_masked:
        records = [
            record if "reasoning" in record else {**record, "reasoning": ""}
            for record in records
        ]

    all_input_ids = []
    all_labels = []
    all_attention_mask = []

    for record in records:
        result = _tokenize_record(
            record=record,
            tokenizer=tokenizer,
            style=model_info.style,
            open_tag=open_tag,
            max_seq_length=max_seq_length,
            masked=effective_masked,
            ignore_index=ignore_index,
        )
        all_input_ids.append(result["input_ids"])
        all_labels.append(result["labels"])
        all_attention_mask.append(result["attention_mask"])

    return Dataset.from_dict(
        {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "attention_mask": all_attention_mask,
        }
    )
