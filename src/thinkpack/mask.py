"""Training-time loss masking for reasoning blocks."""

import re
from enum import IntFlag

from datasets import Dataset

from thinkpack.chat import apply_chat_template as _apply_chat_template
from thinkpack.model import ModelInfo, _Tokenizer, get_model_info


# pytorch's CrossEntropyLoss uses ignore_index=-100 by default, and all major
# training frameworks (transformers Trainer, trl SFTTrainer, unsloth) inherit
# this default — exposed as a parameter for the rare case where it differs
_DEFAULT_IGNORE_INDEX = -100


class MaskType(IntFlag):
    """
    Sections of the training sequence to mask from the loss.

    Combine with | to mask multiple sections at once:
        MaskType.THINK                    — mask only the think block (most common)
        MaskType.PROMPT | MaskType.THINK  — train on response only

    PROMPT covers the user instruction. THINK covers the full reasoning block
    including its opening and closing tags. RESPONSE covers the model's answer.
    Masking RESPONSE is unusual (nothing useful remains to train on) but valid.
    """

    PROMPT = 1
    THINK = 2
    RESPONSE = 4


def _tokenize_prefix(
    tokenizer: _Tokenizer,
    text: str,
    max_seq_length: int,
) -> int:
    """
    Tokenize a text prefix and return its token count.

    Used to locate section boundaries within the full token sequence by tokenizing
    the text up to a known character position.

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
    conversation: list[dict[str, str]],
    tokenizer: _Tokenizer,
    model_info: ModelInfo,
    max_seq_length: int,
    masked: MaskType,
    ignore_index: int,
    override_tag: str | None,
) -> dict[str, list[int]]:
    """
    Tokenize a single training conversation and apply label masking.

    Locates PROMPT / THINK / RESPONSE boundaries by tokenizing text prefixes rather
    than using add_generation_prompt=True. This avoids a subtle issue with PREFIXED
    templates: the generation prompt already ends with <think>, so using it as a
    prefix boundary would leave the opening tag trainable while masking the closing
    tag — teaching the model to "open but never close" the reasoning block.

    Each section flagged in `masked` has its labels set to ignore_index so PyTorch's
    cross-entropy ignores those tokens during loss computation.

    Returns a dict with input_ids, labels, and attention_mask.
    """
    # response text is used to locate the response boundary
    response = conversation[-1]["content"]

    # apply the chat template with reasoning embedded, producing the full training sequence
    full_text = _apply_chat_template(
        conversation=conversation,
        tokenizer=tokenizer,
        add_generation_prompt=False,
        add_generation_reasoning=False,
        override_tag=override_tag,
    )

    # some templates strip think tags from assistant content (detected in model_info);
    # re-insert the think block so the training sequence is complete
    reasoning_raw = conversation[-1].get("reasoning", None)
    if reasoning_raw is not None and model_info.strips_think_tags:
        reasoning = reasoning_raw.strip()
        response_char = full_text.rfind(response)
        think_block = f"{model_info.open_tag}\n{reasoning}\n{model_info.close_tag}\n"
        full_text = full_text[:response_char] + think_block + full_text[response_char:]

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
    open_match = re.search(re.escape(model_info.open_tag), full_text)
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
    response_start_char = full_text.rfind(response)
    response_start = _tokenize_prefix(
        tokenizer=tokenizer,
        text=full_text[:response_start_char],
        max_seq_length=max_seq_length,
    )

    # mask each requested section independently
    if MaskType.PROMPT in masked:
        # mask everything from the start up to the think block (or response if no think)
        prompt_end = think_start if think_start is not None else response_start
        for i in range(prompt_end):
            labels[i] = ignore_index

    if MaskType.THINK in masked and think_start is not None:
        # mask the full reasoning block including its opening and closing tags
        for i in range(think_start, response_start):
            labels[i] = ignore_index

    if MaskType.RESPONSE in masked:
        # mask the response tokens
        for i in range(response_start, len(labels)):
            labels[i] = ignore_index

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def apply_mask(
    conversations: list[list[dict[str, str]]],
    tokenizer: _Tokenizer,
    masked: MaskType | None = MaskType.THINK,
    max_seq_length: int = 32768,
    ignore_index: int = _DEFAULT_IGNORE_INDEX,
    override_tag: str | None = None,
) -> Dataset:
    """
    Tokenize training conversations and mask selected sections from the loss.

    Each conversation must end with an assistant message containing at least a
    "content" key (the response). An optional "reasoning" key on the assistant message
    provides think block content — if absent when masking is active, an empty reasoning
    block is injected so training context matches inference time. Combine MaskType flags
    with | to mask multiple sections at once (see MaskType for details).

    Pass masked=None to train on all tokens.

    Returns a HuggingFace Dataset with input_ids, labels, and attention_mask columns.
    """
    model_info = get_model_info(tokenizer=tokenizer, override_tag=override_tag)

    # normalise None to an empty MaskType so downstream logic is consistent
    effective_masked = masked if masked is not None else MaskType(0)

    # when masking is active, inject an empty "reasoning" key into conversations that
    # lack one — ensures the think block appears for training/inference context alignment
    # on PREFIXED models that always emit think blocks at inference time
    if effective_masked:
        conversations = [
            conv
            if any("reasoning" in m for m in conv)
            else [*conv[:-1], {**conv[-1], "reasoning": ""}]
            for conv in conversations
        ]

    all_input_ids = []
    all_labels = []
    all_attention_mask = []

    for conv in conversations:
        result = _tokenize_record(
            conversation=conv,
            tokenizer=tokenizer,
            model_info=model_info,
            max_seq_length=max_seq_length,
            masked=effective_masked,
            ignore_index=ignore_index,
            override_tag=override_tag,
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
