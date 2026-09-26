"""Training-time loss masking for reasoning blocks."""

import logging
from enum import IntFlag

from datasets import Dataset

from thinkpack.chat import apply_chat_template as _apply_chat_template
from thinkpack.model import ModelInfo, _Tokenizer, _unwrap_tokenizer, get_model_info


_logger = logging.getLogger(__name__)


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

    PROMPT covers everything before the final reasoning block (system prompt, user
    instruction, any earlier turns). THINK covers the final reasoning block including its
    opening and closing tags. RESPONSE covers the model's answer and end-of-turn tokens.
    Masking RESPONSE is unusual (nothing useful remains to train on) but valid.
    """

    PROMPT = 1
    THINK = 2
    RESPONSE = 4


def _tokenize_with_offsets(
    tokenizer: _Tokenizer,
    text: str,
) -> tuple[list[int], list[tuple[int, int]]]:
    """
    Tokenize text once, keeping the character span of every token.

    The character spans (offsets) let section boundaries found in the text be mapped
    exactly onto token positions, without re-tokenizing partial strings.

    Returns (input_ids, offsets) where offsets[i] is the (start, end) character span
    of token i.
    """
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except NotImplementedError as error:
        # slow (pure python) tokenizers cannot report character offsets
        raise ValueError(
            "apply_mask() requires a fast tokenizer that can report character "
            "offsets — load the tokenizer with use_fast=True."
        ) from error

    input_ids = list(encoded["input_ids"])
    offsets = [(start, end) for start, end in encoded["offset_mapping"]]
    return input_ids, offsets


def _char_to_token(
    offsets: list[tuple[int, int]],
    char_pos: int,
) -> int:
    """
    Find the index of the first token that belongs at or after a character position.

    This is the first token whose span ends after char_pos, so a token that straddles
    the boundary is counted as part of the later section.

    Returns the token index, or len(offsets) if char_pos is beyond the last token.
    """
    for i, (_, end) in enumerate(offsets):
        if end > char_pos:
            return i
    return len(offsets)


def _locate_sections(
    full_text: str,
    response: str,
    model_info: ModelInfo,
) -> tuple[int, int]:
    """
    Find where the final reasoning block and the final response start in the text.

    Searches backwards from the end of the text, so reasoning tags that appear earlier
    (e.g. in a default system prompt, or an earlier assistant turn) are never matched.
    The response starts at the first non-whitespace character after the closing tag,
    and is checked against the expected response text so errors fail loudly.

    Returns (think_start_char, response_start_char) as character positions.
    """
    open_tag = model_info.open_tag
    close_tag = model_info.close_tag

    # the last closing tag in the text ends the final assistant turn's reasoning block
    close_char = full_text.rfind(close_tag)

    # its opening tag is the last one before that closing tag
    open_char = full_text.rfind(open_tag, 0, close_char) if close_char != -1 else -1
    if open_char == -1:
        raise ValueError(
            f"Could not find the {open_tag}...{close_tag} reasoning block of the final "
            "assistant message in the templated text. Check the detected tag, or pass "
            "override_tag."
        )

    # the response starts after the closing tag and any whitespace that follows it
    response_char = close_char + len(close_tag)
    while response_char < len(full_text) and full_text[response_char].isspace():
        response_char += 1

    # sanity check that the response really starts here — templates may trim
    # surrounding whitespace, so compare against the stripped response
    if not full_text.startswith(response.strip(), response_char):
        raise ValueError(
            "The text after the final reasoning block does not match the assistant "
            f"response, so the section boundaries cannot be trusted. Does the response "
            f"contain a literal {close_tag} tag?"
        )

    return open_char, response_char


def _tokenize_record(
    conversation: list[dict[str, str]],
    tokenizer: _Tokenizer,
    model_info: ModelInfo,
    max_seq_length: int,
    masked: MaskType,
    ignore_index: int,
    override_tag: str | None,
    add_history_reasoning: bool | None,
) -> tuple[dict[str, list[int]], bool]:
    """
    Tokenize a single training conversation and apply label masking.

    The conversation is rendered with the chat template (reasoning embedded), then
    tokenized once with character offsets. The PROMPT / THINK / RESPONSE boundaries are
    found as character positions in the rendered text and mapped onto token positions.

    Boundaries are found from the rendered text rather than by rendering the prompt with
    add_generation_prompt=True. This avoids a subtle issue with PREFIXED templates: the
    generation prompt already ends with <think>, so using it as a prefix boundary would
    leave the opening tag trainable while masking the closing tag — teaching the model
    to "open but never close" the reasoning block.

    Each section flagged in `masked` has its labels set to ignore_index so PyTorch's
    cross-entropy ignores those tokens during loss computation.

    Returns (record, truncated): record is a dict with input_ids, labels, and
    attention_mask, and truncated is true if the sequence was cut to max_seq_length.
    """
    if not conversation or conversation[-1].get("role") != "assistant":
        raise ValueError(
            "Each conversation must end with an assistant message (the training target)."
        )

    # response text of the final assistant message, used to check the boundaries
    response = conversation[-1].get("content", "")

    # apply the chat template with reasoning embedded, producing the full training
    # sequence — templates that strip think blocks are handled inside this call
    full_text = _apply_chat_template(
        conversation=conversation,
        tokenizer=tokenizer,
        add_generation_prompt=False,
        add_generation_reasoning=False,
        add_history_reasoning=add_history_reasoning,
        override_tag=override_tag,
    )

    # tokenize the full text once, keeping each token's character span
    input_ids, offsets = _tokenize_with_offsets(
        tokenizer=tokenizer,
        text=full_text,
    )

    # truncate to the maximum sequence length by dropping the end of the sequence
    truncated = len(input_ids) > max_seq_length
    input_ids = input_ids[:max_seq_length]
    offsets = offsets[:max_seq_length]

    # default: all tokens contribute to the loss
    labels = list(input_ids)

    if masked:
        # find the section boundaries as character positions in the rendered text
        think_start_char, response_start_char = _locate_sections(
            full_text=full_text,
            response=response,
            model_info=model_info,
        )

        # map the character positions onto token positions
        think_start = _char_to_token(
            offsets=offsets,
            char_pos=think_start_char,
        )
        response_start = _char_to_token(
            offsets=offsets,
            char_pos=response_start_char,
        )

        # mask each requested section independently
        if MaskType.PROMPT in masked:
            # mask everything from the start up to the final reasoning block
            for i in range(think_start):
                labels[i] = ignore_index

        if MaskType.THINK in masked:
            # mask the full reasoning block including its opening and closing tags
            for i in range(think_start, response_start):
                labels[i] = ignore_index

        if MaskType.RESPONSE in masked:
            # mask the response tokens
            for i in range(response_start, len(labels)):
                labels[i] = ignore_index

    record = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }
    return record, truncated


def apply_mask(
    conversations: list[list[dict[str, str]]],
    tokenizer: _Tokenizer,
    masked: MaskType | None = MaskType.THINK,
    max_seq_length: int = 32768,
    ignore_index: int = _DEFAULT_IGNORE_INDEX,
    override_tag: str | None = None,
    add_history_reasoning: bool | None = None,
) -> Dataset:
    """
    Tokenize training conversations and mask selected sections from the loss.

    Each conversation must end with an assistant message containing at least a
    "content" key (the response). An optional "reasoning" key on that final message
    provides think block content — if absent when masking is active, an empty reasoning
    block is injected so training context matches inference time. Combine MaskType flags
    with | to mask multiple sections at once (see MaskType for details).

    Pass masked=None to train on all tokens. add_history_reasoning controls reasoning
    on earlier assistant messages, exactly as in apply_chat_template(). A warning is
    logged if any conversation is truncated or left with no trainable tokens.

    Returns a HuggingFace Dataset with input_ids, labels, and attention_mask columns.
    """
    # multimodal processors wrap the text tokenizer — use the tokenizer directly
    tokenizer = _unwrap_tokenizer(tokenizer)
    model_info = get_model_info(tokenizer=tokenizer, override_tag=override_tag)

    # normalise None to an empty MaskType so downstream logic is consistent
    effective_masked = masked if masked is not None else MaskType(0)

    # when masking is active, inject an empty "reasoning" key into the final assistant
    # message if it lacks one — ensures the think block appears so it can be masked, and
    # matches inference time on PREFIXED models that always emit think blocks
    if effective_masked:
        conversations = [
            conv
            if not conv or "reasoning" in conv[-1]
            else [*conv[:-1], {**conv[-1], "reasoning": ""}]
            for conv in conversations
        ]

    all_input_ids = []
    all_labels = []
    all_attention_mask = []
    n_truncated = 0
    n_untrainable = 0

    for idx, conv in enumerate(conversations):
        try:
            record, truncated = _tokenize_record(
                conversation=conv,
                tokenizer=tokenizer,
                model_info=model_info,
                max_seq_length=max_seq_length,
                masked=effective_masked,
                ignore_index=ignore_index,
                override_tag=override_tag,
                add_history_reasoning=add_history_reasoning,
            )
        except ValueError as error:
            # add the position of the failing conversation to the error message
            raise ValueError(f"Conversation {idx}: {error}") from error

        # track problems to report once at the end
        n_truncated += int(truncated)
        if all(label == ignore_index for label in record["labels"]):
            n_untrainable += 1

        all_input_ids.append(record["input_ids"])
        all_labels.append(record["labels"])
        all_attention_mask.append(record["attention_mask"])

    if n_truncated:
        _logger.warning(
            "%d of %d conversations were longer than max_seq_length=%d tokens and "
            "were truncated.",
            n_truncated,
            len(conversations),
            max_seq_length,
        )
    if n_untrainable:
        _logger.warning(
            "%d of %d conversations have no trainable tokens after masking and "
            "truncation, so they contribute nothing to the loss.",
            n_untrainable,
            len(conversations),
        )

    return Dataset.from_dict(
        {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "attention_mask": all_attention_mask,
        }
    )
