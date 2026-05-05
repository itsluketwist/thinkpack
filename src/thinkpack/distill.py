"""Distillation utilities for constructing reasoning prompts and extracting reasoning traces."""

import re
from typing import overload


# default preamble used when none is provided — presents the task as a
# backwards explanation: given the answer, produce the reasoning that leads to it
_DEFAULT_PREAMBLE = (
    "I need assistance constructing a reasoning dataset.\n"
    "Given the following question and its correct answer, "
    "give a concise summary of the reasoning steps that "
    "explains how to arrive at the answer."
)


def build_prompts(
    records: list[dict[str, str]],
    instruction_key: str = "instruction",
    response_key: str = "response",
    distill_tag: str = "reasoning_steps",
    preamble: str = _DEFAULT_PREAMBLE,
    reasoning_example: str | None = None,
) -> list[str]:
    """
    Build construct-mode distillation prompts from a list of records.

    Each prompt presents the question and correct answer, asking the model to produce
    a reasoning trace inside the specified tag. The closing tag should be configured as
    a stop token so the model stops after reasoning.

    Returns a list of prompt strings, one per record.
    """
    prompts = []
    for record in records:
        instruction = record[instruction_key]
        response = record[response_key]

        # build the example block only if one was provided
        if reasoning_example is not None:
            example_block = f"\n\nHere is a complete example:\n{reasoning_example}"
        else:
            example_block = (
                "\n\nStart your reasoning with 'Okay, ', for example:\n"
                f"<{distill_tag}>\nOkay, [your reasoning steps here]\n</{distill_tag}>"
            )

        prompt = (
            f"{preamble}\n\n"
            f"Question: {instruction}\n\n"
            f"Answer: {response}\n\n"
            f"In your response, give the reasoning steps inside <{distill_tag}> tags."
            f"{example_block}"
        )
        prompts.append(prompt)

    return prompts


@overload
def extract_distilled_reasoning(
    text: str,
    distill_tag: str = ...,
) -> str | None: ...


@overload
def extract_distilled_reasoning(
    text: list[str],
    distill_tag: str = ...,
) -> list[str | None]: ...


def extract_distilled_reasoning(
    text: str | list[str],
    distill_tag: str = "reasoning_steps",
) -> str | None | list[str | None]:
    """
    Extract a distilled reasoning trace from a response to a build_prompts prompt.

    Accepts a single string or a list; the return type matches the input. Finds the last
    occurrence of the opening distill_tag and takes everything after it up to the closing
    tag if present, or everything remaining if it is absent (stop-token scenario).

    Returns the extracted reasoning string (or None if not found / blank), or a list of
    the same for list input.
    """
    if isinstance(text, list):
        return [
            extract_distilled_reasoning(
                text=t,
                distill_tag=distill_tag,
            )
            for t in text
        ]

    open_tag_re = re.compile(rf"<{re.escape(distill_tag)}>", re.IGNORECASE)
    close_tag_re = re.compile(rf"</{re.escape(distill_tag)}>", re.IGNORECASE)

    # use the last open tag — model may output preamble text before the final attempt
    open_matches = list(open_tag_re.finditer(text))
    if not open_matches:
        return None

    after_open = text[open_matches[-1].end() :]

    # take up to the closing tag if present, otherwise take everything remaining
    close_match = close_tag_re.search(after_open)
    content = (
        after_open[: close_match.start()].strip() if close_match else after_open.strip()
    )

    return content if content else None


def update_records(
    records: list[dict[str, str]],
    responses: list[str],
    reasoning_field: str = "reasoning",
    distill_tag: str = "reasoning_steps",
) -> list[dict[str, str]]:
    """
    Add extracted reasoning traces into a list of records.

    Calls extract_distilled_reasoning on each response and writes the result into the
    corresponding record under reasoning_field. Only adds the field where extraction
    succeeded; records where extraction returns None are returned unchanged. Original
    records are not mutated.

    Returns a new list of record dicts with the reasoning_field added where available.
    """
    # extract from all responses in one call (list path)
    extractions: list[str | None] = extract_distilled_reasoning(
        text=responses,
        distill_tag=distill_tag,
    )

    updated = []
    for record, reasoning in zip(records, extractions, strict=True):
        new_record = {**record}
        if reasoning is not None:
            new_record[reasoning_field] = reasoning
        updated.append(new_record)

    return updated


def to_conversations(
    records: list[dict[str, str]],
    instruction_key: str = "instruction",
    response_key: str = "response",
    reasoning_key: str = "reasoning",
) -> list[list[dict[str, str]]]:
    """
    Convert records into conversation format compatible with apply_chat_template and apply_mask.

    Each record becomes a two-turn conversation: a user message with the instruction and
    an assistant message with the response. If the record contains an entry at reasoning_key,
    it is included as a "reasoning" key on the assistant message so the think block is
    embedded when the conversation is passed to apply_chat_template or apply_mask.

    Returns a list of conversations, one per record.
    """
    conversations = []
    for record in records:
        assistant: dict[str, str] = {
            "role": "assistant",
            "content": record[response_key],
        }
        # only attach reasoning if the record has the key — absence means no think block
        if reasoning_key in record:
            assistant["reasoning"] = record[reasoning_key]

        conversations.append(
            [
                {"role": "user", "content": record[instruction_key]},
                assistant,
            ]
        )

    return conversations
