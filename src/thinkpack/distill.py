"""Helpers for building reasoning traces for training data, using a teacher model."""

import re
from typing import overload


# default preamble — asks the teacher model to explain how to reach a known answer
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
    Build prompts asking a teacher model to write the reasoning for each record.

    Each prompt gives the question and its correct answer, and asks for the reasoning
    steps inside <distill_tag> tags. Set the closing tag (e.g. "</reasoning_steps>") as
    a stop sequence when generating, so the model stops after the reasoning.
    reasoning_example optionally replaces the default formatting example.

    Returns a list of prompt strings, one per record.
    """
    prompts = []
    for record in records:
        instruction = record[instruction_key]
        response = record[response_key]

        # use the given example, or a default one showing the expected format
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
    Extract the reasoning from a teacher model's response to a build_prompts() prompt.

    Accepts a single string or a list. Takes the text after the last opening distill_tag,
    up to the closing tag, or to the end if there is no closing tag (e.g. when it was
    used as a stop sequence).

    Returns the reasoning string, or None if none was found or it is blank (a list of
    these for list input).
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

    # use the last opening tag, in case the model repeats the tag earlier in its output
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
    Add the reasoning from each teacher response to the matching record.

    Extracts the reasoning from each response with extract_distilled_reasoning(), and
    stores it under reasoning_field. Records with no reasoning found are left without
    the field. responses must be the same length as records. The input records are
    not changed.

    Returns a new list of records, with reasoning_field added where available.
    """
    # extract the reasoning from every response
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
    Convert records into conversations for apply_chat_template() and apply_mask().

    Each record becomes a user message (the instruction) and an assistant message (the
    response). If the record has reasoning_key, its value is added to the assistant
    message as "reasoning", so it is embedded as the think block.

    Returns a list of conversations, one per record.
    """
    conversations = []
    for record in records:
        assistant: dict[str, str] = {
            "role": "assistant",
            "content": record[response_key],
        }
        # only add reasoning if the record has it
        if reasoning_key in record:
            assistant["reasoning"] = record[reasoning_key]

        conversations.append(
            [
                {"role": "user", "content": record[instruction_key]},
                assistant,
            ]
        )

    return conversations
