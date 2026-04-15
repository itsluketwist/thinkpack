"""Distillation utilities for constructing reasoning prompts and extracting reasoning traces."""

import re
from typing import overload

from thinkpack.parse import parse


# default preamble used when none is provided — presents the task as a
# backwards explanation: given the answer, produce the reasoning that leads to it
_DEFAULT_PREAMBLE = (
    "Given the following question and its correct answer, "
    "produce a step-by-step reasoning trace that "
    "explains how to arrive at the answer."
)


def build_prompts(
    records: list[dict[str, str]],
    instruction_key: str = "instruction",
    response_key: str = "response",
    tag: str = "reasoning_trace",
    preamble: str = _DEFAULT_PREAMBLE,
    example: str | None = None,
) -> list[str]:
    """
    Build construct-mode distillation prompts from a list of records.

    Each prompt presents the question and correct answer, asking the model
    to produce a reasoning trace inside the specified tag. The closing tag
    should be configured as a stop token so the model stops after reasoning.

    Returns a list of prompt strings, one per record.
    """
    prompts = []
    for record in records:
        instruction = record[instruction_key]
        response = record[response_key]

        # build the example block only if one was provided
        if example is not None:
            example_block = f"Here is an example:\n<{tag}>\n{example}\n</{tag}>\n\n"
        else:
            example_block = ""

        prompt = (
            f"{preamble}\n\n"
            f"Question: {instruction}\n\n"
            f"Answer: {response}\n\n"
            f"{example_block}"
            f"Provide your reasoning inside <{tag}> tags."
        )
        prompts.append(prompt)

    return prompts


@overload
def extract_reasoning(
    text: str,
    tag: str | None = ...,
    prefixed: bool = ...,
    strip_think: bool = ...,
) -> str | None: ...


@overload
def extract_reasoning(
    text: list[str],
    tag: str | None = ...,
    prefixed: bool = ...,
    strip_think: bool = ...,
) -> list[str | None]: ...


def extract_reasoning(
    text: str | list[str],
    tag: str | None = None,
    prefixed: bool = False,
    strip_think: bool = True,
) -> str | None | list[str | None]:
    """
    Extract a reasoning trace from a model response or a list of responses.

    Accepts a single string or a list; the return type matches the input.
    Delegates to parse() for standard think/reasoning/thought tags, including
    the truncated case where the closing tag is a stop token.

    For custom tags (e.g. "reasoning_trace"), finds the opening tag and takes
    everything after it — the closing tag is assumed to be a stop token and
    absent from the output.

    Returns the extracted reasoning string (or None if not found / blank) for
    a single input, or a list of the same for a list input.
    """
    if isinstance(text, list):
        return [
            extract_reasoning(
                text=t,
                tag=tag,
                prefixed=prefixed,
                strip_think=strip_think,
            )
            for t in text
        ]

    if tag is None:
        # delegate to parse() which handles all standard reasoning tags and
        # the truncated case (open tag, no close tag = stop token scenario)
        parsed = parse(response=text, prefixed=prefixed)
        content = parsed.reasoning.strip()
        return content if content else None

    # custom tag mode: the closing tag is a stop token and never present,
    # so find the opening tag and take everything after it
    if strip_think:
        # strip any standard think block first (its </think> is NOT a stop
        # token in this mode, so it will appear in the output)
        parsed = parse(response=text, prefixed=prefixed)
        search_text = parsed.answer
    else:
        search_text = text

    open_tag_re = re.compile(rf"<{re.escape(tag)}>", re.IGNORECASE)
    match = open_tag_re.search(search_text)
    if match is None:
        return None

    content = search_text[match.end() :].strip()
    return content if content else None


def update_records(
    records: list[dict[str, str]],
    responses: list[str],
    field: str = "reasoning_constructed",
    tag: str | None = None,
    prefixed: bool = False,
    strip_think: bool = True,
) -> list[dict[str, str]]:
    """
    Add extracted reasoning traces into a list of records.

    Calls extract_reasoning on each response and writes the result into the
    corresponding record under field. Only adds the field where extraction
    succeeded; records where extraction returns None are returned unchanged.
    Original records are not mutated.

    Returns a new list of record dicts with the reasoning field added where available.
    """
    # extract from all responses in one call (list path)
    extractions: list[str | None] = extract_reasoning(
        text=responses,
        tag=tag,
        prefixed=prefixed,
        strip_think=strip_think,
    )

    updated = []
    for record, reasoning in zip(records, extractions, strict=True):
        new_record = {**record}
        if reasoning is not None:
            new_record[field] = reasoning
        updated.append(new_record)

    return updated
