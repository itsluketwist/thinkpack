"""Parsing of model responses into reasoning and answer components."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from thinkpack._tags import CLOSE_TAG as _REASONING_CLOSE_TAG
from thinkpack._tags import OPEN_TAG as _REASONING_OPEN_TAG


class _Completion(Protocol):
    """Minimal protocol for a vLLM-compatible generation completion."""

    text: str


class _GenerationOutput(Protocol):
    """Minimal protocol for a vLLM-compatible generation output (e.g. RequestOutput)."""

    # sequence (not list) so structurally-typed mocks returning list[T] satisfy this
    outputs: Sequence[_Completion]


@dataclass
class ParsedResponse:
    """
    A model response split into reasoning and answer components.

    answer and reasoning contain the extracted text; the boolean flags
    describe the structure of the reasoning block at a glance.
    """

    # the model's final answer — text after the closing reasoning tag, or the
    # full response if no reasoning block was found
    answer: str

    # content inside the reasoning block; empty string if no block was present
    reasoning: str

    # lowercase tag name used, e.g. "think", "reasoning" (None if no tag found)
    reasoning_tag: str | None

    # true if any reasoning block structure is present, even if blank or truncated
    has_reasoning_block: bool

    # true if the reasoning block was completed and non-blank
    has_valid_reasoning: bool

    # true if an opening tag was found but the model never produced a closing tag
    has_truncated_reasoning: bool


def parse(
    response: str,
    prefixed: bool = False,
    tag: str | None = None,
) -> ParsedResponse:
    """Parse a single model response into reasoning and answer components.

    Handles four formats:
    - standard:           <think>content</think>answer
    - prefixed:           content</think>answer  (opening tag injected by chat template)
    - truncated standard: <think>content...      (open tag, no close tag)
    - truncated prefixed: content...             (no tags; only detectable with prefixed=True)

    Returns a ParsedResponse with the split content and status flags.
    """
    # compile tag-specific patterns if the caller has specified an exact tag name,
    # otherwise fall back to the shared patterns that match all known variants
    if tag is not None:
        escaped = re.escape(tag)
        open_re = re.compile(rf"<({escaped})>", re.IGNORECASE)
        close_re = re.compile(rf"</({escaped})>", re.IGNORECASE)
    else:
        open_re = _REASONING_OPEN_TAG
        close_re = _REASONING_CLOSE_TAG

    close_match = close_re.search(response)

    if close_match:
        tag_name = close_match.group(1).lower()
        # extract everything before the closing tag, then strip any open tag
        # (prefixed template responses have no opening tag in decoded output)
        before_close = response[: close_match.start()]
        thinking = open_re.sub("", before_close, count=1)
        answer = response[close_match.end() :].strip()
        has_valid_reasoning = bool(thinking.strip())
        return ParsedResponse(
            answer=answer,
            reasoning=thinking,
            reasoning_tag=tag_name,
            has_reasoning_block=True,
            has_valid_reasoning=has_valid_reasoning,
            has_truncated_reasoning=False,
        )

    # no closing tag — check for a truncated reasoning block (open tag present)
    open_match = open_re.search(response)
    if open_match:
        # model started reasoning but output was cut off before the close tag
        return ParsedResponse(
            answer="",
            reasoning=response[open_match.end() :],
            reasoning_tag=open_match.group(1).lower(),
            has_reasoning_block=True,
            has_valid_reasoning=False,
            has_truncated_reasoning=True,
        )

    if prefixed:
        # for PREFIXED template models the opening tag is injected by the chat
        # template and never appears in decoded output — a missing close tag means
        # the reasoning was truncated, not that there was no think block at all
        return ParsedResponse(
            answer="",
            reasoning=response,
            reasoning_tag=None,
            has_reasoning_block=True,
            has_valid_reasoning=False,
            has_truncated_reasoning=True,
        )

    # no reasoning tags at all — plain response with no think block
    return ParsedResponse(
        answer=response,
        reasoning="",
        reasoning_tag=None,
        has_reasoning_block=False,
        has_valid_reasoning=False,
        has_truncated_reasoning=False,
    )


def parse_all(
    responses: list[list[str]],
    prefixed: bool = False,
    tag: str | None = None,
) -> list[list[ParsedResponse]]:
    """Parse a batch of model responses into ParsedResponse objects.

    Accepts a nested list of shape [task][sample] and returns the same shape.
    Pass tag to restrict matching to a single tag name (see parse() for details).

    Returns a nested list of ParsedResponse objects matching the input shape.
    """
    return [
        [parse(response=r, prefixed=prefixed, tag=tag) for r in sample_responses]
        for sample_responses in responses
    ]


def parse_output(
    output: _GenerationOutput | list[_GenerationOutput],
    prefixed: bool = False,
    tag: str | None = None,
) -> list[ParsedResponse] | list[list[ParsedResponse]]:
    """Parse one or more generation output objects into ParsedResponse objects.

    Accepts either:
    - a single output object with an .outputs attribute (e.g. a vLLM RequestOutput),
      returning a flat list of ParsedResponse — one per sample/completion, or
    - a list of such objects, returning a nested list of shape [task][sample].

    Each completion object is expected to have a .text (str) attribute —
    compatible with vLLM's RequestOutput and similar interfaces.

    Pass tag to restrict matching to a single tag name (see parse() for details).

    Returns a list of ParsedResponse (single output) or list[list[ParsedResponse]] (list).
    """
    if isinstance(output, list):
        # list of output objects — recurse to produce a nested [task][sample] structure
        # cast needed: isinstance(x, list) narrows to list[object], losing the element type
        return cast(
            list[list[ParsedResponse]],
            [
                parse_output(output=o, prefixed=prefixed, tag=tag)
                for o in cast(list[_GenerationOutput], output)
            ],
        )
    # single output object — parse each completion in its .outputs attribute
    completions = output.outputs
    return [
        parse(
            response=completion.text,
            prefixed=prefixed,
            tag=tag,
        )
        for completion in completions
    ]
