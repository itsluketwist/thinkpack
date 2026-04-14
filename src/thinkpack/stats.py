"""Aggregate statistics over a collection of parsed model responses."""

from dataclasses import dataclass
from typing import cast

from thinkpack.parse import ParsedResponse


@dataclass
class ResponseStats:
    """
    Aggregate counts computed over a collection of ParsedResponse objects.

    All integer fields are raw counts; divide by total to get rates.
    """

    # total number of responses processed
    total: int

    # responses with any reasoning block structure (even empty or truncated)
    has_reasoning_block: int

    # responses where a reasoning block opened but never closed
    has_truncated_reasoning: int

    # responses where a reasoning block opened and closed but content was blank
    has_empty_reasoning: int

    # responses where a reasoning block was completed with non-blank content
    has_valid_reasoning: int

    # responses that produced a non-blank answer
    has_answer: int


def stats(
    responses: list[ParsedResponse] | list[list[ParsedResponse]],
) -> ResponseStats:
    """Compute aggregate statistics over a collection of parsed responses.

    Accepts either a flat list[ParsedResponse] or the nested list[list[ParsedResponse]]
    shape returned by parse_all() / parse_output() — both are flattened before counting.

    Returns a ResponseStats with counts for each statistic.
    """
    # flatten nested list[list[ParsedResponse]] into a single flat list
    flat: list[ParsedResponse]
    if responses and isinstance(responses[0], list):
        nested = cast(list[list[ParsedResponse]], responses)
        flat = [r for group in nested for r in group]
    else:
        flat = cast(list[ParsedResponse], responses)

    return ResponseStats(
        total=len(flat),
        # any reasoning block structure detected, even if blank or truncated
        has_reasoning_block=sum(r.has_reasoning_block for r in flat),
        # block started but the model never produced a closing tag
        has_truncated_reasoning=sum(r.has_truncated_reasoning for r in flat),
        # block opened and closed, but content was whitespace-only
        has_empty_reasoning=sum(
            r.has_reasoning_block
            and not r.has_valid_reasoning
            and not r.has_truncated_reasoning
            for r in flat
        ),
        # completed reasoning block with non-blank content
        has_valid_reasoning=sum(r.has_valid_reasoning for r in flat),
        # a non-blank answer was produced after the reasoning block
        has_answer=sum(bool(r.answer.strip()) for r in flat),
    )
