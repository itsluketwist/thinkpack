"""Aggregate statistics over a collection of parsed model responses."""

from dataclasses import dataclass
from typing import cast

from thinkpack.parse import ParsedResponse


@dataclass
class ResponseStats:
    """Aggregate rates computed over a collection of ParsedResponse objects.

    total is a raw count; all other fields are rates in [0, 1]. For nested input
    (tasks × samples), rates are macro-averaged across tasks so each task contributes
    equally. Token averages and pass rates are None when not applicable.

    valid_reasoning_rate and invalid_reasoning_rate sum to 1. The three invalid sub-type
    rates (missing_reasoning_rate, empty_reasoning_rate, truncated_reasoning_rate) sum to
    invalid_reasoning_rate.
    """

    # total number of responses processed
    total: int

    # fraction of responses where a reasoning block was completed with non-blank content
    valid_reasoning_rate: float

    # fraction of responses where reasoning was absent, truncated, or empty
    invalid_reasoning_rate: float

    # fraction of responses with no reasoning block structure at all (model skipped thinking)
    missing_reasoning_rate: float

    # fraction of responses where a reasoning block opened but never closed
    truncated_reasoning_rate: float

    # fraction of responses where a reasoning block opened and closed but content was blank
    empty_reasoning_rate: float

    # fraction of responses that produced a non-blank answer
    answer_rate: float

    # mean reasoning token count per response (None if not calculated)
    avg_reasoning_tokens: float | None = None

    # mean answer token count per response (None if not calculated)
    avg_answer_tokens: float | None = None

    # fraction of responses with a correct answer; macro-averaged for nested input
    # (None if results not provided)
    pass_at_1: float | None = None

    # fraction correct among responses with valid reasoning; macro-averaged for nested
    # (None if results not provided or no responses had valid reasoning)
    rpass_at_1: float | None = None

    @property
    def vr(self) -> float:
        """Short for valid_reasoning_rate."""
        return self.valid_reasoning_rate

    @property
    def mr(self) -> float:
        """Short for missing_reasoning_rate."""
        return self.missing_reasoning_rate

    @property
    def tr(self) -> float:
        """Short for truncated_reasoning_rate."""
        return self.truncated_reasoning_rate

    @property
    def er(self) -> float:
        """Short for empty_reasoning_rate."""
        return self.empty_reasoning_rate


def _aggregate(task_stats: list[ResponseStats]) -> ResponseStats:
    """Macro-average a list of per-task ResponseStats into a single ResponseStats."""
    n = len(task_stats)
    if n == 0:
        return ResponseStats(
            total=0,
            valid_reasoning_rate=0.0,
            invalid_reasoning_rate=0.0,
            missing_reasoning_rate=0.0,
            truncated_reasoning_rate=0.0,
            empty_reasoning_rate=0.0,
            answer_rate=0.0,
        )

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals)

    def _mean_opt(vals: list[float | None]) -> float | None:
        # average only the non-None values; return None if all are None
        non_none = [v for v in vals if v is not None]
        return sum(non_none) / len(non_none) if non_none else None

    return ResponseStats(
        total=sum(s.total for s in task_stats),
        valid_reasoning_rate=_mean([s.valid_reasoning_rate for s in task_stats]),
        invalid_reasoning_rate=_mean([s.invalid_reasoning_rate for s in task_stats]),
        missing_reasoning_rate=_mean([s.missing_reasoning_rate for s in task_stats]),
        truncated_reasoning_rate=_mean(
            [s.truncated_reasoning_rate for s in task_stats]
        ),
        empty_reasoning_rate=_mean([s.empty_reasoning_rate for s in task_stats]),
        answer_rate=_mean([s.answer_rate for s in task_stats]),
        avg_reasoning_tokens=_mean_opt([s.avg_reasoning_tokens for s in task_stats]),
        avg_answer_tokens=_mean_opt([s.avg_answer_tokens for s in task_stats]),
        pass_at_1=_mean_opt([s.pass_at_1 for s in task_stats]),
        rpass_at_1=_mean_opt([s.rpass_at_1 for s in task_stats]),
    )


def compute_stats(
    responses: list[ParsedResponse] | list[list[ParsedResponse]],
    results: list[bool] | list[list[bool]] | None = None,
) -> ResponseStats:
    """Compute aggregate statistics over a flat or nested list of parsed responses.

    For nested list[list[ParsedResponse]] input (tasks × samples), runs the same flat
    computation on each task then macro-averages the rates — so each task contributes
    equally regardless of sample count. total is always the sum across all tasks.

    Pass results (matching the shape of responses) to compute pass_at_1 and rpass_at_1.
    Token averages are populated if the responses were parsed with calculate_tokens=True.

    Returns a ResponseStats with a total count, macro-averaged rates, token averages,
    and pass rates.
    """
    # nested input — compute per-task stats then aggregate
    if responses and isinstance(responses[0], list):
        nested_responses = cast(list[list[ParsedResponse]], responses)
        if results is not None:
            nested_results = cast(list[list[bool]], results)
            if len(nested_results) != len(nested_responses):
                raise ValueError(
                    f"results has {len(nested_results)} tasks but responses has "
                    f"{len(nested_responses)}."
                )
            task_stats = [
                compute_stats(responses=tr, results=tres)
                for tr, tres in zip(nested_responses, nested_results)
            ]
        else:
            task_stats = [compute_stats(responses=tr) for tr in nested_responses]
        return _aggregate(task_stats)

    # flat input — compute rates directly
    flat = cast(list[ParsedResponse], responses)
    total = len(flat)

    def _rate(vals: list[bool | int]) -> float:
        return sum(vals) / total if total else 0.0

    avg_reasoning_tokens: float | None = None
    avg_answer_tokens: float | None = None
    if total > 0 and any(r.reasoning_token_count is not None for r in flat):
        avg_reasoning_tokens = sum(r.reasoning_token_count or 0 for r in flat) / total
        avg_answer_tokens = sum(r.answer_token_count or 0 for r in flat) / total

    pass_at_1: float | None = None
    rpass_at_1: float | None = None
    if results is not None and total > 0:
        flat_results = cast(list[bool], results)
        if len(flat_results) != total:
            raise ValueError(
                f"results length ({len(flat_results)}) does not match "
                f"responses length ({total})."
            )
        pass_at_1 = sum(flat_results) / total
        valid_count = sum(r.has_valid_reasoning for r in flat)
        if valid_count > 0:
            rpass_at_1 = (
                sum(c for c, r in zip(flat_results, flat) if r.has_valid_reasoning)
                / valid_count
            )

    return ResponseStats(
        total=total,
        valid_reasoning_rate=_rate([r.has_valid_reasoning for r in flat]),
        invalid_reasoning_rate=_rate([r.has_invalid_reasoning for r in flat]),
        missing_reasoning_rate=_rate([r.has_missing_reasoning for r in flat]),
        truncated_reasoning_rate=_rate([r.has_truncated_reasoning for r in flat]),
        empty_reasoning_rate=_rate([r.has_empty_reasoning for r in flat]),
        answer_rate=_rate([r.extracted_answer for r in flat]),
        avg_reasoning_tokens=avg_reasoning_tokens,
        avg_answer_tokens=avg_answer_tokens,
        pass_at_1=pass_at_1,
        rpass_at_1=rpass_at_1,
    )
