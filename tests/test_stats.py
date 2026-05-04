"""Tests for thinkpack.stats — aggregate statistics over parsed responses."""

import pytest

from thinkpack.parse import ParsedResponse
from thinkpack.stats import ResponseStats, compute_stats


# ---------------------------------------------------------------------------
# helpers — build ParsedResponse objects directly without going through parse()
# ---------------------------------------------------------------------------


def _valid(
    answer: str = "the answer", reasoning_tokens: int | None = None
) -> ParsedResponse:
    """ParsedResponse with a completed, non-blank reasoning block and an answer."""
    return ParsedResponse(
        answer=answer,
        reasoning="some reasoning",
        reasoning_tag="think",
        has_valid_reasoning=True,
        has_truncated_reasoning=False,
        has_empty_reasoning=False,
        has_missing_reasoning=False,
        reasoning_token_count=reasoning_tokens,
        answer_token_count=len(answer.split())
        if reasoning_tokens is not None
        else None,
    )


def _truncated() -> ParsedResponse:
    """ParsedResponse with a reasoning block that opened but never closed."""
    return ParsedResponse(
        answer="",
        reasoning="reasoning that never finished",
        reasoning_tag="think",
        has_valid_reasoning=False,
        has_truncated_reasoning=True,
        has_empty_reasoning=False,
        has_missing_reasoning=False,
    )


def _empty_block(answer: str = "the answer") -> ParsedResponse:
    """ParsedResponse with a reasoning block that opened and closed but was blank."""
    return ParsedResponse(
        answer=answer,
        reasoning="",
        reasoning_tag="think",
        has_valid_reasoning=False,
        has_truncated_reasoning=False,
        has_empty_reasoning=True,
        has_missing_reasoning=False,
    )


def _plain(answer: str = "just an answer") -> ParsedResponse:
    """ParsedResponse with no reasoning tags at all."""
    return ParsedResponse(
        answer=answer,
        reasoning="",
        reasoning_tag=None,
        has_valid_reasoning=False,
        has_truncated_reasoning=False,
        has_empty_reasoning=False,
        has_missing_reasoning=True,
    )


# ---------------------------------------------------------------------------
# flat input tests
# ---------------------------------------------------------------------------


class TestStats:
    """Tests for compute_stats() with flat list[ParsedResponse] input."""

    def test_mixed_batch(self) -> None:
        """Rates are correct across a mixed flat list of parsed responses."""
        responses = [_valid(), _truncated(), _empty_block(), _plain()]
        result = compute_stats(responses=responses)

        assert result.total == 4
        assert result.valid_reasoning_rate == pytest.approx(1 / 4)
        assert result.invalid_reasoning_rate == pytest.approx(3 / 4)
        assert result.missing_reasoning_rate == pytest.approx(1 / 4)
        assert result.truncated_reasoning_rate == pytest.approx(1 / 4)
        assert result.empty_reasoning_rate == pytest.approx(1 / 4)
        assert result.answer_rate == pytest.approx(3 / 4)

    def test_all_valid_reasoning(self) -> None:
        """All valid responses produce a rate of 1.0."""
        result = compute_stats(responses=[_valid(), _valid(), _valid()])

        assert result.total == 3
        assert result.valid_reasoning_rate == pytest.approx(1.0)
        assert result.invalid_reasoning_rate == pytest.approx(0.0)
        assert result.missing_reasoning_rate == pytest.approx(0.0)
        assert result.answer_rate == pytest.approx(1.0)

    def test_all_truncated(self) -> None:
        """All truncated responses produce truncated_reasoning_rate of 1.0."""
        result = compute_stats(responses=[_truncated(), _truncated()])

        assert result.total == 2
        assert result.valid_reasoning_rate == pytest.approx(0.0)
        assert result.invalid_reasoning_rate == pytest.approx(1.0)
        assert result.missing_reasoning_rate == pytest.approx(0.0)
        assert result.truncated_reasoning_rate == pytest.approx(1.0)
        assert result.answer_rate == pytest.approx(0.0)

    def test_rates_sum_correctly(self) -> None:
        """vr + tr + er + mr always sums to 1."""
        result = compute_stats(
            responses=[_valid(), _truncated(), _empty_block(), _plain()]
        )

        assert (
            result.valid_reasoning_rate
            + result.truncated_reasoning_rate
            + result.empty_reasoning_rate
            + result.missing_reasoning_rate
            == pytest.approx(1.0)
        )
        # invalid_reasoning_rate is the complement of valid_reasoning_rate
        assert (
            result.valid_reasoning_rate + result.invalid_reasoning_rate
            == pytest.approx(1.0)
        )

    def test_no_answer_blank_string(self) -> None:
        """A blank answer string does not count towards has_answer."""
        assert compute_stats(
            responses=[_valid(answer="   ")]
        ).answer_rate == pytest.approx(0.0)

    def test_empty_input(self) -> None:
        """Empty input returns zero total, zero rates, and None for optional fields."""
        result = compute_stats(responses=[])

        assert result == ResponseStats(
            total=0,
            valid_reasoning_rate=0.0,
            invalid_reasoning_rate=0.0,
            missing_reasoning_rate=0.0,
            truncated_reasoning_rate=0.0,
            empty_reasoning_rate=0.0,
            answer_rate=0.0,
        )

    def test_returns_response_stats_type(self) -> None:
        """compute_stats() returns a ResponseStats instance for flat input."""
        assert isinstance(compute_stats(responses=[_plain()]), ResponseStats)

    def test_optional_fields_none_by_default(self) -> None:
        """Pass rates and token averages are None when not provided."""
        result = compute_stats(responses=[_valid()])

        assert result.pass_at_1 is None
        assert result.rpass_at_1 is None
        assert result.avg_reasoning_tokens is None
        assert result.avg_answer_tokens is None


# ---------------------------------------------------------------------------
# nested input tests
# ---------------------------------------------------------------------------


class TestStatsNested:
    """Tests for compute_stats() with nested list[list[ParsedResponse]] input."""

    def test_returns_single_stats(self) -> None:
        """Nested input returns a single macro-averaged ResponseStats."""
        result = compute_stats(responses=[[_valid(), _truncated()], [_plain()]])

        assert isinstance(result, ResponseStats)

    def test_total_is_sum_of_all_samples(self) -> None:
        """total counts all responses across all tasks."""
        result = compute_stats(
            responses=[[_valid(), _truncated(), _plain()], [_valid()]]
        )

        assert result.total == 4

    def test_rates_are_macro_averaged(self) -> None:
        """Rates are averaged per task, so each task contributes equally."""
        # task 1: 3 samples, 1 valid → valid_reasoning_rate = 1/3
        # task 2: 1 sample,  1 valid → valid_reasoning_rate = 1/1
        # macro-avg: (1/3 + 1.0) / 2 = 2/3  (micro would be 2/4 = 0.5)
        result = compute_stats(responses=[[_valid(), _plain(), _plain()], [_valid()]])

        assert result.valid_reasoning_rate == pytest.approx((1 / 3 + 1.0) / 2)

    def test_pass_at_1_macro_averaged(self) -> None:
        """pass_at_1 is macro-averaged across tasks."""
        # task 1: 1/2 correct → 0.5; task 2: 1/1 correct → 1.0; avg = 0.75
        result = compute_stats(
            responses=[[_valid(), _valid()], [_valid()]],
            results=[[True, False], [True]],
        )

        assert result.pass_at_1 == pytest.approx(0.75)

    def test_rpass_at_1_excludes_tasks_without_valid_reasoning(self) -> None:
        """rpass_at_1 macro-average skips tasks where no sample has valid reasoning."""
        # task 1: 1 valid-reasoning sample, correct → rpass = 1.0
        # task 2: plain only, no valid reasoning → excluded from rpass average
        result = compute_stats(
            responses=[[_valid()], [_plain()]],
            results=[[True], [True]],
        )

        assert result.rpass_at_1 == pytest.approx(1.0)
        assert result.pass_at_1 == pytest.approx(1.0)

    def test_task_count_mismatch_raises(self) -> None:
        """Mismatched task counts between responses and results raise ValueError."""
        with pytest.raises(ValueError, match="tasks"):
            compute_stats(responses=[[_valid()], [_valid()]], results=[[True]])


# ---------------------------------------------------------------------------
# pass rate tests (flat input)
# ---------------------------------------------------------------------------


class TestPassRates:
    """Tests for pass_at_1 and rpass_at_1 with flat input."""

    def test_pass_at_1_all_correct(self) -> None:
        """pass_at_1 is 1.0 when all results are True."""
        result = compute_stats(responses=[_valid(), _valid()], results=[True, True])

        assert result.pass_at_1 == pytest.approx(1.0)
        assert result.rpass_at_1 == pytest.approx(1.0)

    def test_pass_at_1_none_correct(self) -> None:
        """pass_at_1 is 0.0 when no results are True."""
        result = compute_stats(responses=[_valid(), _valid()], results=[False, False])

        assert result.pass_at_1 == pytest.approx(0.0)
        assert result.rpass_at_1 == pytest.approx(0.0)

    def test_pass_at_1_mixed(self) -> None:
        """pass_at_1 is the fraction of correct results."""
        result = compute_stats(
            responses=[_valid(), _valid(), _valid(), _valid()],
            results=[True, False, True, False],
        )

        assert result.pass_at_1 == pytest.approx(0.5)

    def test_rpass_at_1_only_counts_valid_reasoning(self) -> None:
        """rpass_at_1 is computed only over responses with valid reasoning."""
        # 2 valid-reasoning responses (1 correct), 1 plain (correct)
        result = compute_stats(
            responses=[_valid(), _valid(), _plain()], results=[True, False, True]
        )

        assert result.rpass_at_1 == pytest.approx(0.5)
        assert result.pass_at_1 == pytest.approx(2 / 3)

    def test_rpass_at_1_none_when_no_valid_reasoning(self) -> None:
        """rpass_at_1 is None when no responses have valid reasoning."""
        result = compute_stats(
            responses=[_plain(), _truncated()], results=[True, False]
        )

        assert result.rpass_at_1 is None
        assert result.pass_at_1 == pytest.approx(0.5)

    def test_results_length_mismatch_raises(self) -> None:
        """Mismatched results and responses lengths raise ValueError."""
        with pytest.raises(ValueError, match="length"):
            compute_stats(responses=[_valid(), _valid()], results=[True])


# ---------------------------------------------------------------------------
# token aggregation tests
# ---------------------------------------------------------------------------


class TestTokenAggregation:
    """Tests for average token count fields."""

    def test_token_averages_populated(self) -> None:
        """Token averages are computed when responses have token counts."""
        # _valid sets answer_token_count = len("the answer".split()) = 2
        result = compute_stats(
            responses=[_valid(reasoning_tokens=10), _valid(reasoning_tokens=20)]
        )

        assert result.avg_reasoning_tokens == pytest.approx(15.0)
        assert result.avg_answer_tokens == pytest.approx(2.0)

    def test_token_averages_none_without_counts(self) -> None:
        """Token averages are None when responses have no token counts."""
        result = compute_stats(responses=[_valid(), _plain()])

        assert result.avg_reasoning_tokens is None
        assert result.avg_answer_tokens is None

    def test_token_averages_in_nested_input(self) -> None:
        """Token averages are macro-averaged across tasks for nested input."""
        result = compute_stats(
            responses=[[_valid(reasoning_tokens=10)], [_valid(reasoning_tokens=30)]],
        )

        # macro-avg: (10.0 + 30.0) / 2 = 20.0
        assert result.avg_reasoning_tokens == pytest.approx(20.0)
