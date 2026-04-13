"""Tests for thinkpack.stats — aggregate statistics over parsed responses."""

from thinkpack.parse import ParsedResponse
from thinkpack.stats import ResponseStats, stats


# ---------------------------------------------------------------------------
# helpers — build ParsedResponse objects directly without going through parse()
# ---------------------------------------------------------------------------


def _valid(answer: str = "the answer") -> ParsedResponse:
    """ParsedResponse with a completed, non-blank reasoning block and an answer."""
    return ParsedResponse(
        answer=answer,
        reasoning="\nsome reasoning\n",
        reasoning_tag="think",
        has_reasoning_block=True,
        has_valid_reasoning=True,
        has_truncated_reasoning=False,
    )


def _truncated() -> ParsedResponse:
    """ParsedResponse with a reasoning block that opened but never closed."""
    return ParsedResponse(
        answer="",
        reasoning="reasoning that never finished",
        reasoning_tag="think",
        has_reasoning_block=True,
        has_valid_reasoning=False,
        has_truncated_reasoning=True,
    )


def _empty_block(answer: str = "the answer") -> ParsedResponse:
    """ParsedResponse with a reasoning block that opened and closed but was blank."""
    return ParsedResponse(
        answer=answer,
        reasoning="\n",
        reasoning_tag="think",
        has_reasoning_block=True,
        has_valid_reasoning=False,
        has_truncated_reasoning=False,
    )


def _plain(answer: str = "just an answer") -> ParsedResponse:
    """ParsedResponse with no reasoning tags at all."""
    return ParsedResponse(
        answer=answer,
        reasoning="",
        reasoning_tag=None,
        has_reasoning_block=False,
        has_valid_reasoning=False,
        has_truncated_reasoning=False,
    )


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class TestStats:
    """Tests for the stats() function."""

    def test_mixed_batch_flat(self) -> None:
        """Counts are correct across a mixed flat list of parsed responses."""
        responses = [
            _valid(),         # reasoning block, valid reasoning, answer
            _truncated(),     # reasoning block, truncated, no answer
            _empty_block(),   # reasoning block, empty, answer
            _plain(),         # no block, plain answer
        ]
        result = stats(responses=responses)

        assert result.total == 4
        assert result.has_reasoning_block == 3
        assert result.has_truncated_reasoning == 1
        assert result.has_empty_reasoning == 1
        assert result.has_valid_reasoning == 1
        assert result.has_answer == 3

    def test_mixed_batch_nested(self) -> None:
        """Nested list[list[ParsedResponse]] is flattened to the same result."""
        nested = [
            [_valid(), _truncated()],
            [_empty_block(), _plain()],
        ]
        result = stats(responses=nested)

        assert result.total == 4
        assert result.has_reasoning_block == 3
        assert result.has_truncated_reasoning == 1
        assert result.has_empty_reasoning == 1
        assert result.has_valid_reasoning == 1
        assert result.has_answer == 3

    def test_all_valid_reasoning(self) -> None:
        """All valid responses are counted correctly."""
        responses = [_valid(), _valid(), _valid()]
        result = stats(responses=responses)

        assert result.total == 3
        assert result.has_reasoning_block == 3
        assert result.has_truncated_reasoning == 0
        assert result.has_empty_reasoning == 0
        assert result.has_valid_reasoning == 3
        assert result.has_answer == 3

    def test_all_truncated(self) -> None:
        """All truncated responses are counted as has_truncated_reasoning."""
        responses = [_truncated(), _truncated()]
        result = stats(responses=responses)

        assert result.total == 2
        assert result.has_reasoning_block == 2
        assert result.has_truncated_reasoning == 2
        assert result.has_empty_reasoning == 0
        assert result.has_valid_reasoning == 0
        assert result.has_answer == 0

    def test_empty_block_not_truncated_or_valid(self) -> None:
        """A completed but blank reasoning block counts only as has_empty_reasoning."""
        result = stats(responses=[_empty_block()])

        assert result.has_reasoning_block == 1
        assert result.has_truncated_reasoning == 0
        assert result.has_empty_reasoning == 1
        assert result.has_valid_reasoning == 0

    def test_reasoning_block_states_are_mutually_exclusive(self) -> None:
        """has_truncated + has_empty + has_valid always sums to has_reasoning_block."""
        responses = [_valid(), _truncated(), _empty_block(), _plain()]
        result = stats(responses=responses)

        assert (
            result.has_truncated_reasoning + result.has_empty_reasoning + result.has_valid_reasoning
            == result.has_reasoning_block
        )

    def test_no_answer_blank_string(self) -> None:
        """A blank answer string is not counted as has_answer."""
        result = stats(responses=[_valid(answer="   ")])

        assert result.has_answer == 0

    def test_empty_input(self) -> None:
        """Empty input returns all zero counts."""
        result = stats(responses=[])

        assert result == ResponseStats(
            total=0,
            has_reasoning_block=0,
            has_truncated_reasoning=0,
            has_empty_reasoning=0,
            has_valid_reasoning=0,
            has_answer=0,
        )

    def test_returns_response_stats_type(self) -> None:
        """stats() returns a ResponseStats instance."""
        result = stats(responses=[_plain()])
        assert isinstance(result, ResponseStats)
