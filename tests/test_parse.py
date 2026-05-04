"""Tests for thinkpack.parse — response parsing into reasoning and answer.

Non-slow tests use explicit ModelInfo instances to avoid needing a tokenizer.
Slow tests use real tokenizers for Qwen3 (non-prefixed, <think>), OLMo-3
(prefixed, <think>), and Ministral (non-prefixed, [THINK]).

Skip slow tests with: pytest --no-slow
"""

import pytest

from thinkpack.model import ModelInfo, TagStyle, detect_model
from thinkpack.parse import ParsedResponse, parse


# convenience model_info instances shared across non-slow tests
_THINK = ModelInfo(prefixed=False)
_THINK_PREFIXED = ModelInfo(prefixed=True)
_THINKING = ModelInfo(prefixed=False, tag_content="thinking")
_REASONING = ModelInfo(prefixed=False, tag_content="reasoning")
_BRACKET_THINK = ModelInfo(
    prefixed=False, tag_content="THINK", tag_style=TagStyle.BRACKET
)


# ---------------------------------------------------------------------------
# non-slow unit tests — use explicit ModelInfo to avoid needing a tokenizer
# ---------------------------------------------------------------------------


class TestParse:
    """Tests for parse() with a single string input."""

    def test_standard_format(self) -> None:
        """Standard <think>...</think> format is parsed correctly."""
        response = "<think>\nsome reasoning\n</think>\nthe answer"
        result = parse(response=response, model_info=_THINK)

        assert result.answer == "the answer"
        assert result.reasoning == "some reasoning"
        assert result.reasoning_tag == "think"
        assert result.has_valid_reasoning is True
        assert result.has_invalid_reasoning is False
        assert result.has_truncated_reasoning is False

    def test_olmo_style_format(self) -> None:
        """Close-tag-only format (no opening tag in output) is parsed correctly."""
        # prefixed models inject <think> via the chat template, so decoded output has no open tag
        response = "some reasoning\n</think>\nthe answer"
        result = parse(response=response, model_info=_THINK)

        assert result.answer == "the answer"
        assert result.has_valid_reasoning is True

    def test_plain_response_no_tags(self) -> None:
        """A response with no reasoning tags is returned as a plain answer."""
        response = "just the answer"
        result = parse(response=response, model_info=_THINK)

        assert result.answer == "just the answer"
        assert result.reasoning == ""
        assert result.reasoning_tag is None
        assert result.has_missing_reasoning is True
        assert result.has_valid_reasoning is False
        assert result.has_truncated_reasoning is False

    def test_truncated_standard(self) -> None:
        """A response with an open tag but no closing tag is marked as truncated."""
        response = "<think>\nreasoning started but never finis"
        result = parse(response=response, model_info=_THINK)

        assert result.has_missing_reasoning is False
        assert result.has_valid_reasoning is False
        assert result.has_truncated_reasoning is True
        assert result.answer == ""

    def test_truncated_prefixed(self) -> None:
        """Prefixed model with no tags: treated as truncated reasoning."""
        response = "reasoning started but never finished"
        result = parse(response=response, model_info=_THINK_PREFIXED)

        assert result.has_missing_reasoning is False
        assert result.has_truncated_reasoning is True
        assert result.answer == ""

    def test_empty_reasoning_block(self) -> None:
        """An empty think block is not counted as valid reasoning."""
        response = "<think>\n</think>\nthe answer"
        result = parse(response=response, model_info=_THINK)

        assert result.has_missing_reasoning is False
        assert result.has_valid_reasoning is False

    def test_thinking_tag_variant(self) -> None:
        """Alternative tag names (thinking, reasoning, thought) are handled."""
        response = "<thinking>\nsome thoughts\n</thinking>\nthe answer"
        result = parse(response=response, model_info=_THINKING)

        assert result.reasoning_tag == "thinking"
        assert result.has_valid_reasoning is True

    def test_no_arg_raises(self) -> None:
        """Calling parse() without tokenizer or model_info raises ValueError."""
        with pytest.raises(ValueError, match="tokenizer or model_info"):
            parse(response="some response")

    def test_valid_answer_true(self) -> None:
        """valid_answer is True when the answer contains non-whitespace text."""
        result = parse(response="<think>\nr\n</think>\nthe answer", model_info=_THINK)

        assert result.extracted_answer is True

    def test_valid_answer_false_on_empty(self) -> None:
        """valid_answer is False when the answer is empty (e.g. truncated response)."""
        result = parse(response="<think>\nstarted...", model_info=_THINK)

        assert result.extracted_answer is False


class TestParseCustomTag:
    """Tests for parse() with explicit non-default ModelInfo tag settings."""

    def test_custom_tag_standard_format(self) -> None:
        """Custom tag with matching open and close is parsed correctly."""
        response = "<reasoning>\nsome thoughts\n</reasoning>\nthe answer"
        result = parse(response=response, model_info=_REASONING)

        assert result.reasoning_tag == "reasoning"
        assert result.has_valid_reasoning is True
        assert result.answer == "the answer"
        assert result.has_truncated_reasoning is False

    def test_custom_tag_truncated(self) -> None:
        """Custom tag with open tag but no close tag is marked as truncated."""
        response = "<reasoning>\nthoughts that never finish"
        result = parse(response=response, model_info=_REASONING)

        assert result.reasoning_tag == "reasoning"
        assert result.has_truncated_reasoning is True
        assert result.has_valid_reasoning is False
        assert result.answer == ""

    def test_custom_tag_no_match_returns_plain(self) -> None:
        """Tag not present in response returns a plain answer result."""
        response = "just a plain answer"
        result = parse(response=response, model_info=_REASONING)

        assert result.has_missing_reasoning is True
        assert result.answer == "just a plain answer"

    def test_bracket_tag_standard_format(self) -> None:
        """[THINK]...[/THINK] bracket-style format is parsed correctly."""
        response = "[THINK]\nsome thoughts\n[/THINK]\nthe answer"
        result = parse(response=response, model_info=_BRACKET_THINK)

        assert result.reasoning_tag == "THINK"
        assert result.has_valid_reasoning is True
        assert result.answer == "the answer"
        assert result.has_truncated_reasoning is False

    def test_bracket_tag_truncated(self) -> None:
        """Bracket tag with open but no close is marked as truncated."""
        response = "[THINK]\nthoughts that never finish"
        result = parse(response=response, model_info=_BRACKET_THINK)

        assert result.reasoning_tag == "THINK"
        assert result.has_truncated_reasoning is True
        assert result.has_valid_reasoning is False
        assert result.answer == ""

    def test_bracket_tag_no_match_html_response(self) -> None:
        """Bracket tag model_info but HTML tags in response: treated as plain answer."""
        response = "<think>\nthoughts\n</think>\nthe answer"
        result = parse(response=response, model_info=_BRACKET_THINK)

        assert result.has_missing_reasoning is True
        assert result.answer == response


class TestParseBatch:
    """Tests for parse() with list[str] and list[list[str]] inputs."""

    def test_flat_list(self) -> None:
        """parse() with list[str] returns a flat list of ParsedResponse."""
        responses = [
            "<think>\nr1\n</think>\na1",
            "plain answer",
        ]
        result = parse(response=responses, model_info=_THINK)

        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], ParsedResponse)
        assert result[0].has_valid_reasoning is True
        assert result[1].has_missing_reasoning is True

    def test_nested_list(self) -> None:
        """parse() with list[list[str]] returns a nested [task][sample] list."""
        responses = [
            ["<think>\nr1\n</think>\na1", "<think>\nr2\n</think>\na2"],
            ["plain answer"],
        ]
        result = parse(response=responses, model_info=_THINK)

        assert isinstance(result, list)
        assert len(result) == 2
        assert len(result[0]) == 2  # type: ignore[arg-type]
        assert len(result[1]) == 1  # type: ignore[arg-type]
        assert result[0][0].has_valid_reasoning is True  # type: ignore[index]
        assert result[1][0].has_missing_reasoning is True  # type: ignore[index]

    def test_empty_flat_list(self) -> None:
        """parse() with an empty list returns an empty list."""
        result = parse(response=[], model_info=_THINK)

        assert result == []


class TestParseEmptyReasoning:
    """Tests for the has_empty_reasoning flag on ParsedResponse."""

    def test_blank_block_sets_has_empty_reasoning(self) -> None:
        """A completed but blank reasoning block sets has_empty_reasoning=True."""
        result = parse(response="<think>\n   \n</think>\nthe answer", model_info=_THINK)

        assert result.has_empty_reasoning is True
        assert result.has_missing_reasoning is False
        assert result.has_valid_reasoning is False
        assert result.has_truncated_reasoning is False

    def test_valid_block_clears_has_empty_reasoning(self) -> None:
        """A completed, non-blank reasoning block sets has_empty_reasoning=False."""
        result = parse(
            response="<think>\nsome reasoning\n</think>\nthe answer",
            model_info=_THINK,
        )

        assert result.has_empty_reasoning is False
        assert result.has_valid_reasoning is True

    def test_truncated_block_clears_has_empty_reasoning(self) -> None:
        """A truncated block (no close tag) sets has_empty_reasoning=False."""
        result = parse(
            response="<think>\nreasoning that never finished",
            model_info=_THINK,
        )

        assert result.has_empty_reasoning is False
        assert result.has_truncated_reasoning is True

    def test_no_block_clears_has_empty_reasoning(self) -> None:
        """A plain response with no block sets has_empty_reasoning=False."""
        result = parse(response="just an answer", model_info=_THINK)

        assert result.has_empty_reasoning is False
        assert result.has_missing_reasoning is True

    def test_mutually_exclusive_flags_sum_to_one(self) -> None:
        """has_valid + has_truncated + has_empty + has_missing always equals 1."""
        responses = [
            parse(
                response="<think>\nreasoning\n</think>\nans", model_info=_THINK
            ),  # valid
            parse(response="<think>\nstarted...", model_info=_THINK),  # truncated
            parse(response="<think>\n\n</think>\nans", model_info=_THINK),  # empty
            parse(response="just an answer", model_info=_THINK),  # missing
        ]
        for r in responses:
            sub_total = (
                r.has_valid_reasoning
                + r.has_truncated_reasoning
                + r.has_empty_reasoning
                + r.has_missing_reasoning
            )
            assert sub_total == 1


# ---------------------------------------------------------------------------
# Qwen3 — non-prefixed, <think> tags
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestQwen3Parse:
    """parse() tests for Qwen/Qwen3-8B — non-prefixed, <think> tags."""

    def test_standard_format(self, qwen3_tokenizer) -> None:
        """Standard <think>...</think> response is parsed into reasoning and answer."""
        result = parse(
            response="<think>\nsome reasoning\n</think>\nthe answer",
            tokenizer=qwen3_tokenizer,
        )

        assert result.has_valid_reasoning is True
        assert result.reasoning == "some reasoning"
        assert result.answer == "the answer"
        assert result.reasoning_tag == "think"
        assert result.extracted_answer is True

    def test_tagless_is_plain_answer(self, qwen3_tokenizer) -> None:
        """Non-prefixed tokenizer: response without tags is a plain answer."""
        result = parse(
            response="just an answer",
            tokenizer=qwen3_tokenizer,
        )

        assert result.has_missing_reasoning is True
        assert result.answer == "just an answer"
        assert result.extracted_answer is True

    def test_token_counts_populated(self, qwen3_tokenizer) -> None:
        """Token counts are set when calculate_tokens=True."""
        result = parse(
            response="<think>\nsome reasoning\n</think>\nhi",
            tokenizer=qwen3_tokenizer,
            calculate_tokens=True,
        )

        assert result.reasoning_token_count is not None
        assert result.reasoning_token_count > 0
        assert result.answer_token_count is not None
        assert result.answer_token_count > 0

    def test_token_counts_none_by_default(self, qwen3_tokenizer) -> None:
        """Token counts remain None when calculate_tokens is not set."""
        result = parse(
            response="<think>\nreasoning\n</think>\nthe answer",
            tokenizer=qwen3_tokenizer,
        )

        assert result.reasoning_token_count is None
        assert result.answer_token_count is None

    def test_flat_list(self, qwen3_tokenizer) -> None:
        """Tokenizer is applied to every string in a flat list."""
        result = parse(
            response=["<think>\nreasoning\n</think>\nhi", "plain answer"],
            tokenizer=qwen3_tokenizer,
        )

        assert result[0].has_valid_reasoning is True
        assert result[1].has_missing_reasoning is True

    def test_nested_list(self, qwen3_tokenizer) -> None:
        """Tokenizer is applied to every string in a nested [task][sample] list."""
        result = parse(
            response=[["<think>\nreasoning\n</think>\nhi"], ["plain answer"]],
            tokenizer=qwen3_tokenizer,
        )

        assert result[0][0].has_valid_reasoning is True
        assert result[1][0].has_missing_reasoning is True


# ---------------------------------------------------------------------------
# OLMo-3 — prefixed, <think> tags
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestOlmo3Parse:
    """parse() tests for allenai/OLMo-3-7B-Think — prefixed, <think> tags.

    The template injects <think> into the generation prompt, so decoded output
    contains only the closing tag — the opening tag is never in the model output.
    """

    def test_prefixed_format(self, olmo3_tokenizer) -> None:
        """Close-tag-only response (prefixed template) is parsed correctly."""
        result = parse(
            response="some reasoning\n</think>\nthe answer",
            tokenizer=olmo3_tokenizer,
        )

        assert result.has_valid_reasoning is True
        assert result.answer == "the answer"
        assert result.extracted_answer is True

    def test_tagless_is_truncated(self, olmo3_tokenizer) -> None:
        """Prefixed tokenizer: response without any tags is treated as truncated."""
        result = parse(
            response="reasoning that never closed",
            tokenizer=olmo3_tokenizer,
        )

        assert result.has_truncated_reasoning is True
        assert result.has_missing_reasoning is False
        assert result.answer == ""

    def test_empty_answer_not_valid(self, olmo3_tokenizer) -> None:
        """valid_answer is False when the response is truncated (no answer produced)."""
        result = parse(
            response="reasoning that never closed",
            tokenizer=olmo3_tokenizer,
        )

        assert result.extracted_answer is False

    def test_model_info_param(self, olmo3_tokenizer) -> None:
        """Passing model_info directly gives the same result as passing a tokenizer."""
        model_info = detect_model(olmo3_tokenizer)
        response = "reasoning\n</think>\nthe answer"

        result_via_tokenizer = parse(response=response, tokenizer=olmo3_tokenizer)
        result_via_model_info = parse(response=response, model_info=model_info)

        assert (
            result_via_model_info.has_valid_reasoning
            == result_via_tokenizer.has_valid_reasoning
        )
        assert result_via_model_info.answer == result_via_tokenizer.answer
        assert result_via_model_info.reasoning == result_via_tokenizer.reasoning


# ---------------------------------------------------------------------------
# Ministral — non-prefixed, [THINK] bracket tags
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMinistralParse:
    """parse() tests for Ministral-3-3B-Reasoning-2512 — non-prefixed, [THINK] tags."""

    def test_bracket_tag_format(self, ministral_tokenizer) -> None:
        """[THINK]...[/THINK] response is parsed correctly."""
        result = parse(
            response="[THINK]\nsome reasoning\n[/THINK]\nthe answer",
            tokenizer=ministral_tokenizer,
        )

        assert result.has_valid_reasoning is True
        assert result.reasoning == "some reasoning"
        assert result.answer == "the answer"
        assert result.reasoning_tag == "THINK"
        assert result.extracted_answer is True

    def test_tagless_is_plain_answer(self, ministral_tokenizer) -> None:
        """Non-prefixed tokenizer: response without tags is a plain answer."""
        result = parse(
            response="just an answer",
            tokenizer=ministral_tokenizer,
        )

        assert result.has_missing_reasoning is True
        assert result.answer == "just an answer"

    def test_token_counts_populated(self, ministral_tokenizer) -> None:
        """Token counts are set when calculate_tokens=True."""
        result = parse(
            response="[THINK]\nsome reasoning\n[/THINK]\nhi",
            tokenizer=ministral_tokenizer,
            calculate_tokens=True,
        )

        assert result.reasoning_token_count is not None
        assert result.reasoning_token_count > 0
        assert result.answer_token_count is not None
        assert result.answer_token_count > 0
