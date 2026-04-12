"""Tests for thinkpack.parse — response parsing into reasoning and answer."""

from thinkpack.parse import parse, parse_all


class TestParse:
    """Tests for the parse() function."""

    def test_standard_format(self) -> None:
        """Standard <think>...</think> format is parsed correctly."""
        response = "<think>\nsome reasoning\n</think>\nthe answer"
        result = parse(response=response)

        assert result.answer == "the answer"
        assert result.reasoning == "\nsome reasoning\n"
        assert result.reasoning_tag == "think"
        assert result.has_reasoning_block is True
        assert result.has_valid_reasoning is True
        assert result.has_truncated_reasoning is False

    def test_olmo_style_format(self) -> None:
        """OLMo-style format (no opening tag in output) is parsed correctly."""
        # olmo models inject <think> via the chat template, so decoded output has no open tag
        response = "some reasoning\n</think>\nthe answer"
        result = parse(response=response)

        assert result.answer == "the answer"
        assert result.has_valid_reasoning is True

    def test_plain_response_no_tags(self) -> None:
        """A response with no reasoning tags is returned as plain answer."""
        response = "just the answer"
        result = parse(response=response)

        assert result.answer == "just the answer"
        assert result.reasoning == ""
        assert result.reasoning_tag is None
        assert result.has_reasoning_block is False
        assert result.has_valid_reasoning is False
        assert result.has_truncated_reasoning is False

    def test_truncated_standard(self) -> None:
        """A response with an open tag but no closing tag is marked as truncated."""
        response = "<think>\nreasoning started but never finis"
        result = parse(response=response)

        assert result.has_reasoning_block is True
        assert result.has_valid_reasoning is False
        assert result.has_truncated_reasoning is True
        assert result.answer == ""

    def test_truncated_prefixed(self) -> None:
        """PREFIXED template truncation (no tags at all) is detected with prefixed=True."""
        response = "reasoning started but never finished"
        result = parse(response=response, prefixed=True)

        assert result.has_reasoning_block is True
        assert result.has_truncated_reasoning is True
        assert result.answer == ""

    def test_empty_reasoning_block(self) -> None:
        """An empty think block is not counted as valid reasoning."""
        response = "<think>\n</think>\nthe answer"
        result = parse(response=response)

        assert result.has_reasoning_block is True
        assert result.has_valid_reasoning is False

    def test_thinking_tag_variant(self) -> None:
        """Alternative tag names (thinking, reasoning, thought) are handled."""
        response = "<thinking>\nsome thoughts\n</thinking>\nthe answer"
        result = parse(response=response)

        assert result.reasoning_tag == "thinking"
        assert result.has_valid_reasoning is True


class TestParseCustomTag:
    """Tests for parse() with an explicit tag= argument."""

    def test_custom_tag_standard_format(self) -> None:
        """Custom tag with matching open and close is parsed correctly."""
        response = "<reasoning>\nsome thoughts\n</reasoning>\nthe answer"
        result = parse(response=response, tag="reasoning")

        assert result.reasoning_tag == "reasoning"
        assert result.has_valid_reasoning is True
        assert result.answer == "the answer"
        assert result.has_truncated_reasoning is False

    def test_custom_tag_truncated(self) -> None:
        """Custom tag with open tag but no close tag is marked as truncated."""
        response = "<reasoning>\nthoughts that never finish"
        result = parse(response=response, tag="reasoning")

        assert result.reasoning_tag == "reasoning"
        assert result.has_truncated_reasoning is True
        assert result.has_valid_reasoning is False
        assert result.answer == ""

    def test_custom_tag_no_match_returns_plain(self) -> None:
        """Custom tag not present in response returns a plain answer result."""
        response = "just a plain answer"
        result = parse(response=response, tag="reasoning")

        assert result.has_reasoning_block is False
        assert result.answer == "just a plain answer"


class TestParseAll:
    """Tests for the parse_all() batch function."""

    def test_basic_batch(self) -> None:
        """parse_all returns correct shape and contents for a simple batch."""
        responses = [
            ["<think>\nr1\n</think>\na1", "<think>\nr2\n</think>\na2"],
            ["plain answer"],
        ]
        result = parse_all(responses=responses)

        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 1
        assert result[0][0].has_valid_reasoning is True
        assert result[1][0].has_reasoning_block is False
