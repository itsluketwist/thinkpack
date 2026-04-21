"""Tests for thinkpack.parse — response parsing into reasoning and answer."""

from typing import cast

from thinkpack.parse import (
    ParsedResponse,
    _GenerationOutput,
    parse,
    parse_all,
    parse_output,
)


# ---------------------------------------------------------------------------
# mock tokenizers — implement the _Tokenizer protocol without real models
# ---------------------------------------------------------------------------


class _MockTokenizerBase:
    """Base for mock tokenizers; encode returns one token per character."""

    chat_template: str = ""

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        raise NotImplementedError

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int = 0,
    ) -> list[int]:
        # one token per character — makes expected counts trivially verifiable
        return list(range(len(text)))

    def decode(self, token_ids: list[int]) -> str:
        return ""


class MockPrefixedTokenizer(_MockTokenizerBase):
    """Simulates a PREFIXED-style tokenizer (e.g. OLMo-3): generation prompt ends with <think>."""

    chat_template = "mock_prefixed_template"

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        # native reasoning_content check raises so detection falls through to PREFIXED
        if any("reasoning_content" in m for m in conversation):
            raise ValueError("unsupported field")
        if add_generation_prompt:
            return "<|im_start|>assistant\n<think>"
        return "<|im_start|>assistant\n"


class MockInlineTokenizer(_MockTokenizerBase):
    """Simulates an INLINE-style tokenizer: generation prompt has no trailing tag."""

    chat_template = "mock_inline_template"

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        if any("reasoning_content" in m for m in conversation):
            raise ValueError("unsupported field")
        return "<|im_start|>assistant\n"


class MockCompletion:
    """Minimal stand-in for a vLLM CompletionOutput (has a .text attribute)."""

    def __init__(self, text: str) -> None:
        self.text = text


class MockRequestOutput:
    """Minimal stand-in for a vLLM RequestOutput (has an .outputs list)."""

    def __init__(self, *texts: str) -> None:
        # each text becomes one sample/completion in .outputs
        self.outputs = [MockCompletion(t) for t in texts]


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


class TestParseOutput:
    """Tests for parse_output() — parses vLLM-style generation output objects."""

    def test_single_output_single_sample(self) -> None:
        """A single RequestOutput with one completion returns a flat list."""
        output = MockRequestOutput("<think>\nr\n</think>\nthe answer")
        result = cast(list[ParsedResponse], parse_output(output=output))

        assert len(result) == 1
        assert result[0].answer == "the answer"
        assert result[0].has_valid_reasoning is True

    def test_single_output_multiple_samples(self) -> None:
        """A single RequestOutput with multiple completions returns a flat list."""
        output = MockRequestOutput(
            "<think>\nr1\n</think>\na1",
            "<think>\nr2\n</think>\na2",
        )
        result = cast(list[ParsedResponse], parse_output(output=output))

        assert len(result) == 2
        assert result[0].answer == "a1"
        assert result[1].answer == "a2"

    def test_list_of_outputs_returns_nested_list(self) -> None:
        """A list of RequestOutputs returns a nested [task][sample] structure."""
        outputs = cast(
            list[_GenerationOutput],
            [
                MockRequestOutput("<think>\nr\n</think>\na1"),
                MockRequestOutput("plain answer"),
            ],
        )
        result = cast(list[list[ParsedResponse]], parse_output(output=outputs))

        assert len(result) == 2
        # each inner list contains the samples for one task
        assert result[0][0].has_valid_reasoning is True
        assert result[1][0].has_reasoning_block is False

    def test_prefixed_flag_forwarded(self) -> None:
        """The prefixed flag is passed through to parse() for each completion."""
        # prefixed=True treats the whole text as a truncated think block when no tags
        output = MockRequestOutput("reasoning without close tag")
        result = cast(list[ParsedResponse], parse_output(output=output, prefixed=True))

        assert result[0].has_truncated_reasoning is True

    def test_tag_forwarded(self) -> None:
        """The tag argument is passed through to parse() for each completion."""
        output = MockRequestOutput("<reasoning>\nthoughts\n</reasoning>\nans")
        result = cast(
            list[ParsedResponse], parse_output(output=output, tag="reasoning")
        )

        assert result[0].reasoning_tag == "reasoning"
        assert result[0].has_valid_reasoning is True


class TestParseEmptyReasoning:
    """Tests for the has_empty_reasoning flag on ParsedResponse."""

    def test_blank_block_sets_has_empty_reasoning(self) -> None:
        """A completed but blank reasoning block sets has_empty_reasoning=True."""
        result = parse(response="<think>\n   \n</think>\nthe answer")

        assert result.has_empty_reasoning is True
        assert result.has_reasoning_block is True
        assert result.has_valid_reasoning is False
        assert result.has_truncated_reasoning is False

    def test_valid_block_clears_has_empty_reasoning(self) -> None:
        """A completed, non-blank reasoning block sets has_empty_reasoning=False."""
        result = parse(response="<think>\nsome reasoning\n</think>\nthe answer")

        assert result.has_empty_reasoning is False
        assert result.has_valid_reasoning is True

    def test_truncated_block_clears_has_empty_reasoning(self) -> None:
        """A truncated block (no close tag) sets has_empty_reasoning=False."""
        result = parse(response="<think>\nreasoning that never finished")

        assert result.has_empty_reasoning is False
        assert result.has_truncated_reasoning is True

    def test_no_block_clears_has_empty_reasoning(self) -> None:
        """A plain response with no block sets has_empty_reasoning=False."""
        result = parse(response="just an answer")

        assert result.has_empty_reasoning is False
        assert result.has_reasoning_block is False

    def test_mutually_exclusive_flags_sum_to_has_reasoning_block(self) -> None:
        """has_valid + has_truncated + has_empty always equals has_reasoning_block."""
        responses = [
            parse(response="<think>\nreasoning\n</think>\nans"),  # valid
            parse(response="<think>\nstarted..."),  # truncated
            parse(response="<think>\n\n</think>\nans"),  # empty
            parse(response="just an answer"),  # no block
        ]
        for r in responses:
            sub_total = (
                r.has_valid_reasoning
                + r.has_truncated_reasoning
                + r.has_empty_reasoning
            )
            assert sub_total == r.has_reasoning_block


class TestParseTokenizerDetection:
    """Tests for automatic prefixed detection via the tokenizer argument."""

    def test_prefixed_tokenizer_detects_prefixed_style(self) -> None:
        """A PREFIXED tokenizer causes tagless output to be treated as truncated."""
        # no tags in response — without tokenizer this would be a plain answer
        result = parse(
            response="reasoning without any tags", tokenizer=MockPrefixedTokenizer()
        )

        assert result.has_truncated_reasoning is True
        assert result.has_reasoning_block is True
        assert result.answer == ""

    def test_inline_tokenizer_treats_tagless_as_plain(self) -> None:
        """An INLINE tokenizer leaves tagless output as a plain answer."""
        result = parse(response="just an answer", tokenizer=MockInlineTokenizer())

        assert result.has_reasoning_block is False
        assert result.answer == "just an answer"

    def test_tokenizer_overrides_explicit_prefixed_false(self) -> None:
        """A PREFIXED tokenizer overrides prefixed=False passed explicitly."""
        result = parse(
            response="reasoning without any tags",
            prefixed=False,
            tokenizer=MockPrefixedTokenizer(),
        )

        assert result.has_truncated_reasoning is True

    def test_tokenizer_overrides_explicit_prefixed_true(self) -> None:
        """An INLINE tokenizer overrides prefixed=True passed explicitly."""
        result = parse(
            response="just an answer",
            prefixed=True,
            tokenizer=MockInlineTokenizer(),
        )

        # INLINE detection means the response is treated as a plain answer, not truncated
        assert result.has_reasoning_block is False
        assert result.answer == "just an answer"

    def test_prefixed_tokenizer_forwarded_through_parse_all(self) -> None:
        """Tokenizer is forwarded to every parse() call inside parse_all()."""
        responses = [["reasoning with no tags"], ["also no tags"]]
        result = parse_all(responses=responses, tokenizer=MockPrefixedTokenizer())

        assert result[0][0].has_truncated_reasoning is True
        assert result[1][0].has_truncated_reasoning is True

    def test_prefixed_tokenizer_forwarded_through_parse_output(self) -> None:
        """Tokenizer is forwarded to every parse() call inside parse_output()."""
        output = MockRequestOutput("reasoning with no tags", "also no tags")
        result = cast(
            list[ParsedResponse],
            parse_output(output=output, tokenizer=MockPrefixedTokenizer()),
        )

        assert result[0].has_truncated_reasoning is True
        assert result[1].has_truncated_reasoning is True


class TestParseCalculateTokens:
    """Tests for the calculate_tokens argument."""

    def test_token_counts_populated_when_requested(self) -> None:
        """Token counts are set when calculate_tokens=True and a tokenizer is provided."""
        response = "<think>\nabc\n</think>\nhi"
        result = parse(
            response=response,
            tokenizer=MockInlineTokenizer(),
            calculate_tokens=True,
        )

        # mock encode returns one token per character
        assert result.reasoning_token_count == len("\nabc\n")
        assert result.answer_token_count == len("hi")

    def test_token_counts_none_without_tokenizer(self) -> None:
        """Token counts remain None when calculate_tokens=True but no tokenizer given."""
        result = parse(
            response="<think>\nreasoning\n</think>\nthe answer",
            calculate_tokens=True,
        )

        assert result.reasoning_token_count is None
        assert result.answer_token_count is None

    def test_token_counts_none_by_default(self) -> None:
        """Token counts are None by default even when a tokenizer is provided."""
        result = parse(
            response="<think>\nreasoning\n</think>\nthe answer",
            tokenizer=MockInlineTokenizer(),
        )

        assert result.reasoning_token_count is None
        assert result.answer_token_count is None

    def test_token_counts_zero_for_empty_strings(self) -> None:
        """Empty reasoning or answer produces a token count of zero."""
        # truncated response has no answer
        result = parse(
            response="<think>\nstarted...",
            tokenizer=MockInlineTokenizer(),
            calculate_tokens=True,
        )

        assert result.answer_token_count == 0

    def test_calculate_tokens_forwarded_through_parse_all(self) -> None:
        """calculate_tokens is forwarded to every parse() call inside parse_all()."""
        responses = [["<think>\nabc\n</think>\nhi"]]
        result = parse_all(
            responses=responses,
            tokenizer=MockInlineTokenizer(),
            calculate_tokens=True,
        )

        assert result[0][0].reasoning_token_count is not None
        assert result[0][0].answer_token_count is not None

    def test_calculate_tokens_forwarded_through_parse_output(self) -> None:
        """calculate_tokens is forwarded to every parse() call inside parse_output()."""
        output = MockRequestOutput("<think>\nabc\n</think>\nhi")
        result = cast(
            list[ParsedResponse],
            parse_output(
                output=output,
                tokenizer=MockInlineTokenizer(),
                calculate_tokens=True,
            ),
        )

        assert result[0].reasoning_token_count is not None
        assert result[0].answer_token_count is not None
