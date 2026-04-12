"""Tests for thinkpack.distill — distillation prompt building and reasoning extraction."""

from thinkpack.distill import build_prompts, extract_reasoning, update_records


_RECORD = {"instruction": "What is 2 + 2?", "response": "4"}


class TestBuildPrompts:
    """Tests for build_prompts()."""

    def test_default_contains_instruction_response_tag(self) -> None:
        """Default prompt contains the instruction, response, and closing tag instruction."""
        prompts = build_prompts(records=[_RECORD])
        assert len(prompts) == 1
        p = prompts[0]
        assert "What is 2 + 2?" in p
        assert "4" in p
        assert "<reasoning_trace>" in p

    def test_default_preamble_present(self) -> None:
        """Default preamble appears at the start of the prompt."""
        prompts = build_prompts(records=[_RECORD])
        assert prompts[0].startswith("Given the following question")

    def test_custom_preamble_replaces_default(self) -> None:
        """A custom preamble is used instead of the default."""
        prompts = build_prompts(records=[_RECORD], preamble="My custom preamble.")
        assert prompts[0].startswith("My custom preamble.")
        assert "Given the following question" not in prompts[0]

    def test_custom_tag_appears_in_prompt(self) -> None:
        """A custom tag name appears in the closing instruction."""
        prompts = build_prompts(records=[_RECORD], tag="my_tag")
        assert "<my_tag>" in prompts[0]

    def test_example_absent_by_default(self) -> None:
        """No example block is present when example=None."""
        prompts = build_prompts(records=[_RECORD])
        assert "Here is an example:" not in prompts[0]

    def test_example_present_with_tags(self) -> None:
        """When example is provided it appears wrapped in the response tags."""
        prompts = build_prompts(
            records=[_RECORD],
            tag="reasoning_trace",
            example="Step 1: add the numbers.",
        )
        p = prompts[0]
        assert "Here is an example:" in p
        assert "<reasoning_trace>" in p
        assert "Step 1: add the numbers." in p
        assert "</reasoning_trace>" in p

    def test_multiple_records(self) -> None:
        """Returns one prompt per record."""
        records = [_RECORD, {"instruction": "Q2", "response": "A2"}]
        prompts = build_prompts(records=records)
        assert len(prompts) == 2
        assert "Q2" in prompts[1]
        assert "A2" in prompts[1]

    def test_custom_keys(self) -> None:
        """Custom instruction_key and response_key are respected."""
        records = [{"q": "What?", "a": "This."}]
        prompts = build_prompts(
            records=records,
            instruction_key="q",
            response_key="a",
        )
        assert "What?" in prompts[0]
        assert "This." in prompts[0]


class TestExtractReasoningSingle:
    """Tests for extract_reasoning() with a single string input."""

    def test_standard_think_tag(self) -> None:
        """Extracts content from a standard <think> block."""
        result = extract_reasoning(text="<think>\nmy reasoning\n</think>\nanswer")
        assert result == "my reasoning"

    def test_alternative_tags(self) -> None:
        """Extracts from <reasoning> and <thought> tags as well."""
        assert extract_reasoning(text="<reasoning>\nr\n</reasoning>\na") == "r"
        assert extract_reasoning(text="<thought>\nt\n</thought>\na") == "t"

    def test_no_tags_returns_none(self) -> None:
        """Plain text with no reasoning tags returns None."""
        result = extract_reasoning(text="just a plain answer")
        assert result is None

    def test_truncated_stop_token_case(self) -> None:
        """Open tag with no close tag (stop-token scenario) still extracts content."""
        result = extract_reasoning(text="<think>\nmy reasoning here")
        assert result == "my reasoning here"

    def test_blank_reasoning_returns_none(self) -> None:
        """Empty reasoning block returns None."""
        result = extract_reasoning(text="<think>\n   \n</think>\nanswer")
        assert result is None

    def test_custom_tag_extracts_after_open(self) -> None:
        """Custom tag extracts everything after the opening tag."""
        result = extract_reasoning(
            text="<reasoning_trace>\nmy trace here",
            tag="reasoning_trace",
        )
        assert result == "my trace here"

    def test_custom_tag_strips_think_block(self) -> None:
        """With strip_think=True, a preceding <think> block is removed before searching."""
        text = "<think>\ninner thought\n</think>\n<reasoning_trace>\nthe trace"
        result = extract_reasoning(text=text, tag="reasoning_trace", strip_think=True)
        assert result == "the trace"

    def test_custom_tag_no_strip_think(self) -> None:
        """With strip_think=False, the raw text is searched directly."""
        # without stripping, the <reasoning_trace> tag is still found after </think>
        text = "<think>\ninner\n</think>\n<reasoning_trace>\nthe trace"
        result = extract_reasoning(text=text, tag="reasoning_trace", strip_think=False)
        assert result == "the trace"

    def test_custom_tag_not_found_returns_none(self) -> None:
        """Custom tag that doesn't appear in the text returns None."""
        result = extract_reasoning(text="no custom tag here", tag="reasoning_trace")
        assert result is None

    def test_custom_tag_blank_content_returns_none(self) -> None:
        """Custom tag with blank content returns None."""
        result = extract_reasoning(text="<reasoning_trace>\n   ", tag="reasoning_trace")
        assert result is None


class TestExtractReasoningList:
    """Tests for extract_reasoning() with a list input."""

    def test_list_returns_list(self) -> None:
        """List input returns a list of the same length."""
        texts = [
            "<think>\nreasoning one\n</think>\nanswer",
            "plain text",
            "<think>\nreasoning three\n</think>\nanswer",
        ]
        results = extract_reasoning(text=texts)
        assert isinstance(results, list)
        assert len(results) == 3
        assert results[0] == "reasoning one"
        assert results[1] is None
        assert results[2] == "reasoning three"


class TestUpdateRecords:
    """Tests for update_records()."""

    def test_field_added_where_extraction_succeeds(self) -> None:
        """Records where extraction succeeds get the reasoning field added."""
        records = [{"instruction": "Q", "response": "A"}]
        responses = ["<think>\nmy reasoning\n</think>\nanswer"]
        updated = update_records(records=records, responses=responses)
        assert "reasoning_constructed" in updated[0]
        assert updated[0]["reasoning_constructed"] == "my reasoning"

    def test_field_absent_where_extraction_fails(self) -> None:
        """Records where extraction fails do not get the field added."""
        records = [{"instruction": "Q", "response": "A"}]
        responses = ["plain response with no reasoning tags"]
        updated = update_records(records=records, responses=responses)
        assert "reasoning_constructed" not in updated[0]

    def test_custom_field_name(self) -> None:
        """Custom field name is used instead of the default."""
        records = [{"instruction": "Q", "response": "A"}]
        responses = ["<think>\nreasoning\n</think>\nanswer"]
        updated = update_records(
            records=records,
            responses=responses,
            field="reasoning_natural",
        )
        assert "reasoning_natural" in updated[0]
        assert "reasoning_constructed" not in updated[0]

    def test_original_records_not_mutated(self) -> None:
        """Original record dicts are not modified."""
        original = {"instruction": "Q", "response": "A"}
        records = [original]
        responses = ["<think>\nreasoning\n</think>\nanswer"]
        update_records(records=records, responses=responses)
        assert "reasoning_constructed" not in original

    def test_mixed_success_and_failure(self) -> None:
        """Some records succeed, others fail — both handled correctly."""
        records = [
            {"instruction": "Q1", "response": "A1"},
            {"instruction": "Q2", "response": "A2"},
        ]
        responses = [
            "<think>\ngood reasoning\n</think>\nanswer",
            "no reasoning here",
        ]
        updated = update_records(records=records, responses=responses)
        assert "reasoning_constructed" in updated[0]
        assert "reasoning_constructed" not in updated[1]

    def test_custom_tag_extraction(self) -> None:
        """update_records passes tag through to extract_reasoning."""
        records = [{"instruction": "Q", "response": "A"}]
        responses = ["<reasoning_trace>\nmy trace"]
        updated = update_records(
            records=records,
            responses=responses,
            tag="reasoning_trace",
        )
        assert updated[0]["reasoning_constructed"] == "my trace"
