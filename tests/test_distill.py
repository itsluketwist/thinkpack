"""Tests for thinkpack.distill — distillation prompt building and reasoning extraction."""

from thinkpack.distill import (
    build_prompts,
    extract_distilled_reasoning,
    to_conversations,
    update_records,
)


_RECORD = {"instruction": "What is 2 + 2?", "response": "4"}


class TestBuildPrompts:
    """Tests for build_prompts()."""

    def test_default_contains_instruction_response_tag(self) -> None:
        """Default prompt contains the instruction, response, and default distill tag."""
        prompts = build_prompts(records=[_RECORD])
        assert len(prompts) == 1
        p = prompts[0]
        assert "What is 2 + 2?" in p
        assert "4" in p
        assert "<reasoning_steps>" in p

    def test_default_preamble_present(self) -> None:
        """Default preamble appears at the start of the prompt."""
        prompts = build_prompts(records=[_RECORD])
        assert prompts[0].startswith("I need assistance")

    def test_custom_preamble_replaces_default(self) -> None:
        """A custom preamble is used instead of the default."""
        prompts = build_prompts(records=[_RECORD], preamble="My custom preamble.")
        assert prompts[0].startswith("My custom preamble.")
        assert "I need assistance" not in prompts[0]

    def test_custom_tag_appears_in_prompt(self) -> None:
        """A custom distill_tag name appears in the prompt."""
        prompts = build_prompts(records=[_RECORD], distill_tag="my_tag")
        assert "<my_tag>" in prompts[0]

    def test_example_absent_by_default(self) -> None:
        """No custom example block is present when reasoning_example=None."""
        prompts = build_prompts(records=[_RECORD])
        assert "Here is a complete example:" not in prompts[0]

    def test_example_present_with_tags(self) -> None:
        """When reasoning_example is provided it appears with the distill tags."""
        prompts = build_prompts(
            records=[_RECORD],
            distill_tag="reasoning_trace",
            reasoning_example="Step 1: add the numbers.",
        )
        p = prompts[0]
        assert "Here is a complete example:" in p
        assert "Step 1: add the numbers." in p

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
    """Tests for extract_distilled_reasoning() with a single string input."""

    def test_default_tag_extracts_content(self) -> None:
        """Extracts content from the default <reasoning_steps> block."""
        result = extract_distilled_reasoning(
            text="<reasoning_steps>\nmy reasoning\n</reasoning_steps>\nanswer",
        )
        assert result == "my reasoning"

    def test_default_tag_stop_token_case(self) -> None:
        """Open tag with no close tag (stop-token scenario) still extracts content."""
        result = extract_distilled_reasoning(
            text="<reasoning_steps>\nmy reasoning here",
        )
        assert result == "my reasoning here"

    def test_no_tags_returns_none(self) -> None:
        """Plain text with no distill tag returns None."""
        result = extract_distilled_reasoning(text="just a plain answer")
        assert result is None

    def test_blank_content_returns_none(self) -> None:
        """Distill tag with blank content returns None."""
        result = extract_distilled_reasoning(
            text="<reasoning_steps>\n   \n</reasoning_steps>\nanswer"
        )
        assert result is None

    def test_last_instance_used(self) -> None:
        """When the open tag appears multiple times, content after the last one is extracted."""
        text = "<reasoning_steps>\nfirst attempt\n</reasoning_steps>\n<reasoning_steps>\nsecond attempt"
        result = extract_distilled_reasoning(text=text)
        assert result == "second attempt"

    def test_custom_distill_tag(self) -> None:
        """Custom distill_tag is respected for both open and close tag matching."""
        result = extract_distilled_reasoning(
            text="<reasoning_trace>\nmy trace here",
            distill_tag="reasoning_trace",
        )
        assert result == "my trace here"

    def test_custom_tag_with_closing_tag(self) -> None:
        """Extraction stops at the closing tag when it is present."""
        result = extract_distilled_reasoning(
            text="<reasoning_trace>\nmy trace\n</reasoning_trace>\nextra text",
            distill_tag="reasoning_trace",
        )
        assert result == "my trace"

    def test_tag_not_found_returns_none(self) -> None:
        """Tag that doesn't appear in the text returns None."""
        result = extract_distilled_reasoning(
            text="no custom tag here", distill_tag="reasoning_trace"
        )
        assert result is None

    def test_blank_content_after_last_tag_returns_none(self) -> None:
        """Blank content after the last open tag returns None."""
        result = extract_distilled_reasoning(
            text="<reasoning_trace>\n   ", distill_tag="reasoning_trace"
        )
        assert result is None


class TestExtractReasoningList:
    """Tests for extract_distilled_reasoning() with a list input."""

    def test_list_returns_list(self) -> None:
        """List input returns a list of the same length."""
        texts = [
            "<reasoning_steps>\nreasoning one\n</reasoning_steps>",
            "plain text",
            "<reasoning_steps>\nreasoning three",
        ]
        results = extract_distilled_reasoning(text=texts)
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
        responses = ["<reasoning_steps>\nmy reasoning\n</reasoning_steps>"]
        updated = update_records(records=records, responses=responses)
        assert "reasoning" in updated[0]
        assert updated[0]["reasoning"] == "my reasoning"

    def test_field_absent_where_extraction_fails(self) -> None:
        """Records where extraction fails do not get the field added."""
        records = [{"instruction": "Q", "response": "A"}]
        responses = ["plain response with no reasoning tags"]
        updated = update_records(records=records, responses=responses)
        assert "reasoning" not in updated[0]

    def test_custom_field_name(self) -> None:
        """Custom field name is used instead of the default."""
        records = [{"instruction": "Q", "response": "A"}]
        responses = ["<reasoning_steps>\nreasoning\n</reasoning_steps>"]
        updated = update_records(
            records=records,
            responses=responses,
            reasoning_field="reasoning_natural",
        )
        assert "reasoning_natural" in updated[0]
        assert "reasoning" not in updated[0]

    def test_original_records_not_mutated(self) -> None:
        """Original record dicts are not modified."""
        original = {"instruction": "Q", "response": "A"}
        records = [original]
        responses = ["<reasoning_steps>\nreasoning\n</reasoning_steps>"]
        update_records(records=records, responses=responses)
        assert "reasoning" not in original

    def test_mixed_success_and_failure(self) -> None:
        """Some records succeed, others fail — both handled correctly."""
        records = [
            {"instruction": "Q1", "response": "A1"},
            {"instruction": "Q2", "response": "A2"},
        ]
        responses = [
            "<reasoning_steps>\ngood reasoning\n</reasoning_steps>",
            "no reasoning here",
        ]
        updated = update_records(records=records, responses=responses)
        assert "reasoning" in updated[0]
        assert "reasoning" not in updated[1]

    def test_custom_tag_extraction(self) -> None:
        """update_records passes distill_tag through to extract_distilled_reasoning."""
        records = [{"instruction": "Q", "response": "A"}]
        responses = ["<reasoning_trace>\nmy trace"]
        updated = update_records(
            records=records,
            responses=responses,
            distill_tag="reasoning_trace",
        )
        assert updated[0]["reasoning"] == "my trace"


class TestToConversations:
    """Tests for to_conversations()."""

    def test_single_record_with_reasoning(self) -> None:
        """Record with reasoning produces a conversation with reasoning on assistant message."""
        records = [
            {"instruction": "What?", "response": "This.", "reasoning": "step by step"}
        ]
        convs = to_conversations(records=records)
        assert len(convs) == 1
        conv = convs[0]
        assert conv[0] == {"role": "user", "content": "What?"}
        assert conv[1]["role"] == "assistant"
        assert conv[1]["content"] == "This."
        assert conv[1]["reasoning"] == "step by step"

    def test_record_without_reasoning_omits_key(self) -> None:
        """Record without a reasoning key produces an assistant message without 'reasoning'."""
        records = [{"instruction": "Q", "response": "A"}]
        convs = to_conversations(records=records)
        assert "reasoning" not in convs[0][1]

    def test_multiple_records_correct_length(self) -> None:
        """Returns one conversation per record."""
        records = [
            {"instruction": "Q1", "response": "A1"},
            {"instruction": "Q2", "response": "A2", "reasoning": "r2"},
        ]
        convs = to_conversations(records=records)
        assert len(convs) == 2
        assert "reasoning" not in convs[0][1]
        assert convs[1][1]["reasoning"] == "r2"

    def test_custom_keys(self) -> None:
        """Custom instruction_key, response_key, and reasoning_key are respected."""
        records = [{"q": "Q?", "a": "A.", "r": "the reason"}]
        convs = to_conversations(
            records=records,
            instruction_key="q",
            response_key="a",
            reasoning_key="r",
        )
        conv = convs[0]
        assert conv[0]["content"] == "Q?"
        assert conv[1]["content"] == "A."
        assert conv[1]["reasoning"] == "the reason"

    def test_output_compatible_with_apply_mask_interface(self) -> None:
        """Output conversations have the expected structure for apply_mask."""
        records = [{"instruction": "Q", "response": "A", "reasoning": "R"}]
        convs = to_conversations(records=records)
        conv = convs[0]
        # verify structure: list of two dicts, user then assistant
        assert len(conv) == 2
        assert conv[0]["role"] == "user"
        assert conv[1]["role"] == "assistant"
