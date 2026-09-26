"""Tests for thinkpack.mask — training-time loss masking for reasoning blocks."""

import logging
from typing import Any, cast

import pytest

from thinkpack.mask import MaskType, apply_mask


# ---------------------------------------------------------------------------
# helpers — decode only the masked or unmasked portion of a tokenized sequence
# ---------------------------------------------------------------------------


def _decode_masked(tokenizer, row: dict) -> str:
    """Decode the tokens whose label is -100 (masked from the loss)."""
    ids = [
        row["input_ids"][i] for i, label in enumerate(row["labels"]) if label == -100
    ]
    return tokenizer.decode(ids)


def _decode_unmasked(tokenizer, row: dict) -> str:
    """Decode the tokens that contribute to the loss (label matches input_id)."""
    ids = [
        row["input_ids"][i] for i, label in enumerate(row["labels"]) if label != -100
    ]
    return tokenizer.decode(ids)


def _conversation(
    instruction: str = "test question",
    reasoning: str | None = "detailed reasoning here",
    response: str = "final answer",
) -> list[dict[str, str]]:
    """Build a training conversation; pass reasoning=None to omit the key from the assistant message."""
    assistant: dict[str, str] = {"role": "assistant", "content": response}
    if reasoning is not None:
        assistant["reasoning"] = reasoning
    return [
        {"role": "user", "content": instruction},
        assistant,
    ]


# ---------------------------------------------------------------------------
# enum tests — no tokenizer required, always fast
# ---------------------------------------------------------------------------


class TestMaskEnum:
    """Tests for the MaskType IntFlag enum."""

    def test_values(self) -> None:
        assert MaskType.PROMPT == 1
        assert MaskType.THINK == 2
        assert MaskType.RESPONSE == 4

    def test_combination_contains_both_flags(self) -> None:
        combined = MaskType.PROMPT | MaskType.THINK
        assert MaskType.PROMPT in combined
        assert MaskType.THINK in combined
        assert MaskType.RESPONSE not in combined

    def test_zero_is_falsy(self) -> None:
        assert not MaskType(0)

    def test_nonzero_is_truthy(self) -> None:
        assert MaskType.THINK
        assert MaskType.PROMPT | MaskType.THINK


# ---------------------------------------------------------------------------
# integration tests — require real tokenizers, marked slow
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMaskOutputStructure:
    """Tests for the shape and columns of the Dataset returned by apply_mask()."""

    def test_returns_three_columns(self, qwen3_tokenizer) -> None:
        ds = apply_mask(conversations=[_conversation()], tokenizer=qwen3_tokenizer)
        assert set(ds.column_names) == {"input_ids", "labels", "attention_mask"}

    def test_length_matches_conversation_count(self, qwen3_tokenizer) -> None:
        convs = [_conversation(), _conversation(instruction="q2", response="a2")]
        ds = apply_mask(conversations=convs, tokenizer=qwen3_tokenizer)
        assert len(ds) == 2

    def test_empty_conversations_returns_empty_dataset(self, qwen3_tokenizer) -> None:
        ds = apply_mask(conversations=[], tokenizer=qwen3_tokenizer)
        assert len(ds) == 0

    def test_attention_mask_all_ones(self, qwen3_tokenizer) -> None:
        ds = apply_mask(conversations=[_conversation()], tokenizer=qwen3_tokenizer)
        assert all(v == 1 for v in ds[0]["attention_mask"])

    def test_input_ids_labels_same_length(self, qwen3_tokenizer) -> None:
        ds = apply_mask(conversations=[_conversation()], tokenizer=qwen3_tokenizer)
        row = ds[0]
        assert len(row["input_ids"]) == len(row["labels"]) == len(row["attention_mask"])


@pytest.mark.slow
class TestMaskThink:
    """Tests for MaskType.THINK — reasoning block masked, prompt and response trained."""

    def test_inline_model_masks_reasoning_not_response(
        self,
        qwen3_tokenizer,
    ) -> None:
        """THINK masking: reasoning text is masked, response is trained on."""
        ds = apply_mask(
            conversations=[_conversation()],
            tokenizer=qwen3_tokenizer,
            masked=MaskType.THINK,
        )
        row = ds[0]

        assert "detailed reasoning here" in _decode_masked(qwen3_tokenizer, row)
        assert "final answer" in _decode_unmasked(qwen3_tokenizer, row)
        # the prompt (instruction) is also trained on when only THINK is masked
        assert "test question" in _decode_unmasked(qwen3_tokenizer, row)

    def test_prefixed_model_masks_reasoning_not_response(
        self,
        deepseek_r1_llama_tokenizer,
    ) -> None:
        """THINK masking works identically for PREFIXED models."""
        ds = apply_mask(
            conversations=[_conversation()],
            tokenizer=deepseek_r1_llama_tokenizer,
            masked=MaskType.THINK,
        )
        row = ds[0]

        masked_text = _decode_masked(deepseek_r1_llama_tokenizer, row)
        assert "detailed reasoning here" in masked_text
        unmasked_text = _decode_unmasked(deepseek_r1_llama_tokenizer, row)
        assert "final answer" in unmasked_text
        assert "detailed" not in unmasked_text

    def test_native_key_model_masks_reasoning_correctly(
        self,
        qwen3_tokenizer,
    ) -> None:
        """Qwen3 passes reasoning via reasoning_content; masking still covers it."""
        # Qwen3 uses reasoning_content natively — the reasoning is rendered inside
        # <think> tags by the template, and the masking logic finds and masks it
        ds = apply_mask(
            conversations=[_conversation()],
            tokenizer=qwen3_tokenizer,
            masked=MaskType.THINK,
        )
        row = ds[0]

        assert "detailed reasoning here" in _decode_masked(qwen3_tokenizer, row)
        assert "final answer" in _decode_unmasked(qwen3_tokenizer, row)


@pytest.mark.slow
class TestMaskPrompt:
    """Tests for MaskType.PROMPT — instruction masked, reasoning and response trained."""

    def test_prompt_masks_instruction_not_response(self, qwen3_tokenizer) -> None:
        """PROMPT masking: instruction is masked, response is trained on."""
        ds = apply_mask(
            conversations=[_conversation()],
            tokenizer=qwen3_tokenizer,
            masked=MaskType.PROMPT,
        )
        row = ds[0]

        assert "test question" in _decode_masked(qwen3_tokenizer, row)
        assert "final answer" in _decode_unmasked(qwen3_tokenizer, row)


@pytest.mark.slow
class TestMaskPromptAndThink:
    """Tests for MaskType.PROMPT | MaskType.THINK — only response is trained."""

    def test_only_response_is_unmasked(self, qwen3_tokenizer) -> None:
        """PROMPT | THINK masking: only the response contributes to the loss."""
        ds = apply_mask(
            conversations=[_conversation()],
            tokenizer=qwen3_tokenizer,
            masked=MaskType.PROMPT | MaskType.THINK,
        )
        row = ds[0]

        # both instruction and reasoning are masked
        masked_text = _decode_masked(qwen3_tokenizer, row)
        assert "test question" in masked_text
        assert "detailed reasoning here" in masked_text
        # only the response is trained on
        assert "final answer" in _decode_unmasked(qwen3_tokenizer, row)


@pytest.mark.slow
class TestMaskNone:
    """Tests for masked=None — all tokens trained, no labels set to -100."""

    def test_no_labels_masked(self, qwen3_tokenizer) -> None:
        """masked=None trains on all tokens — no -100 labels."""
        ds = apply_mask(
            conversations=[_conversation()],
            tokenizer=qwen3_tokenizer,
            masked=None,
        )
        assert all(v != -100 for v in ds[0]["labels"])

    def test_labels_equal_input_ids(self, qwen3_tokenizer) -> None:
        """When masked=None, labels are identical to input_ids."""
        ds = apply_mask(
            conversations=[_conversation()],
            tokenizer=qwen3_tokenizer,
            masked=None,
        )
        row = ds[0]
        assert row["input_ids"] == row["labels"]


@pytest.mark.slow
class TestMaskMiscellaneous:
    """Additional apply_mask() behaviour tests."""

    def test_conversation_without_reasoning_gets_empty_think_block_injected(
        self,
        qwen3_tokenizer,
    ) -> None:
        """When masking is active, missing 'reasoning' key gets an empty block injected."""
        ds = apply_mask(
            conversations=[_conversation(reasoning=None)],
            tokenizer=qwen3_tokenizer,
            masked=MaskType.THINK,
        )
        # the injected empty think block creates some masked tokens
        assert -100 in ds[0]["labels"]

    def test_conversation_without_reasoning_unmasked_when_masked_none(
        self,
        qwen3_tokenizer,
    ) -> None:
        """Conversations without 'reasoning' are left unchanged when masked=None."""
        ds = apply_mask(
            conversations=[_conversation(reasoning=None)],
            tokenizer=qwen3_tokenizer,
            masked=None,
        )
        assert all(v != -100 for v in ds[0]["labels"])

    def test_custom_tag_override(self, qwen3_tokenizer) -> None:
        """The override_tag parameter overrides the detected tag — block is found and masked."""
        # override to <reasoning> so _build_assistant_message embeds the block
        # with that tag, and _tokenize_record searches for it correctly
        ds = apply_mask(
            conversations=[_conversation()],
            tokenizer=qwen3_tokenizer,
            masked=MaskType.THINK,
            override_tag="reasoning",
        )
        assert -100 in ds[0]["labels"]
        row = ds[0]
        assert "detailed reasoning here" in _decode_masked(qwen3_tokenizer, row)

    def test_multiple_conversations_all_processed(self, qwen3_tokenizer) -> None:
        """All conversations in the list are tokenized and masked independently."""
        convs = [
            _conversation(instruction="q1", reasoning="r1", response="a1"),
            _conversation(instruction="q2", reasoning="r2", response="a2"),
        ]
        ds = apply_mask(
            conversations=convs,
            tokenizer=qwen3_tokenizer,
            masked=MaskType.THINK,
        )
        assert len(ds) == 2
        for i in range(2):
            assert -100 in ds[i]["labels"]

    def test_max_seq_length_truncates(self, qwen3_tokenizer) -> None:
        """max_seq_length truncates token sequences that exceed the limit."""
        ds = apply_mask(
            conversations=[_conversation()],
            tokenizer=qwen3_tokenizer,
            max_seq_length=20,
        )
        assert len(ds[0]["input_ids"]) <= 20

    def test_custom_ignore_index(self, qwen3_tokenizer) -> None:
        """A custom ignore_index is used instead of the -100 default."""
        ds = apply_mask(
            conversations=[_conversation()],
            tokenizer=qwen3_tokenizer,
            masked=MaskType.THINK,
            ignore_index=-999,
        )
        assert any(v == -999 for v in ds[0]["labels"])
        assert -100 not in ds[0]["labels"]


# ---------------------------------------------------------------------------
# regression tests — boundary location across all supported models
# ---------------------------------------------------------------------------


_ALL_MODELS = [
    "qwen3_tokenizer",
    "qwen35_tokenizer",
    "deepseek_r1_llama_tokenizer",
    "olmo3_tokenizer",
    "ministral_tokenizer",
]


@pytest.mark.slow
@pytest.mark.parametrize("fixture_name", _ALL_MODELS)
class TestMaskBoundaries:
    """The final reasoning block and response are located correctly for every model."""

    @pytest.mark.parametrize(
        "response",
        [
            "final answer",
            # templates such as qwen3.5 trim content, so the response text changes
            "final answer with a trailing newline\n",
            "\n\nfinal answer with leading newlines",
            # short responses can also appear inside special tokens like <|im_end|>
            "d",
            "e",
        ],
    )
    def test_only_response_trained(
        self,
        request: pytest.FixtureRequest,
        fixture_name: str,
        response: str,
    ) -> None:
        """PROMPT | THINK masking: the trained tokens are exactly the response (+ end tokens)."""
        tokenizer = request.getfixturevalue(fixture_name)

        ds = apply_mask(
            conversations=[_conversation(response=response)],
            tokenizer=tokenizer,
            masked=MaskType.PROMPT | MaskType.THINK,
        )
        row = ds[0]

        assert _decode_unmasked(tokenizer, row).lstrip().startswith(response.strip())
        assert "detailed reasoning here" in _decode_masked(tokenizer, row)
        assert "test question" in _decode_masked(tokenizer, row)

    def test_single_think_block(
        self,
        request: pytest.FixtureRequest,
        fixture_name: str,
    ) -> None:
        """The reasoning appears exactly once in the training sequence (no duplicate blocks)."""
        tokenizer = request.getfixturevalue(fixture_name)

        for masked in [MaskType.THINK, None]:
            ds = apply_mask(
                conversations=[_conversation()],
                tokenizer=tokenizer,
                masked=masked,
            )
            full_text = tokenizer.decode(ds[0]["input_ids"])
            assert full_text.count("detailed reasoning here") == 1

    def test_empty_reasoning_single_block_unmasked(
        self,
        request: pytest.FixtureRequest,
        fixture_name: str,
    ) -> None:
        """reasoning='' with masked=None (the "empty" strategy) adds exactly one empty block."""
        tokenizer = request.getfixturevalue(fixture_name)
        open_tag = "[THINK]" if fixture_name == "ministral_tokenizer" else "<think>"

        ds = apply_mask(
            conversations=[_conversation(reasoning="")],
            tokenizer=tokenizer,
            masked=None,
        )
        full_text = tokenizer.decode(ds[0]["input_ids"])

        # count only tags in the assistant turn (ministral's system prompt has its own)
        assistant_text = full_text[full_text.rfind("test question") :]
        assert assistant_text.count(open_tag) == 1

    def test_history_block_not_masked_as_final(
        self,
        request: pytest.FixtureRequest,
        fixture_name: str,
    ) -> None:
        """Reasoning in an earlier turn does not move the THINK boundary of the final turn."""
        tokenizer = request.getfixturevalue(fixture_name)
        conversation = [
            {"role": "user", "content": "first question"},
            {
                "role": "assistant",
                "content": "first answer",
                "reasoning": "old thoughts",
            },
            {"role": "user", "content": "second question"},
            # no reasoning key on the final turn — an empty block is injected
            {"role": "assistant", "content": "second answer"},
        ]

        ds = apply_mask(
            conversations=[conversation],
            tokenizer=tokenizer,
            masked=MaskType.THINK,
            add_history_reasoning=True,
        )
        unmasked = _decode_unmasked(tokenizer, ds[0])

        # everything outside the final (empty) think block is trained on
        assert "old thoughts" in unmasked
        assert "second question" in unmasked
        assert "second answer" in unmasked


@pytest.mark.slow
class TestMaskSystemPrompts:
    """Reasoning tags inside a system prompt do not affect masking."""

    def test_system_prompt_mentioning_think_tag(self, qwen3_tokenizer) -> None:
        """THINK masking leaves a system prompt that mentions <think> trainable."""
        conversation = [
            {"role": "system", "content": "Put your reasoning inside <think> tags."},
            *_conversation(),
        ]

        ds = apply_mask(
            conversations=[conversation],
            tokenizer=qwen3_tokenizer,
            masked=MaskType.THINK,
        )
        unmasked = _decode_unmasked(qwen3_tokenizer, ds[0])

        assert "Put your reasoning inside <think> tags." in unmasked
        assert "test question" in unmasked
        assert "detailed reasoning here" not in unmasked

    def test_ministral_default_system_prompt_think(self, ministral_tokenizer) -> None:
        """Ministral's default system prompt contains [THINK]; the instruction stays trained."""
        ds = apply_mask(
            conversations=[_conversation()],
            tokenizer=ministral_tokenizer,
            masked=MaskType.THINK,
        )
        unmasked = _decode_unmasked(ministral_tokenizer, ds[0])

        assert "test question" in unmasked
        assert "final answer" in unmasked
        assert "detailed reasoning here" not in unmasked

    def test_ministral_prompt_mask_covers_instruction(
        self, ministral_tokenizer
    ) -> None:
        """PROMPT masking on Ministral masks the instruction, not just the system prompt."""
        ds = apply_mask(
            conversations=[_conversation()],
            tokenizer=ministral_tokenizer,
            masked=MaskType.PROMPT,
        )
        row = ds[0]

        assert "test question" in _decode_masked(ministral_tokenizer, row)
        assert "detailed reasoning here" in _decode_unmasked(ministral_tokenizer, row)
        assert "final answer" in _decode_unmasked(ministral_tokenizer, row)


@pytest.mark.slow
class TestMaskValidation:
    """Invalid inputs fail loudly, and silent data loss is reported."""

    def test_last_message_must_be_assistant(self, qwen3_tokenizer) -> None:
        """A conversation that does not end with an assistant message raises ValueError."""
        with pytest.raises(ValueError, match="Conversation 0"):
            apply_mask(
                conversations=[[{"role": "user", "content": "q"}]],
                tokenizer=qwen3_tokenizer,
            )

    def test_truncation_warning(
        self,
        qwen3_tokenizer,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Truncating a conversation logs a warning, as does leaving no trainable tokens."""
        with caplog.at_level(logging.WARNING, logger="thinkpack"):
            apply_mask(
                conversations=[_conversation()],
                tokenizer=qwen3_tokenizer,
                masked=MaskType.PROMPT | MaskType.THINK,
                max_seq_length=5,
            )

        assert "were truncated" in caplog.text
        assert "no trainable tokens" in caplog.text

    def test_no_warning_without_truncation(
        self,
        qwen3_tokenizer,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A short conversation produces no warnings."""
        with caplog.at_level(logging.WARNING, logger="thinkpack"):
            apply_mask(
                conversations=[_conversation()],
                tokenizer=qwen3_tokenizer,
            )

        assert caplog.text == ""


class _ProcessorStub:
    """Mimics a multimodal processor: wraps a tokenizer but has no encode() method."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer


@pytest.mark.slow
class TestMaskProcessor:
    """Multimodal processors (e.g. Qwen3.5 loaded via AutoProcessor) are unwrapped."""

    def test_processor_matches_tokenizer(self, qwen35_tokenizer) -> None:
        """Passing a processor gives the same dataset as passing its tokenizer."""
        # typed as Any, since the stub only implements part of the tokenizer protocol
        processor = cast(Any, _ProcessorStub(qwen35_tokenizer))

        from_processor = apply_mask(
            conversations=[_conversation()],
            tokenizer=processor,
        )
        from_tokenizer = apply_mask(
            conversations=[_conversation()],
            tokenizer=qwen35_tokenizer,
        )

        assert from_processor[0] == from_tokenizer[0]
