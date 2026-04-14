"""Tests for thinkpack.mask — training-time loss masking for reasoning blocks."""

from thinkpack.mask import Mask, mask


# sentinel used by detect_model() for NATIVE detection — must match _model.py
_NATIVE_SENTINEL = "__thinkpack_detect__"


class MockInlineTokenizer:
    """
    Minimal mock tokenizer with INLINE template style.

    Uses character-level encoding (one token per character) so boundary
    positions in labels map directly to character offsets in the rendered text,
    making assertions straightforward.
    """

    # unique string so detect_model() cache entry is keyed to this mock only
    chat_template = "mock-inline-v1"

    def apply_chat_template(
        self,
        messages: list,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> str:
        """Return a simple bracketed template; no trailing xml tag (INLINE)."""
        parts = []
        for m in messages:
            if m["role"] == "user":
                parts.append(f"[U]{m['content']}[/U]")
            elif m["role"] == "assistant":
                # ignore reasoning_content if present — INLINE does not handle it
                parts.append(f"[A]{m['content']}[/A]")
        result = "".join(parts)
        if add_generation_prompt:
            # no trailing xml tag — ensures detect_model() returns INLINE
            result += "[A]"
        return result

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int = 32768,
    ) -> list[int]:
        """Character-level encoding: one token per character."""
        tokens = [ord(c) % 1000 for c in text]
        if truncation:
            tokens = tokens[:max_length]
        return tokens


class MockNativeTokenizer:
    """
    Minimal mock tokenizer with NATIVE template style (e.g. Qwen3).

    Renders reasoning_content inside think tags when present, which allows
    detect_model() to identify the NATIVE style via the _NATIVE_SENTINEL probe.
    """

    chat_template = "mock-native-v1"

    def apply_chat_template(
        self,
        messages: list,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> str:
        """Include reasoning_content inside <think> tags when the field is present."""
        parts = []
        for m in messages:
            if m["role"] == "user":
                parts.append(f"[U]{m['content']}[/U]")
            elif m["role"] == "assistant":
                reasoning = m.get("reasoning_content", "")
                if reasoning:
                    # native templates wrap reasoning in dedicated think tags
                    parts.append(
                        f"[A]<think>\n{reasoning}\n</think>\n{m['content']}[/A]"
                    )
                else:
                    parts.append(f"[A]{m['content']}[/A]")
        return "".join(parts)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int = 32768,
    ) -> list[int]:
        """Character-level encoding: one token per character."""
        tokens = [ord(c) % 1000 for c in text]
        if truncation:
            tokens = tokens[:max_length]
        return tokens


class MockPrefixedTokenizer:
    """
    Minimal mock tokenizer with PREFIXED template style.

    Appends the open_tag at the end of the generation prompt, which is what
    detect_model() looks for to classify a tokenizer as PREFIXED.
    """

    def __init__(self, open_tag: str = "<think>") -> None:
        self._open_tag = open_tag
        # unique string per tag variant so each config gets its own cache entry
        self.chat_template = f"mock-prefixed-v1 tag={open_tag}"

    def apply_chat_template(
        self,
        messages: list,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> str:
        """Append open_tag at the end of the generation prompt (PREFIXED style)."""
        parts = []
        for m in messages:
            if m["role"] == "user":
                parts.append(f"[U]{m['content']}[/U]")
            elif m["role"] == "assistant":
                parts.append(f"[A]{m['content']}[/A]")
        result = "".join(parts)
        if add_generation_prompt:
            # trailing xml tag triggers PREFIXED detection in detect_model()
            result += self._open_tag
        return result

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int = 32768,
    ) -> list[int]:
        """Character-level encoding: one token per character."""
        tokens = [ord(c) % 1000 for c in text]
        if truncation:
            tokens = tokens[:max_length]
        return tokens


# module-level tokenizer instances — reused across tests so detect_model() only
# runs once per mock configuration (cache is keyed on chat_template string)
_INLINE = MockInlineTokenizer()
_NATIVE = MockNativeTokenizer()
_PREFIXED = MockPrefixedTokenizer()


def _record(
    instruction: str = "hi",
    reasoning: str | None = "step1",
    response: str = "answer",
) -> dict[str, str]:
    """Build a training record; pass reasoning=None to omit the key entirely."""
    r: dict[str, str] = {"instruction": instruction, "response": response}
    if reasoning is not None:
        r["reasoning"] = reasoning
    return r


# expected rendered text for the default _record() with INLINE/PREFIXED/NATIVE:
# "[U]hi[/U][A]<think>\nstep1\n</think>\nanswer[/A]"
# character positions:
#   <think>  starts at 12
#   answer   starts at 35
_DEFAULT_FULL_TEXT = "[U]hi[/U][A]<think>\nstep1\n</think>\nanswer[/A]"
_THINK_START = _DEFAULT_FULL_TEXT.index("<think>")  # 12
_RESPONSE_START = _DEFAULT_FULL_TEXT.rfind("answer")  # 35


class TestMaskEnum:
    """Tests for the Mask IntFlag enum."""

    def test_values(self) -> None:
        assert Mask.PROMPT == 1
        assert Mask.THINK == 2
        assert Mask.RESPONSE == 4

    def test_combination_contains_both_flags(self) -> None:
        combined = Mask.PROMPT | Mask.THINK
        assert Mask.PROMPT in combined
        assert Mask.THINK in combined
        assert Mask.RESPONSE not in combined

    def test_zero_is_falsy(self) -> None:
        assert not Mask(0)

    def test_nonzero_is_truthy(self) -> None:
        assert Mask.THINK
        assert Mask.PROMPT | Mask.THINK


class TestMaskOutputStructure:
    """Tests for the shape and columns of the Dataset returned by mask()."""

    def test_returns_three_columns(self) -> None:
        ds = mask(records=[_record()], tokenizer=_INLINE)
        assert set(ds.column_names) == {"input_ids", "labels", "attention_mask"}

    def test_length_matches_record_count(self) -> None:
        records = [_record(), _record(instruction="q2", response="a2")]
        ds = mask(records=records, tokenizer=_INLINE)
        assert len(ds) == 2

    def test_empty_records_returns_empty_dataset(self) -> None:
        ds = mask(records=[], tokenizer=_INLINE)
        assert len(ds) == 0

    def test_attention_mask_all_ones(self) -> None:
        ds = mask(records=[_record()], tokenizer=_INLINE)
        assert all(v == 1 for v in ds[0]["attention_mask"])

    def test_input_ids_labels_attention_mask_same_length(self) -> None:
        ds = mask(records=[_record()], tokenizer=_INLINE)
        row = ds[0]
        assert len(row["input_ids"]) == len(row["labels"]) == len(row["attention_mask"])


class TestMaskInline:
    """Tests for mask() with an INLINE-style tokenizer."""

    def test_mask_think_masks_only_think_block(self) -> None:
        """Mask.THINK sets -100 from the opening tag to the start of the response."""
        ds = mask(records=[_record()], tokenizer=_INLINE, masked=Mask.THINK)
        labels = ds[0]["labels"]

        # inside the think block — all masked
        assert all(v == -100 for v in labels[_THINK_START:_RESPONSE_START])
        # before the think block (prompt region) — not masked
        assert all(v != -100 for v in labels[:_THINK_START])
        # the response — not masked
        assert all(v != -100 for v in labels[_RESPONSE_START : _RESPONSE_START + 6])

    def test_mask_prompt_masks_before_think_block(self) -> None:
        """Mask.PROMPT sets -100 for tokens before the think block opens."""
        ds = mask(records=[_record()], tokenizer=_INLINE, masked=Mask.PROMPT)
        labels = ds[0]["labels"]

        assert all(v == -100 for v in labels[:_THINK_START])
        # think block itself — not masked
        assert all(v != -100 for v in labels[_THINK_START : _THINK_START + 7])

    def test_mask_response_masks_response_tokens(self) -> None:
        """Mask.RESPONSE sets -100 for the response tokens only."""
        ds = mask(records=[_record()], tokenizer=_INLINE, masked=Mask.RESPONSE)
        labels = ds[0]["labels"]

        assert all(v == -100 for v in labels[_RESPONSE_START : _RESPONSE_START + 6])
        # think block is not masked
        assert all(v != -100 for v in labels[_THINK_START:_RESPONSE_START])

    def test_mask_prompt_and_think_masks_up_to_response(self) -> None:
        """Mask.PROMPT | Mask.THINK masks everything before the response."""
        ds = mask(
            records=[_record()], tokenizer=_INLINE, masked=Mask.PROMPT | Mask.THINK
        )
        labels = ds[0]["labels"]

        assert all(v == -100 for v in labels[:_RESPONSE_START])
        assert all(v != -100 for v in labels[_RESPONSE_START : _RESPONSE_START + 6])

    def test_masked_none_no_labels_are_masked(self) -> None:
        """masked=None trains on all tokens — no -100 labels produced."""
        ds = mask(records=[_record()], tokenizer=_INLINE, masked=None)
        labels = ds[0]["labels"]
        assert all(v != -100 for v in labels)

    def test_masked_none_labels_equal_input_ids(self) -> None:
        """When masked=None, labels are identical to input_ids."""
        ds = mask(records=[_record()], tokenizer=_INLINE, masked=None)
        row = ds[0]
        assert row["input_ids"] == row["labels"]

    def test_custom_ignore_index(self) -> None:
        """A custom ignore_index is used instead of the -100 default."""
        ds = mask(
            records=[_record()],
            tokenizer=_INLINE,
            masked=Mask.THINK,
            ignore_index=-999,
        )
        labels = ds[0]["labels"]

        assert all(v == -999 for v in labels[_THINK_START:_RESPONSE_START])
        assert -100 not in labels

    def test_max_seq_length_truncates(self) -> None:
        """max_seq_length truncates token sequences that exceed the limit."""
        ds = mask(records=[_record()], tokenizer=_INLINE, max_seq_length=20)
        assert len(ds[0]["input_ids"]) <= 20

    def test_record_without_reasoning_gets_empty_think_block_injected(self) -> None:
        """
        When masking is active and a record has no 'reasoning' key, mask() injects
        an empty reasoning block so training context matches inference time.
        """
        ds = mask(
            records=[_record(reasoning=None)], tokenizer=_INLINE, masked=Mask.THINK
        )
        labels = ds[0]["labels"]
        # the injected empty think block creates some masked tokens
        assert -100 in labels

    def test_record_without_reasoning_not_modified_when_unmasked(self) -> None:
        """Records without 'reasoning' are left unchanged when masked=None."""
        ds = mask(records=[_record(reasoning=None)], tokenizer=_INLINE, masked=None)
        labels = ds[0]["labels"]
        # no think block in the sequence, no masking
        assert all(v != -100 for v in labels)

    def test_custom_tag_overrides_detected_open_tag(self) -> None:
        """The tag parameter overrides the model's detected open_tag."""
        ds = mask(
            records=[_record()],
            tokenizer=_INLINE,
            masked=Mask.THINK,
            tag="reasoning",
        )
        labels = ds[0]["labels"]
        # the think block is still found and masked (now using <reasoning>)
        assert -100 in labels

    def test_multiple_records_all_processed(self) -> None:
        """All records in the list are tokenized and masked independently."""
        records = [
            _record(instruction="q1", reasoning="r1", response="a1"),
            _record(instruction="q2", reasoning="r2", response="a2"),
        ]
        ds = mask(records=records, tokenizer=_INLINE, masked=Mask.THINK)
        assert len(ds) == 2
        for i in range(2):
            assert -100 in ds[i]["labels"]


class TestMaskNative:
    """Tests for mask() with a NATIVE-style tokenizer (e.g. Qwen3)."""

    def test_mask_think_masks_reasoning_block(self) -> None:
        """
        For NATIVE tokenizers, reasoning is passed as a separate field but the
        rendered text is equivalent — the think block is masked correctly.
        """
        ds = mask(records=[_record()], tokenizer=_NATIVE, masked=Mask.THINK)
        labels = ds[0]["labels"]

        # native rendering produces the same full_text as inline for our mock
        assert all(v == -100 for v in labels[_THINK_START:_RESPONSE_START])
        assert all(v != -100 for v in labels[_RESPONSE_START : _RESPONSE_START + 6])

    def test_masked_none_no_labels_masked(self) -> None:
        ds = mask(records=[_record()], tokenizer=_NATIVE, masked=None)
        assert all(v != -100 for v in ds[0]["labels"])

    def test_returns_dataset_with_correct_columns(self) -> None:
        ds = mask(records=[_record()], tokenizer=_NATIVE)
        assert set(ds.column_names) == {"input_ids", "labels", "attention_mask"}


class TestMaskPrefixed:
    """Tests for mask() with a PREFIXED-style tokenizer."""

    def test_mask_think_masks_correct_range(self) -> None:
        """PREFIXED tokenizer: masking boundaries are computed the same as INLINE."""
        ds = mask(records=[_record()], tokenizer=_PREFIXED, masked=Mask.THINK)
        labels = ds[0]["labels"]

        assert all(v == -100 for v in labels[_THINK_START:_RESPONSE_START])
        assert all(v != -100 for v in labels[:_THINK_START])

    def test_prefixed_alternative_tag_detected_and_used(self) -> None:
        """A PREFIXED tokenizer using <reasoning> correctly masks that tag's block."""
        tok = MockPrefixedTokenizer(open_tag="<reasoning>")
        ds = mask(records=[_record()], tokenizer=tok, masked=Mask.THINK)
        labels = ds[0]["labels"]
        # the <reasoning> block is found and masked
        assert -100 in labels

    def test_masked_none_no_labels_masked(self) -> None:
        ds = mask(records=[_record()], tokenizer=_PREFIXED, masked=None)
        assert all(v != -100 for v in ds[0]["labels"])
