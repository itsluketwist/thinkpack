"""Tests for thinkpack.steer — inference-time thought-steering prefix injection."""

from thinkpack.steer import SimplePrefix, apply_steer_template, steer


class MockTokenizer:
    """Minimal mock tokenizer for steer() tests."""

    def __init__(self, prefixed: bool = False, open_tag: str = "<think>") -> None:
        # prefixed=True simulates PREFIXED template style — generation prompt ends with open_tag
        self._prefixed = prefixed
        self._open_tag = open_tag
        # mimic a chat_template string that encodes the prefixed/tag variant so
        # each distinct mock configuration gets its own detect_model() cache entry
        self.chat_template = f"basic template prefixed={prefixed} open_tag={open_tag}"

    def apply_chat_template(
        self,
        messages: list,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> str:
        """Return a fake templated prompt, ending with open_tag for PREFIXED style."""
        base = "".join(m["content"] for m in messages)
        if add_generation_prompt and self._prefixed:
            return f"{base}{self._open_tag}"
        return base


class TestSteerNonePrefix:
    """Tests for steer() with prefix=None — just ensures the reasoning tag is open."""

    def test_inline_think_appended(self) -> None:
        """For an INLINE template, <think> is appended when not already present."""
        tokenizer = MockTokenizer(prefixed=False)
        result = steer(prompts=["my prompt"], tokenizer=tokenizer, prefix=None)

        assert result[0] == "my prompt<think>\n"

    def test_prefixed_template_unchanged(self) -> None:
        """For a PREFIXED template that already ends with the tag, prompt is unchanged."""
        tokenizer = MockTokenizer(prefixed=True)
        result = steer(prompts=["my prompt<think>"], tokenizer=tokenizer, prefix=None)

        assert result[0] == "my prompt<think>"
        assert result[0].count("<think>") == 1

    def test_prefixed_alternative_tag_unchanged(self) -> None:
        """A PREFIXED template using <reasoning> is detected and left unchanged."""
        tokenizer = MockTokenizer(prefixed=True, open_tag="<reasoning>")
        result = steer(
            prompts=["my prompt<reasoning>"], tokenizer=tokenizer, prefix=None
        )

        assert result[0] == "my prompt<reasoning>"
        assert "<think>" not in result[0]


class TestSteerStringPrefix:
    """Tests for steer() with a string prefix — seeds the model's thought."""

    def test_inline_full_prefix_injected(self) -> None:
        """For an INLINE template, <think>\\n{prefix} is appended."""
        tokenizer = MockTokenizer(prefixed=False)
        result = steer(prompts=["my prompt"], tokenizer=tokenizer, prefix="Okay, ")

        assert result[0] == "my prompt<think>\nOkay, "

    def test_prefixed_template_no_duplicate_tag(self) -> None:
        """For a PREFIXED template, only the body is appended — no duplicate tag."""
        tokenizer = MockTokenizer(prefixed=True)
        result = steer(
            prompts=["my prompt<think>"], tokenizer=tokenizer, prefix="Okay, "
        )

        assert "<think><think>" not in result[0]
        assert "Okay, " in result[0]

    def test_inline_always_uses_think_tag(self) -> None:
        """INLINE models always have <think> injected — we control the tag, not the template."""
        tokenizer = MockTokenizer(prefixed=False)
        result = steer(prompts=["my prompt"], tokenizer=tokenizer, prefix="Okay, ")

        assert result[0] == "my prompt<think>\nOkay, "

    def test_prefixed_alternative_tag_no_duplicate(self) -> None:
        """A PREFIXED model using <thought> does not duplicate the tag."""
        tokenizer = MockTokenizer(prefixed=True, open_tag="<thought>")
        result = steer(
            prompts=["my prompt<thought>"], tokenizer=tokenizer, prefix="Okay, "
        )

        assert "<thought><thought>" not in result[0]
        assert "Okay, " in result[0]

    def test_simple_prefix_enum_works_as_string(self) -> None:
        """SimplePrefix values are strings and work transparently as prefix."""
        tokenizer = MockTokenizer(prefixed=False)
        result = steer(
            prompts=["my prompt"],
            tokenizer=tokenizer,
            prefix=SimplePrefix.BRIEF,
        )

        assert result[0] == f"my prompt<think>\n{SimplePrefix.BRIEF}"

    def test_all_simple_prefixes_produce_valid_output(self) -> None:
        """Every SimplePrefix value produces a prompt containing a reasoning tag."""
        tokenizer = MockTokenizer(prefixed=False)
        for sp in SimplePrefix:
            result = steer(prompts=["prompt"], tokenizer=tokenizer, prefix=sp)
            assert "<think>" in result[0]
            assert str(sp) in result[0]

    def test_multiple_prompts(self) -> None:
        """Multiple prompts are all steered correctly."""
        tokenizer = MockTokenizer(prefixed=False)
        result = steer(
            prompts=["one", "two", "three"],
            tokenizer=tokenizer,
            prefix="Okay, ",
        )

        assert len(result) == 3
        for r in result:
            assert "<think>\nOkay, " in r


class TestApplySteerTemplate:
    """Tests for apply_steer_template() — combined template + steer."""

    def test_single_conversation(self) -> None:
        """A single conversation is templated and steered correctly."""
        tokenizer = MockTokenizer(prefixed=False)
        conversations = [[{"role": "user", "content": "hello"}]]
        result = apply_steer_template(
            conversations=conversations,
            tokenizer=tokenizer,
            prefix="Okay, ",
        )

        assert len(result) == 1
        assert "<think>\nOkay, " in result[0]

    def test_multiple_conversations(self) -> None:
        """Multiple conversations each produce a steered prompt."""
        tokenizer = MockTokenizer(prefixed=False)
        conversations = [
            [{"role": "user", "content": "one"}],
            [{"role": "user", "content": "two"}],
        ]
        result = apply_steer_template(
            conversations=conversations,
            tokenizer=tokenizer,
        )

        assert len(result) == 2
        for r in result:
            assert "<think>" in r

    def test_none_prefix_ensures_think(self) -> None:
        """prefix=None still ensures the reasoning tag is appended."""
        tokenizer = MockTokenizer(prefixed=False)
        result = apply_steer_template(
            conversations=[[{"role": "user", "content": "q"}]],
            tokenizer=tokenizer,
            prefix=None,
        )

        assert result[0].endswith("<think>\n")
