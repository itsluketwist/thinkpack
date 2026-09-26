"""Tests for ModelInfo, detect_model(), and get_model_info() in thinkpack.model."""

import logging
from collections.abc import Iterator
from typing import Any

import pytest

from thinkpack import model as model_module
from thinkpack.model import ModelInfo, TagStyle, detect_model, get_model_info


# ---------------------------------------------------------------------------
# ModelInfo unit tests — no tokenizer required, always fast
# ---------------------------------------------------------------------------


class TestModelInfoDefaults:
    """ModelInfo defaults to HTML think tags."""

    def test_defaults(self) -> None:
        info = ModelInfo(prefixed=False)

        assert info.prefixed is False
        assert info.tag_content == "think"
        assert info.tag_style == TagStyle.HTML
        assert info.open_tag == "<think>"
        assert info.close_tag == "</think>"


class TestModelInfoTags:
    """open_tag and close_tag produce correctly formatted strings."""

    def test_html_tags(self) -> None:
        info = ModelInfo(prefixed=False, tag_content="reasoning")

        assert info.prefixed is False
        assert info.tag_content == "reasoning"
        assert info.tag_style == TagStyle.HTML
        assert info.open_tag == "<reasoning>"
        assert info.close_tag == "</reasoning>"

    def test_bracket_tags(self) -> None:
        info = ModelInfo(
            prefixed=False,
            tag_content="THINK",
            tag_style=TagStyle.BRACKET,
        )

        assert info.prefixed is False
        assert info.tag_content == "THINK"
        assert info.tag_style == TagStyle.BRACKET
        assert info.open_tag == "[THINK]"
        assert info.close_tag == "[/THINK]"


class TestModelInfoWithTag:
    """with_tag() infers tag format from the string and returns a new object."""

    def test_raw_name_keeps_existing_style(self) -> None:
        """A plain name updates tag_content and preserves the existing tag_style."""
        html_result = ModelInfo(prefixed=False).with_tag("reasoning")
        bracket_result = ModelInfo(prefixed=False, tag_style=TagStyle.BRACKET).with_tag(
            "REASONING"
        )

        assert html_result.prefixed is False
        assert html_result.tag_content == "reasoning"
        assert html_result.tag_style == TagStyle.HTML

        assert bracket_result.prefixed is False
        assert bracket_result.tag_content == "REASONING"
        assert bracket_result.tag_style == TagStyle.BRACKET

    def test_html_formatted_tag(self) -> None:
        """<tag> format sets HTML style regardless of the original."""
        result = ModelInfo(prefixed=False, tag_style=TagStyle.BRACKET).with_tag(
            "<reasoning>"
        )

        assert result.prefixed is False
        assert result.tag_content == "reasoning"
        assert result.tag_style == TagStyle.HTML

    def test_bracket_formatted_tag(self) -> None:
        """[TAG] format sets BRACKET style regardless of the original."""
        result = ModelInfo(prefixed=False).with_tag("[THINK]")

        assert result.prefixed is False
        assert result.tag_content == "THINK"
        assert result.tag_style == TagStyle.BRACKET

    def test_other_fields_preserved_and_original_unchanged(self) -> None:
        """with_tag() keeps all other fields; the original is not modified."""
        info = ModelInfo(
            prefixed=True,
            tag_content="think",
            strips_think_tags=True,
            strips_history_think_tags=True,
        )
        result = info.with_tag("reasoning")

        assert result.prefixed is True
        assert result.tag_content == "reasoning"
        assert result.tag_style == TagStyle.HTML
        assert result.strips_think_tags is True
        assert result.strips_history_think_tags is True
        assert info.tag_content == "think"


# ---------------------------------------------------------------------------
# get_model_info() integration tests — require real tokenizers, marked slow
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestGetModelInfo:
    """get_model_info() detects model properties and applies optional tag overrides."""

    def test_qwen3_detected_values(self, qwen3_tokenizer) -> None:
        """Qwen3 is not prefixed and uses <think> tags."""
        info = get_model_info(tokenizer=qwen3_tokenizer)

        assert info.prefixed is False
        assert info.tag_content == "think"
        assert info.tag_style == TagStyle.HTML

    def test_qwen35_detected_values(self, qwen35_tokenizer) -> None:
        """Qwen3.5 is prefixed and uses <think> tags."""
        info = get_model_info(tokenizer=qwen35_tokenizer)

        assert info.prefixed is True
        assert info.tag_content == "think"
        assert info.tag_style == TagStyle.HTML

    def test_deepseek_r1_llama_detected_values(
        self,
        deepseek_r1_llama_tokenizer,
    ) -> None:
        """DeepSeek-R1-Llama is prefixed and uses <think> tags."""
        info = get_model_info(tokenizer=deepseek_r1_llama_tokenizer)

        assert info.prefixed is True
        assert info.tag_content == "think"
        assert info.tag_style == TagStyle.HTML

    def test_olmo3_detected_values(self, olmo3_tokenizer) -> None:
        """OLMo-3 is prefixed and uses <think> tags."""
        info = get_model_info(tokenizer=olmo3_tokenizer)

        assert info.prefixed is True
        assert info.tag_content == "think"
        assert info.tag_style == TagStyle.HTML

    def test_ministral_detected_values(self, ministral_tokenizer) -> None:
        """Ministral is not prefixed and uses bracket [THINK] tags, auto-detected."""
        info = detect_model(tokenizer=ministral_tokenizer)

        assert info.prefixed is False
        assert info.tag_content == "THINK"
        assert info.tag_style == TagStyle.BRACKET

    def test_raw_tag_override_qwen3(self, qwen3_tokenizer) -> None:
        """Raw tag override on an inline model — tag_content changes, other fields unchanged."""
        result = get_model_info(tokenizer=qwen3_tokenizer, override_tag="reasoning")

        assert result.prefixed is False
        assert result.tag_content == "reasoning"
        assert result.tag_style == TagStyle.HTML

    def test_raw_tag_override_deepseek(self, deepseek_r1_llama_tokenizer) -> None:
        """Raw tag override on a prefixed model — prefixed=True is preserved."""
        result = get_model_info(
            tokenizer=deepseek_r1_llama_tokenizer, override_tag="reasoning"
        )

        assert result.prefixed is True
        assert result.tag_content == "reasoning"
        assert result.tag_style == TagStyle.HTML

    def test_bracket_tag_override_qwen3(self, qwen3_tokenizer) -> None:
        """[TAG] override switches an HTML model to BRACKET style."""
        result = get_model_info(tokenizer=qwen3_tokenizer, override_tag="[THINK]")

        assert result.prefixed is False
        assert result.tag_content == "THINK"
        assert result.tag_style == TagStyle.BRACKET

    def test_bracket_tag_override_olmo3(self, olmo3_tokenizer) -> None:
        """[TAG] override switches a prefixed HTML model to BRACKET style."""
        result = get_model_info(tokenizer=olmo3_tokenizer, override_tag="[THINK]")

        assert result.prefixed is True
        assert result.tag_content == "THINK"
        assert result.tag_style == TagStyle.BRACKET

    def test_html_tag_override_ministral(self, ministral_tokenizer) -> None:
        """<tag> override switches a BRACKET model to HTML style."""
        result = get_model_info(tokenizer=ministral_tokenizer, override_tag="<think>")

        assert result.prefixed is False
        assert result.tag_content == "think"
        assert result.tag_style == TagStyle.HTML

    def test_tag_override_does_not_corrupt_cache(self, qwen3_tokenizer) -> None:
        """Tag overrides are not stored in the cache — detected values remain intact."""
        _ = get_model_info(tokenizer=qwen3_tokenizer, override_tag="reasoning")
        result = get_model_info(tokenizer=qwen3_tokenizer)

        assert result.prefixed is False
        assert result.tag_content == "think"
        assert result.tag_style == TagStyle.HTML


# ---------------------------------------------------------------------------
# detection edge cases — a stub tokenizer, no downloads required
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """A minimal tokenizer whose template renders messages as "<role>content" text.

    The template never strips reasoning blocks. gen_suffix is appended to the generation
    prompt, and drops_spaces simulates a broken tokenizer that loses spaces on encode.
    """

    def __init__(
        self,
        chat_template: Any,
        gen_suffix: str = "",
        drops_spaces: bool = False,
    ) -> None:
        self.chat_template = chat_template
        self.gen_suffix = gen_suffix
        self.drops_spaces = drops_spaces

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs: object,
    ) -> str:
        text = "".join(f"<{m['role']}>{m['content']}" for m in conversation)
        if add_generation_prompt:
            text += "<assistant>" + self.gen_suffix
        return text

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        **kwargs: object,
    ) -> list[int]:
        if self.drops_spaces:
            text = text.replace(" ", "")
        # one "token" per character is enough for a round-trip check
        return [ord(c) for c in text]

    def decode(
        self,
        token_ids: list[int],
    ) -> str:
        return "".join(chr(i) for i in token_ids)


def _stub(**kwargs: Any) -> Any:
    """Build a _StubTokenizer typed as Any, as it only implements part of the protocol.

    Returns the stub tokenizer.
    """
    return _StubTokenizer(**kwargs)


@pytest.fixture
def clear_detection_cache() -> Iterator[None]:
    """Clear the detection cache so each stub is detected afresh."""
    model_module._cache.clear()
    yield
    model_module._cache.clear()


@pytest.mark.usefixtures("clear_detection_cache")
class TestDetectModelStub:
    """detect_model() edge cases that real tokenizers do not cover."""

    def test_dict_chat_template(self) -> None:
        """A dict of named templates is supported."""
        tokenizer = _stub(
            chat_template={"default": "uses <think> tags", "tool_use": "other"},
        )

        info = detect_model(tokenizer=tokenizer)

        assert info.tag_content == "think"
        assert info.tag_style == TagStyle.HTML

    def test_trailing_reasoning_tag_is_prefixed(self) -> None:
        """A generation prompt ending with the reasoning tag (plus whitespace) is prefixed."""
        tokenizer = _stub(
            chat_template="uses <think> tags",
            gen_suffix="<think>\n",
        )

        assert detect_model(tokenizer=tokenizer).prefixed is True

    def test_trailing_other_tag_is_not_prefixed(self) -> None:
        """A generation prompt ending with a non-reasoning tag is not prefixed."""
        tokenizer = _stub(
            chat_template="uses <think> tags",
            gen_suffix="<answer>",
        )

        assert detect_model(tokenizer=tokenizer).prefixed is False

    def test_non_stripping_template(self) -> None:
        """A template that keeps reasoning everywhere strips nothing."""
        tokenizer = _stub(chat_template="uses <think> tags")

        info = detect_model(tokenizer=tokenizer)

        assert info.strips_think_tags is False
        assert info.strips_history_think_tags is False

    def test_round_trip_warning_for_broken_tokenizer(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A tokenizer that drops spaces triggers a warning."""
        tokenizer = _stub(
            chat_template="uses <think> tags",
            drops_spaces=True,
        )

        with caplog.at_level(logging.WARNING, logger="thinkpack"):
            detect_model(tokenizer=tokenizer)

        assert "does not round-trip" in caplog.text

    def test_no_round_trip_warning_for_working_tokenizer(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A working tokenizer produces no warning."""
        tokenizer = _stub(chat_template="uses <think> tags")

        with caplog.at_level(logging.WARNING, logger="thinkpack"):
            detect_model(tokenizer=tokenizer)

        assert caplog.text == ""


@pytest.mark.usefixtures("clear_detection_cache")
class TestResolveModelInfo:
    """_resolve_model_info() chooses between a custom ModelInfo and detection."""

    def test_custom_info_skips_detection(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A custom model_info is returned as-is, and the tokenizer is never inspected."""
        # the template has no known tag, so detection would log a warning
        tokenizer = _stub(chat_template="no reasoning tags here")
        custom = ModelInfo(
            prefixed=True,
            tag_content="REASONING",
            tag_style=TagStyle.BRACKET,
        )

        with caplog.at_level(logging.WARNING, logger="thinkpack"):
            result = model_module._resolve_model_info(
                tokenizer=tokenizer,
                model_info=custom,
                override_tag=None,
            )

        assert result == custom
        assert caplog.text == ""
        assert model_module._cache == {}

    def test_override_tag_applied_to_custom_info(self) -> None:
        """override_tag replaces the tag of a custom model_info, keeping other fields."""
        custom = ModelInfo(
            prefixed=True,
            strips_think_tags=True,
        )

        result = model_module._resolve_model_info(
            tokenizer=None,
            model_info=custom,
            override_tag="[THINK]",
        )

        assert result.open_tag == "[THINK]"
        assert result.prefixed is True
        assert result.strips_think_tags is True

    def test_detects_without_custom_info(self) -> None:
        """With no custom model_info, the format is detected from the tokenizer."""
        tokenizer = _stub(
            chat_template="uses <thinking> tags",
            gen_suffix="<thinking>",
        )

        result = model_module._resolve_model_info(
            tokenizer=tokenizer,
            model_info=None,
            override_tag=None,
        )

        assert result.tag_content == "thinking"
        assert result.prefixed is True

    def test_neither_given_raises(self) -> None:
        """Without a tokenizer or model_info there is nothing to work from."""
        with pytest.raises(ValueError, match="tokenizer or model_info"):
            model_module._resolve_model_info(
                tokenizer=None,
                model_info=None,
                override_tag=None,
            )


# ---------------------------------------------------------------------------
# reasoning stripping and tokenizer health — real tokenizers, marked slow
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.usefixtures("clear_detection_cache")
@pytest.mark.parametrize(
    ("fixture_name", "strips_final", "strips_history"),
    [
        ("qwen3_tokenizer", False, True),
        ("qwen35_tokenizer", False, True),
        ("deepseek_r1_llama_tokenizer", True, True),
        ("olmo3_tokenizer", False, False),
        # ministral's default system prompt contains [THINK], which must not fool detection
        ("ministral_tokenizer", False, False),
    ],
)
class TestDetectStripping:
    """strips_think_tags and strips_history_think_tags are detected per model."""

    def test_stripping_flags(
        self,
        request: pytest.FixtureRequest,
        fixture_name: str,
        strips_final: bool,
        strips_history: bool,
    ) -> None:
        """Detected flags match each template's known behaviour."""
        info = detect_model(tokenizer=request.getfixturevalue(fixture_name))

        assert info.strips_think_tags is strips_final
        assert info.strips_history_think_tags is strips_history

    def test_real_tokenizer_round_trips(
        self,
        request: pytest.FixtureRequest,
        fixture_name: str,
        strips_final: bool,
        strips_history: bool,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Supported tokenizers encode and decode plain text without a warning."""
        with caplog.at_level(logging.WARNING, logger="thinkpack"):
            detect_model(tokenizer=request.getfixturevalue(fixture_name))

        assert "does not round-trip" not in caplog.text
