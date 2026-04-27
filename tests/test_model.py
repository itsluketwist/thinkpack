"""Tests for ModelInfo, detect_model(), and get_model_info() in thinkpack.model."""

import pytest

from thinkpack.model import ModelInfo, TagStyle, detect_model, get_model_info


# ---------------------------------------------------------------------------
# ModelInfo unit tests — no tokenizer required, always fast
# ---------------------------------------------------------------------------


class TestModelInfoDefaults:
    """ModelInfo defaults to HTML think tags with no reasoning key."""

    def test_defaults(self) -> None:
        info = ModelInfo(prefixed=False)

        assert info.prefixed is False
        assert info.tag_content == "think"
        assert info.tag_style == TagStyle.HTML
        assert info.reasoning_key is None
        assert info.open_tag == "<think>"
        assert info.close_tag == "</think>"


class TestModelInfoTags:
    """open_tag and close_tag produce correctly formatted strings."""

    def test_html_tags(self) -> None:
        info = ModelInfo(prefixed=False, tag_content="reasoning")

        assert info.prefixed is False
        assert info.tag_content == "reasoning"
        assert info.tag_style == TagStyle.HTML
        assert info.reasoning_key is None
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
        assert info.reasoning_key is None
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
        assert html_result.reasoning_key is None

        assert bracket_result.prefixed is False
        assert bracket_result.tag_content == "REASONING"
        assert bracket_result.tag_style == TagStyle.BRACKET
        assert bracket_result.reasoning_key is None

    def test_html_formatted_tag(self) -> None:
        """<tag> format sets HTML style regardless of the original."""
        result = ModelInfo(prefixed=False, tag_style=TagStyle.BRACKET).with_tag(
            "<reasoning>"
        )

        assert result.prefixed is False
        assert result.tag_content == "reasoning"
        assert result.tag_style == TagStyle.HTML
        assert result.reasoning_key is None

    def test_bracket_formatted_tag(self) -> None:
        """[TAG] format sets BRACKET style regardless of the original."""
        result = ModelInfo(prefixed=False).with_tag("[THINK]")

        assert result.prefixed is False
        assert result.tag_content == "THINK"
        assert result.tag_style == TagStyle.BRACKET
        assert result.reasoning_key is None

    def test_other_fields_preserved_and_original_unchanged(self) -> None:
        """with_tag() preserves prefixed and reasoning_key; the original is not modified."""
        info = ModelInfo(
            prefixed=True,
            reasoning_key="reasoning_content",
            tag_content="think",
        )
        result = info.with_tag("reasoning")

        assert result.prefixed is True
        assert result.tag_content == "reasoning"
        assert result.tag_style == TagStyle.HTML
        assert result.reasoning_key == "reasoning_content"
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
        assert info.reasoning_key is None

    def test_qwen35_detected_values(self, qwen35_tokenizer) -> None:
        """Qwen3.5 is prefixed and uses <think> tags."""
        info = get_model_info(tokenizer=qwen35_tokenizer)

        assert info.prefixed is True
        assert info.tag_content == "think"
        assert info.tag_style == TagStyle.HTML
        assert info.reasoning_key is None

    def test_deepseek_r1_llama_detected_values(
        self,
        deepseek_r1_llama_tokenizer,
    ) -> None:
        """DeepSeek-R1-Llama is prefixed and uses <think> tags."""
        info = get_model_info(tokenizer=deepseek_r1_llama_tokenizer)

        assert info.prefixed is True
        assert info.tag_content == "think"
        assert info.tag_style == TagStyle.HTML
        assert info.reasoning_key is None

    def test_olmo3_detected_values(self, olmo3_tokenizer) -> None:
        """OLMo-3 is prefixed and uses <think> tags."""
        info = get_model_info(tokenizer=olmo3_tokenizer)

        assert info.prefixed is True
        assert info.tag_content == "think"
        assert info.tag_style == TagStyle.HTML
        assert info.reasoning_key is None

    def test_ministral_detected_values(self, ministral_tokenizer) -> None:
        """Ministral is not prefixed and uses bracket [THINK] tags, auto-detected."""
        info = detect_model(tokenizer=ministral_tokenizer)

        assert info.prefixed is False
        assert info.tag_content == "THINK"
        assert info.tag_style == TagStyle.BRACKET
        assert info.reasoning_key is None

    def test_raw_tag_override_qwen3(self, qwen3_tokenizer) -> None:
        """Raw tag override on an inline model — tag_content changes, other fields unchanged."""
        result = get_model_info(tokenizer=qwen3_tokenizer, tag="reasoning")

        assert result.prefixed is False
        assert result.tag_content == "reasoning"
        assert result.tag_style == TagStyle.HTML
        assert result.reasoning_key is None

    def test_raw_tag_override_deepseek(self, deepseek_r1_llama_tokenizer) -> None:
        """Raw tag override on a prefixed model — prefixed=True is preserved."""
        result = get_model_info(tokenizer=deepseek_r1_llama_tokenizer, tag="reasoning")

        assert result.prefixed is True
        assert result.tag_content == "reasoning"
        assert result.tag_style == TagStyle.HTML
        assert result.reasoning_key is None

    def test_bracket_tag_override_qwen3(self, qwen3_tokenizer) -> None:
        """[TAG] override switches an HTML model to BRACKET style."""
        result = get_model_info(tokenizer=qwen3_tokenizer, tag="[THINK]")

        assert result.prefixed is False
        assert result.tag_content == "THINK"
        assert result.tag_style == TagStyle.BRACKET
        assert result.reasoning_key is None

    def test_bracket_tag_override_olmo3(self, olmo3_tokenizer) -> None:
        """[TAG] override switches a prefixed HTML model to BRACKET style."""
        result = get_model_info(tokenizer=olmo3_tokenizer, tag="[THINK]")

        assert result.prefixed is True
        assert result.tag_content == "THINK"
        assert result.tag_style == TagStyle.BRACKET
        assert result.reasoning_key is None

    def test_html_tag_override_ministral(self, ministral_tokenizer) -> None:
        """<tag> override switches a BRACKET model to HTML style."""
        result = get_model_info(tokenizer=ministral_tokenizer, tag="<think>")

        assert result.prefixed is False
        assert result.tag_content == "think"
        assert result.tag_style == TagStyle.HTML
        assert result.reasoning_key is None

    def test_tag_override_does_not_corrupt_cache(self, qwen3_tokenizer) -> None:
        """Tag overrides are not stored in the cache — detected values remain intact."""
        _ = get_model_info(tokenizer=qwen3_tokenizer, tag="reasoning")
        result = get_model_info(tokenizer=qwen3_tokenizer)

        assert result.prefixed is False
        assert result.tag_content == "think"
        assert result.tag_style == TagStyle.HTML
        assert result.reasoning_key is None
