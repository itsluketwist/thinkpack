"""Integration tests for apply_chat_template() using real tokenizers.

Tests cover Qwen3 (non-prefixed, <think>), OLMo-3 (prefixed, <think>), and
Ministral (non-prefixed, [THINK] bracket tags). Each test asserts the full
returned string by building the expected value from the tokenizer directly.

Skip with: pytest --no-slow
"""

import pytest

from thinkpack.chat import apply_chat_template, apply_chat_templates


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _base(tokenizer, messages, **kwargs) -> str:
    """Run the tokenizer template directly and return the raw string."""
    result = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    # some tokenizers return token ids despite tokenize=False
    if isinstance(result, list):
        result = tokenizer.decode(result)
    return result


def _embed(content: str, reasoning: str, open_tag: str, close_tag: str) -> str:
    """Build the assistant message content with reasoning tags prepended."""
    return f"{open_tag}\n{reasoning}\n{close_tag}\n{content}"


def _embed_blank(content: str, open_tag: str, close_tag: str) -> str:
    """Build the assistant message content with an empty reasoning block prepended."""
    return f"{open_tag}\n{close_tag}\n{content}"


# ---------------------------------------------------------------------------
# Qwen3 — non-prefixed, <think> tags
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestQwen3ChatTemplate:
    """apply_chat_template() tests for Qwen/Qwen3-8B — non-prefixed, <think> tags."""

    # --- single-turn generation prompt ---

    def test_default_passthrough(self, qwen3_tokenizer) -> None:
        """add_generation_reasoning=None (default): template output returned as-is, no tag added."""
        base = _base(qwen3_tokenizer, [{"role": "user", "content": "q"}])

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=qwen3_tokenizer,
        )

        assert result == base

    def test_think_prefix_seeded(self, qwen3_tokenizer) -> None:
        """think_prefix is injected after the opening tag."""
        base = _base(qwen3_tokenizer, [{"role": "user", "content": "q"}])
        expected = base.rstrip("\n") + "\n<think>\nOkay, "

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=qwen3_tokenizer,
            think_prefix="Okay, ",
        )

        assert result == expected

    def test_no_reasoning_no_tag(self, qwen3_tokenizer) -> None:
        """add_generation_reasoning=False: no tag added, prompt returned stripped."""
        base = _base(qwen3_tokenizer, [{"role": "user", "content": "q"}])
        expected = base.rstrip("\n")

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=qwen3_tokenizer,
            add_generation_reasoning=False,
        )

        assert result == expected

    def test_no_reasoning_with_response_prefix(self, qwen3_tokenizer) -> None:
        """add_generation_reasoning=False with response_prefix: no tag, response seeded directly."""
        base = _base(qwen3_tokenizer, [{"role": "user", "content": "q"}])
        expected = base.rstrip("\n") + "\nAnswer:"

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=qwen3_tokenizer,
            add_generation_reasoning=False,
            response_prefix="Answer:",
        )

        assert result == expected

    def test_passive_no_change(self, qwen3_tokenizer) -> None:
        """add_generation_reasoning=None: template output returned exactly as-is."""
        base = _base(qwen3_tokenizer, [{"role": "user", "content": "q"}])

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=qwen3_tokenizer,
            add_generation_reasoning=None,
        )

        assert result == base

    # --- multi-turn history ---

    def test_multi_turn_with_reasoning(self, qwen3_tokenizer) -> None:
        """Non-blank reasoning key embeds a complete <think>...</think> block in history."""
        # build expected by manually embedding the tags in the assistant turn
        modified = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": _embed("4", "two plus two is four", "<think>", "</think>"),
            },
            {"role": "user", "content": "And 3+3?"},
        ]
        expected = _base(qwen3_tokenizer, modified)

        result = apply_chat_template(
            conversation=[
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": "4",
                    "reasoning": "two plus two is four",
                },
                {"role": "user", "content": "And 3+3?"},
            ],
            tokenizer=qwen3_tokenizer,
        )

        assert result == expected

    def test_multi_turn_blank_reasoning(self, qwen3_tokenizer) -> None:
        """reasoning='' embeds an empty <think>\\n</think> block in history."""
        modified = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": _embed_blank("4", "<think>", "</think>")},
            {"role": "user", "content": "And 3+3?"},
        ]
        expected = _base(qwen3_tokenizer, modified)

        result = apply_chat_template(
            conversation=[
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4", "reasoning": ""},
                {"role": "user", "content": "And 3+3?"},
            ],
            tokenizer=qwen3_tokenizer,
        )

        assert result == expected

    def test_multi_turn_no_reasoning_key(self, qwen3_tokenizer) -> None:
        """Absent reasoning key passes the assistant turn through unchanged."""
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]
        expected = _base(qwen3_tokenizer, msgs)

        result = apply_chat_template(
            conversation=msgs,
            tokenizer=qwen3_tokenizer,
        )

        assert result == expected

    def test_batching(self, qwen3_tokenizer) -> None:
        """apply_chat_templates returns one correctly built string per conversation."""
        convs = [
            [{"role": "user", "content": "What is 1+1?"}],
            [{"role": "user", "content": "What is 2+2?"}],
        ]
        expected = [_base(qwen3_tokenizer, c) for c in convs]

        result = apply_chat_templates(conversations=convs, tokenizer=qwen3_tokenizer)

        assert result == expected


# ---------------------------------------------------------------------------
# OLMo-3 — prefixed, <think> tags
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestOlmo3ChatTemplate:
    """apply_chat_template() tests for allenai/OLMo-3-7B-Think — prefixed, <think> tags.

    The template injects <think> into the generation prompt, so base.rstrip("\\n")
    already ends with <think>.
    """

    # --- single-turn generation prompt ---

    def test_default_passthrough(self, olmo3_tokenizer) -> None:
        """add_generation_reasoning=None (default): template output returned as-is."""
        base = _base(olmo3_tokenizer, [{"role": "user", "content": "q"}])

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=olmo3_tokenizer,
        )

        assert result == base

    def test_think_prefix_seeded(self, olmo3_tokenizer) -> None:
        """think_prefix is injected after the template-provided open tag."""
        base = _base(olmo3_tokenizer, [{"role": "user", "content": "q"}])
        expected = base.rstrip("\n") + "\nOkay, "

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=olmo3_tokenizer,
            think_prefix="Okay, ",
        )

        assert result == expected

    def test_no_reasoning_strips_tag(self, olmo3_tokenizer) -> None:
        """add_generation_reasoning=False: template-injected <think> is stripped."""
        base = _base(olmo3_tokenizer, [{"role": "user", "content": "q"}])
        # strip the trailing <think> (7 chars)
        expected = base.rstrip("\n")[: -len("<think>")]

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=olmo3_tokenizer,
            add_generation_reasoning=False,
        )

        assert result == expected

    def test_no_reasoning_with_response_prefix(self, olmo3_tokenizer) -> None:
        """add_generation_reasoning=False with response_prefix: tag stripped then response seeded."""
        base = _base(olmo3_tokenizer, [{"role": "user", "content": "q"}])
        expected = base.rstrip("\n")[: -len("<think>")] + "\nAnswer:"

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=olmo3_tokenizer,
            add_generation_reasoning=False,
            response_prefix="Answer:",
        )

        assert result == expected

    def test_passive_keeps_original(self, olmo3_tokenizer) -> None:
        """add_generation_reasoning=None: original template string returned unchanged."""
        base = _base(olmo3_tokenizer, [{"role": "user", "content": "q"}])

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=olmo3_tokenizer,
            add_generation_reasoning=None,
        )

        assert result == base

    def test_none_with_response_prefix(self, olmo3_tokenizer) -> None:
        """add_generation_reasoning=None with response_prefix: tag closed then response seeded."""
        base = _base(olmo3_tokenizer, [{"role": "user", "content": "q"}])
        expected = base.rstrip("\n") + "\n</think>\nAnswer:"

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=olmo3_tokenizer,
            add_generation_reasoning=None,
            response_prefix="Answer:",
        )

        assert result == expected

    # --- multi-turn history ---

    def test_multi_turn_with_reasoning(self, olmo3_tokenizer) -> None:
        """Non-blank reasoning key embeds a complete <think>...</think> block in history."""
        modified = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": _embed("4", "two plus two is four", "<think>", "</think>"),
            },
            {"role": "user", "content": "And 3+3?"},
        ]
        expected = _base(olmo3_tokenizer, modified)

        result = apply_chat_template(
            conversation=[
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": "4",
                    "reasoning": "two plus two is four",
                },
                {"role": "user", "content": "And 3+3?"},
            ],
            tokenizer=olmo3_tokenizer,
        )

        assert result == expected

    def test_multi_turn_blank_reasoning(self, olmo3_tokenizer) -> None:
        """reasoning='' embeds an empty <think>\\n</think> block in history."""
        modified = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": _embed_blank("4", "<think>", "</think>")},
            {"role": "user", "content": "And 3+3?"},
        ]
        expected = _base(olmo3_tokenizer, modified)

        result = apply_chat_template(
            conversation=[
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4", "reasoning": ""},
                {"role": "user", "content": "And 3+3?"},
            ],
            tokenizer=olmo3_tokenizer,
        )

        assert result == expected

    def test_multi_turn_no_reasoning_key(self, olmo3_tokenizer) -> None:
        """Absent reasoning key passes the assistant turn through unchanged."""
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]
        expected = _base(olmo3_tokenizer, msgs)

        result = apply_chat_template(
            conversation=msgs,
            tokenizer=olmo3_tokenizer,
        )

        assert result == expected

    def test_batching(self, olmo3_tokenizer) -> None:
        """apply_chat_templates returns one correctly built string per conversation."""
        convs = [
            [{"role": "user", "content": "What is 1+1?"}],
            [{"role": "user", "content": "What is 2+2?"}],
        ]
        expected = [_base(olmo3_tokenizer, c) for c in convs]

        result = apply_chat_templates(conversations=convs, tokenizer=olmo3_tokenizer)

        assert result == expected


# ---------------------------------------------------------------------------
# Ministral — non-prefixed, [THINK] bracket tags
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMinistralChatTemplate:
    """apply_chat_template() tests for Ministral-3-3B-Reasoning-2512 — non-prefixed, [THINK] tags."""

    # --- single-turn generation prompt ---

    def test_default_passthrough(self, ministral_tokenizer) -> None:
        """add_generation_reasoning=None (default): template output returned as-is, no tag added."""
        base = _base(ministral_tokenizer, [{"role": "user", "content": "q"}])

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=ministral_tokenizer,
            override_tag="[THINK]",
        )

        assert result == base

    def test_think_prefix_seeded(self, ministral_tokenizer) -> None:
        """think_prefix is injected after the opening bracket tag."""
        base = _base(ministral_tokenizer, [{"role": "user", "content": "q"}])
        expected = base.rstrip("\n") + "\n[THINK]\nOkay, "

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=ministral_tokenizer,
            override_tag="[THINK]",
            think_prefix="Okay, ",
        )

        assert result == expected

    def test_no_reasoning_no_tag(self, ministral_tokenizer) -> None:
        """add_generation_reasoning=False: no tag added, prompt returned stripped."""
        base = _base(ministral_tokenizer, [{"role": "user", "content": "q"}])
        expected = base.rstrip("\n")

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=ministral_tokenizer,
            override_tag="[THINK]",
            add_generation_reasoning=False,
        )

        assert result == expected

    def test_no_reasoning_with_response_prefix(self, ministral_tokenizer) -> None:
        """add_generation_reasoning=False with response_prefix: no tag, response seeded directly."""
        base = _base(ministral_tokenizer, [{"role": "user", "content": "q"}])
        expected = base.rstrip("\n") + "\nAnswer:"

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=ministral_tokenizer,
            override_tag="[THINK]",
            add_generation_reasoning=False,
            response_prefix="Answer:",
        )

        assert result == expected

    def test_passive_no_change(self, ministral_tokenizer) -> None:
        """add_generation_reasoning=None: template output returned exactly as-is."""
        base = _base(ministral_tokenizer, [{"role": "user", "content": "q"}])

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=ministral_tokenizer,
            override_tag="[THINK]",
            add_generation_reasoning=None,
        )

        assert result == base

    # --- multi-turn history ---

    def test_multi_turn_with_reasoning(self, ministral_tokenizer) -> None:
        """Non-blank reasoning key embeds a complete [THINK]...[/THINK] block in history."""
        modified = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": _embed("4", "two plus two is four", "[THINK]", "[/THINK]"),
            },
            {"role": "user", "content": "And 3+3?"},
        ]
        expected = _base(ministral_tokenizer, modified)

        result = apply_chat_template(
            conversation=[
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": "4",
                    "reasoning": "two plus two is four",
                },
                {"role": "user", "content": "And 3+3?"},
            ],
            tokenizer=ministral_tokenizer,
            override_tag="[THINK]",
        )

        assert result == expected

    def test_multi_turn_blank_reasoning(self, ministral_tokenizer) -> None:
        """reasoning='' embeds an empty [THINK]\\n[/THINK] block in history."""
        modified = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": _embed_blank("4", "[THINK]", "[/THINK]")},
            {"role": "user", "content": "And 3+3?"},
        ]
        expected = _base(ministral_tokenizer, modified)

        result = apply_chat_template(
            conversation=[
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4", "reasoning": ""},
                {"role": "user", "content": "And 3+3?"},
            ],
            tokenizer=ministral_tokenizer,
            override_tag="[THINK]",
        )

        assert result == expected

    def test_multi_turn_no_reasoning_key(self, ministral_tokenizer) -> None:
        """Absent reasoning key passes the assistant turn through unchanged."""
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]
        expected = _base(ministral_tokenizer, msgs)

        result = apply_chat_template(
            conversation=msgs,
            tokenizer=ministral_tokenizer,
            override_tag="[THINK]",
        )

        assert result == expected

    def test_batching(self, ministral_tokenizer) -> None:
        """apply_chat_templates returns one correctly built string per conversation."""
        convs = [
            [{"role": "user", "content": "What is 1+1?"}],
            [{"role": "user", "content": "What is 2+2?"}],
        ]
        expected = [_base(ministral_tokenizer, c) for c in convs]

        result = apply_chat_templates(
            conversations=convs,
            tokenizer=ministral_tokenizer,
            override_tag="[THINK]",
        )

        assert result == expected


# ---------------------------------------------------------------------------
# DeepSeek-R1-Distill-Llama — prefixed, <think> tags, strips history think blocks
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDeepSeekR1LlamaChatTemplate:
    """apply_chat_template() tests for deepseek-ai/DeepSeek-R1-Distill-Llama-8B.

    Prefixed model (<think> tag is injected into the generation prompt by the
    template). The template also strips <think>...</think> blocks from assistant
    messages when rendering multi-turn history, so reasoning provided in history
    turns is silently removed from the rendered output.
    """

    # --- single-turn generation prompt ---

    def test_default_passthrough(self, deepseek_r1_llama_tokenizer) -> None:
        """add_generation_reasoning=None (default): template output returned as-is."""
        base = _base(deepseek_r1_llama_tokenizer, [{"role": "user", "content": "q"}])

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=deepseek_r1_llama_tokenizer,
        )

        assert result == base

    def test_think_prefix_seeded(self, deepseek_r1_llama_tokenizer) -> None:
        """think_prefix is injected after the template-provided open tag."""
        base = _base(deepseek_r1_llama_tokenizer, [{"role": "user", "content": "q"}])
        expected = base.rstrip("\n") + "\nOkay, "

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=deepseek_r1_llama_tokenizer,
            think_prefix="Okay, ",
        )

        assert result == expected

    def test_no_reasoning_strips_tag(self, deepseek_r1_llama_tokenizer) -> None:
        """add_generation_reasoning=False: template-injected <think> is stripped."""
        base = _base(deepseek_r1_llama_tokenizer, [{"role": "user", "content": "q"}])
        # remove the trailing <think> that the template appended
        expected = base.rstrip("\n")[: -len("<think>")]

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=deepseek_r1_llama_tokenizer,
            add_generation_reasoning=False,
        )

        assert result == expected

    def test_no_reasoning_with_response_prefix(
        self, deepseek_r1_llama_tokenizer
    ) -> None:
        """add_generation_reasoning=False with response_prefix: tag stripped then response seeded."""
        base = _base(deepseek_r1_llama_tokenizer, [{"role": "user", "content": "q"}])
        expected = base.rstrip("\n")[: -len("<think>")] + "\nAnswer:"

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=deepseek_r1_llama_tokenizer,
            add_generation_reasoning=False,
            response_prefix="Answer:",
        )

        assert result == expected

    def test_passive_keeps_original(self, deepseek_r1_llama_tokenizer) -> None:
        """add_generation_reasoning=None: original template string returned unchanged."""
        base = _base(deepseek_r1_llama_tokenizer, [{"role": "user", "content": "q"}])

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=deepseek_r1_llama_tokenizer,
            add_generation_reasoning=None,
        )

        assert result == base

    def test_none_with_response_prefix(self, deepseek_r1_llama_tokenizer) -> None:
        """add_generation_reasoning=None with response_prefix: tag closed then response seeded."""
        base = _base(deepseek_r1_llama_tokenizer, [{"role": "user", "content": "q"}])
        expected = base.rstrip("\n") + "\n</think>\nAnswer:"

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=deepseek_r1_llama_tokenizer,
            add_generation_reasoning=None,
            response_prefix="Answer:",
        )

        assert result == expected

    # --- multi-turn history ---

    def test_multi_turn_with_reasoning(self, deepseek_r1_llama_tokenizer) -> None:
        """Template strips the embedded <think>...</think> block from history; only content remains."""
        # the template removes <think>...</think> blocks from assistant messages, so the
        # expected output matches messages without any embedded reasoning
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]
        expected = _base(deepseek_r1_llama_tokenizer, msgs)

        result = apply_chat_template(
            conversation=[
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": "4",
                    "reasoning": "two plus two is four",
                },
                {"role": "user", "content": "And 3+3?"},
            ],
            tokenizer=deepseek_r1_llama_tokenizer,
        )

        assert result == expected

    def test_multi_turn_blank_reasoning(self, deepseek_r1_llama_tokenizer) -> None:
        """Template strips the empty <think>\\n</think> block from history too."""
        # same result as with reasoning — the template removes any think block regardless
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]
        expected = _base(deepseek_r1_llama_tokenizer, msgs)

        result = apply_chat_template(
            conversation=[
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4", "reasoning": ""},
                {"role": "user", "content": "And 3+3?"},
            ],
            tokenizer=deepseek_r1_llama_tokenizer,
        )

        assert result == expected

    def test_multi_turn_no_reasoning_key(self, deepseek_r1_llama_tokenizer) -> None:
        """Absent reasoning key passes the assistant turn through unchanged."""
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]
        expected = _base(deepseek_r1_llama_tokenizer, msgs)

        result = apply_chat_template(
            conversation=msgs,
            tokenizer=deepseek_r1_llama_tokenizer,
        )

        assert result == expected

    def test_batching(self, deepseek_r1_llama_tokenizer) -> None:
        """apply_chat_templates returns one correctly built string per conversation."""
        convs = [
            [{"role": "user", "content": "What is 1+1?"}],
            [{"role": "user", "content": "What is 2+2?"}],
        ]
        expected = [_base(deepseek_r1_llama_tokenizer, c) for c in convs]

        result = apply_chat_templates(
            conversations=convs,
            tokenizer=deepseek_r1_llama_tokenizer,
        )

        assert result == expected
