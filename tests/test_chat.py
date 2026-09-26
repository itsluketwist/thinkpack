"""Integration tests for apply_chat_template() using real tokenizers.

Tests cover Qwen3 (non-prefixed, <think>), Qwen3.5 (prefixed, <think>), OLMo-3
(prefixed, <think>), Ministral (non-prefixed, [THINK] bracket tags), and
DeepSeek-R1-Distill-Llama (prefixed, strips think blocks). Most tests assert the full
returned string by building the expected value from the tokenizer directly; the
history reasoning tests check for the reasoning text itself, since the template's own
rendering cannot be used as the expected value when it strips reasoning.

Skip with: pytest --no-slow
"""

import pytest

from thinkpack.chat import apply_chat_template, apply_chat_templates
from thinkpack.model import ModelInfo


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
            add_generation_prompt=True,
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
            add_generation_prompt=True,
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
            add_generation_prompt=True,
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
            add_generation_prompt=True,
        )

        assert result == expected

    def test_passive_no_change(self, qwen3_tokenizer) -> None:
        """add_generation_reasoning=None: template output returned exactly as-is."""
        base = _base(qwen3_tokenizer, [{"role": "user", "content": "q"}])

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=qwen3_tokenizer,
            add_generation_reasoning=None,
            add_generation_prompt=True,
        )

        assert result == base

    def test_add_generation_reasoning_adds_tag_no_trailing_newline(
        self, qwen3_tokenizer
    ) -> None:
        """add_generation_reasoning=True: open tag appended without a trailing newline.

        A non-prefixed template doesn't inject <think>, so the tag must be added. The
        result must end with exactly '<think>' — no extra '\\n' after the tag — so that
        the model starts generating inside the block rather than on a blank line.
        """
        base = _base(qwen3_tokenizer, [{"role": "user", "content": "q"}])
        # expect the tag to be appended directly, with no trailing newline
        expected = base.rstrip("\n") + "\n<think>"

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=qwen3_tokenizer,
            add_generation_reasoning=True,
            add_generation_prompt=True,
        )

        assert result == expected
        assert not result.endswith("\n"), "open tag must not be followed by a newline"

    # --- multi-turn history ---

    def test_multi_turn_with_reasoning(self, qwen3_tokenizer) -> None:
        """add_history_reasoning=None: tags are embedded, and Qwen3's template drops them."""
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
            add_generation_prompt=True,
        )

        assert result == expected
        # the qwen3 template removes reasoning from history turns by default
        assert "two plus two is four" not in result

    def test_multi_turn_blank_reasoning(self, qwen3_tokenizer) -> None:
        """add_history_reasoning=None: an empty block is embedded and left to the template."""
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
            add_generation_prompt=True,
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
            add_generation_prompt=True,
        )

        assert result == expected

    def test_batching(self, qwen3_tokenizer) -> None:
        """apply_chat_templates returns one correctly built string per conversation."""
        convs = [
            [{"role": "user", "content": "What is 1+1?"}],
            [{"role": "user", "content": "What is 2+2?"}],
        ]
        expected = [_base(qwen3_tokenizer, c) for c in convs]

        result = apply_chat_templates(
            conversations=convs,
            tokenizer=qwen3_tokenizer,
            add_generation_prompt=True,
        )

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
            add_generation_prompt=True,
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
            add_generation_prompt=True,
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
            add_generation_prompt=True,
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
            add_generation_prompt=True,
        )

        assert result == expected

    def test_passive_keeps_original(self, olmo3_tokenizer) -> None:
        """add_generation_reasoning=None: original template string returned unchanged."""
        base = _base(olmo3_tokenizer, [{"role": "user", "content": "q"}])

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=olmo3_tokenizer,
            add_generation_reasoning=None,
            add_generation_prompt=True,
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
            add_generation_prompt=True,
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
            add_generation_prompt=True,
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
            add_generation_prompt=True,
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
            add_generation_prompt=True,
        )

        assert result == expected

    def test_batching(self, olmo3_tokenizer) -> None:
        """apply_chat_templates returns one correctly built string per conversation."""
        convs = [
            [{"role": "user", "content": "What is 1+1?"}],
            [{"role": "user", "content": "What is 2+2?"}],
        ]
        expected = [_base(olmo3_tokenizer, c) for c in convs]

        result = apply_chat_templates(
            conversations=convs,
            tokenizer=olmo3_tokenizer,
            add_generation_prompt=True,
        )

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
            add_generation_prompt=True,
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
            add_generation_prompt=True,
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
            add_generation_prompt=True,
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
            add_generation_prompt=True,
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
            add_generation_prompt=True,
        )

        assert result == base

    def test_add_generation_reasoning_adds_tag_no_trailing_newline(
        self, ministral_tokenizer
    ) -> None:
        """add_generation_reasoning=True: open tag appended without a trailing newline.

        Same contract as the Qwen3 variant but exercised with bracket-style tags to
        confirm the behaviour is tag-format-agnostic.
        """
        base = _base(ministral_tokenizer, [{"role": "user", "content": "q"}])
        expected = base.rstrip("\n") + "\n[THINK]"

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=ministral_tokenizer,
            override_tag="[THINK]",
            add_generation_reasoning=True,
            add_generation_prompt=True,
        )

        assert result == expected
        assert not result.endswith("\n"), "open tag must not be followed by a newline"

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
            add_generation_prompt=True,
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
            add_generation_prompt=True,
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
            add_generation_prompt=True,
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
            add_generation_prompt=True,
        )

        assert result == expected


# ---------------------------------------------------------------------------
# DeepSeek-R1-Distill-Llama — prefixed, <think> tags, strips history think blocks
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDeepSeekR1LlamaChatTemplate:
    """apply_chat_template() tests for deepseek-ai/DeepSeek-R1-Distill-Llama-8B.

    Prefixed model (<think> tag is injected into the generation prompt by the
    template). The template strips <think>...</think> blocks from assistant
    messages when rendering multi-turn history. With add_history_reasoning=True the
    library handles this via a sentinel placeholder: template rendering uses the
    sentinel, then the full think+content block is re-injected post-rendering.
    """

    # --- single-turn generation prompt ---

    def test_default_passthrough(self, deepseek_r1_llama_tokenizer) -> None:
        """add_generation_reasoning=None (default): template output returned as-is."""
        base = _base(deepseek_r1_llama_tokenizer, [{"role": "user", "content": "q"}])

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=deepseek_r1_llama_tokenizer,
            add_generation_prompt=True,
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
            add_generation_prompt=True,
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
            add_generation_prompt=True,
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
            add_generation_prompt=True,
        )

        assert result == expected

    def test_passive_keeps_original(self, deepseek_r1_llama_tokenizer) -> None:
        """add_generation_reasoning=None: original template string returned unchanged."""
        base = _base(deepseek_r1_llama_tokenizer, [{"role": "user", "content": "q"}])

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=deepseek_r1_llama_tokenizer,
            add_generation_reasoning=None,
            add_generation_prompt=True,
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
            add_generation_prompt=True,
        )

        assert result == expected

    # --- multi-turn history ---

    def test_multi_turn_with_reasoning(self, deepseek_r1_llama_tokenizer) -> None:
        """add_history_reasoning=True re-injects reasoning after template rendering."""
        # the template strips think blocks from history, but our sentinel approach ensures
        # the think+content block is re-injected; compute expected by rendering with a sentinel
        # and replacing it with the full embedded block
        sentinel = "ASSISTANT_CONTENT_SENTINEL"
        raw = _base(
            deepseek_r1_llama_tokenizer,
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": sentinel},
                {"role": "user", "content": "And 3+3?"},
            ],
        )
        expected = raw.replace(
            sentinel, _embed("4", "two plus two is four", "<think>", "</think>")
        )

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
            add_history_reasoning=True,
            add_generation_prompt=True,
        )

        assert result == expected

    def test_multi_turn_blank_reasoning(self, deepseek_r1_llama_tokenizer) -> None:
        """add_history_reasoning=True re-injects an empty block into history."""
        sentinel = "ASSISTANT_CONTENT_SENTINEL"
        raw = _base(
            deepseek_r1_llama_tokenizer,
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": sentinel},
                {"role": "user", "content": "And 3+3?"},
            ],
        )
        expected = raw.replace(sentinel, _embed_blank("4", "<think>", "</think>"))

        result = apply_chat_template(
            conversation=[
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4", "reasoning": ""},
                {"role": "user", "content": "And 3+3?"},
            ],
            tokenizer=deepseek_r1_llama_tokenizer,
            add_history_reasoning=True,
            add_generation_prompt=True,
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
            add_generation_prompt=True,
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
            add_generation_prompt=True,
        )

        assert result == expected


# ---------------------------------------------------------------------------
# Qwen3.5 — prefixed, <think> tags, trims content
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestQwen35ChatTemplate:
    """apply_chat_template() tests for Qwen/Qwen3.5-9B — prefixed, <think> tags."""

    def test_default_passthrough(self, qwen35_tokenizer) -> None:
        """add_generation_reasoning=None (default): template output returned as-is."""
        base = _base(qwen35_tokenizer, [{"role": "user", "content": "q"}])

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=qwen35_tokenizer,
            add_generation_prompt=True,
        )

        assert result == base
        # qwen3.5 is prefixed, so the template itself opens the reasoning block
        assert result.rstrip("\n").endswith("<think>")

    def test_think_prefix_seeded(self, qwen35_tokenizer) -> None:
        """think_prefix is injected after the template-provided open tag."""
        base = _base(qwen35_tokenizer, [{"role": "user", "content": "q"}])
        expected = base.rstrip("\n") + "\nOkay, "

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=qwen35_tokenizer,
            think_prefix="Okay, ",
            add_generation_prompt=True,
        )

        assert result == expected

    def test_no_reasoning_strips_tag(self, qwen35_tokenizer) -> None:
        """add_generation_reasoning=False: template-injected <think> is stripped."""
        base = _base(qwen35_tokenizer, [{"role": "user", "content": "q"}])
        expected = base.rstrip("\n")[: -len("<think>")]

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=qwen35_tokenizer,
            add_generation_reasoning=False,
            add_generation_prompt=True,
        )

        assert result == expected

    def test_none_with_response_prefix(self, qwen35_tokenizer) -> None:
        """add_generation_reasoning=None with response_prefix: tag closed then response seeded."""
        base = _base(qwen35_tokenizer, [{"role": "user", "content": "q"}])
        expected = base.rstrip("\n") + "\n</think>\nAnswer:"

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=qwen35_tokenizer,
            response_prefix="Answer:",
            add_generation_prompt=True,
        )

        assert result == expected

    def test_enable_thinking_false_forwarded(self, qwen35_tokenizer) -> None:
        """Template kwargs such as enable_thinking=False are forwarded unchanged."""
        base = _base(
            qwen35_tokenizer,
            [{"role": "user", "content": "q"}],
            enable_thinking=False,
        )

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=qwen35_tokenizer,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        assert result == base
        # non-thinking mode closes an empty reasoning block in the prompt
        assert result.endswith("<think>\n\n</think>\n\n")

    def test_batching(self, qwen35_tokenizer) -> None:
        """apply_chat_templates returns one correctly built string per conversation."""
        convs = [
            [{"role": "user", "content": "What is 1+1?"}],
            [{"role": "user", "content": "What is 2+2?"}],
        ]
        expected = [_base(qwen35_tokenizer, c) for c in convs]

        result = apply_chat_templates(
            conversations=convs,
            tokenizer=qwen35_tokenizer,
            add_generation_prompt=True,
        )

        assert result == expected


# ---------------------------------------------------------------------------
# add_history_reasoning — reasoning on assistant messages before the last user message
# ---------------------------------------------------------------------------


# fixture name, and whether the template keeps history reasoning by default
_HISTORY_MODELS = [
    ("qwen3_tokenizer", False),
    ("qwen35_tokenizer", False),
    ("deepseek_r1_llama_tokenizer", False),
    ("olmo3_tokenizer", True),
    ("ministral_tokenizer", True),
]

_HISTORY_REASONING = "the history reasoning text"
_FINAL_REASONING = "the final reasoning text"


def _history_conversation() -> list[dict[str, str]]:
    """A conversation whose only assistant message is history (a user message follows)."""
    return [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4", "reasoning": _HISTORY_REASONING},
        {"role": "user", "content": "And 3+3?"},
    ]


@pytest.mark.slow
@pytest.mark.parametrize(("fixture_name", "keeps_by_default"), _HISTORY_MODELS)
class TestHistoryReasoning:
    """add_history_reasoning forces history reasoning in (True), out (False), or defers (None)."""

    def test_true_always_keeps_reasoning_once(
        self,
        request: pytest.FixtureRequest,
        fixture_name: str,
        keeps_by_default: bool,
    ) -> None:
        """True: the reasoning appears exactly once, even if the template would strip it."""
        tokenizer = request.getfixturevalue(fixture_name)

        result = apply_chat_template(
            conversation=_history_conversation(),
            tokenizer=tokenizer,
            add_history_reasoning=True,
            add_generation_prompt=True,
        )

        assert result.count(_HISTORY_REASONING) == 1

    def test_false_always_drops_reasoning(
        self,
        request: pytest.FixtureRequest,
        fixture_name: str,
        keeps_by_default: bool,
    ) -> None:
        """False: the reasoning never appears, even if the template would keep it."""
        tokenizer = request.getfixturevalue(fixture_name)

        result = apply_chat_template(
            conversation=_history_conversation(),
            tokenizer=tokenizer,
            add_history_reasoning=False,
            add_generation_prompt=True,
        )

        assert _HISTORY_REASONING not in result
        # the assistant content itself is still present
        assert "4" in result

    def test_none_defers_to_template(
        self,
        request: pytest.FixtureRequest,
        fixture_name: str,
        keeps_by_default: bool,
    ) -> None:
        """None (default): the template decides whether history reasoning is kept."""
        tokenizer = request.getfixturevalue(fixture_name)

        result = apply_chat_template(
            conversation=_history_conversation(),
            tokenizer=tokenizer,
            add_generation_prompt=True,
        )

        assert (_HISTORY_REASONING in result) is keeps_by_default

    @pytest.mark.parametrize("add_history_reasoning", [True, False, None])
    def test_final_turn_reasoning_always_kept(
        self,
        request: pytest.FixtureRequest,
        fixture_name: str,
        keeps_by_default: bool,
        add_history_reasoning: bool | None,
    ) -> None:
        """Reasoning on the final assistant turn is kept exactly once for every setting."""
        tokenizer = request.getfixturevalue(fixture_name)

        result = apply_chat_template(
            conversation=[
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a", "reasoning": _FINAL_REASONING},
            ],
            tokenizer=tokenizer,
            add_history_reasoning=add_history_reasoning,
            add_generation_prompt=False,
        )

        assert result.count(_FINAL_REASONING) == 1


# ---------------------------------------------------------------------------
# custom ModelInfo — detection is skipped and the given format is used
# ---------------------------------------------------------------------------


# a custom format for Qwen3, using <reasoning> tags instead of the detected <think>
_CUSTOM_INFO = ModelInfo(
    prefixed=False,
    tag_content="reasoning",
)


@pytest.mark.slow
class TestCustomModelInfo:
    """apply_chat_template() with a custom ModelInfo passed instead of detection."""

    def test_custom_tag_used(self, qwen3_tokenizer) -> None:
        """The custom tag opens the reasoning block, not the detected <think> tag."""
        base = _base(qwen3_tokenizer, [{"role": "user", "content": "q"}])
        expected = base.rstrip("\n") + "\n<reasoning>\nOkay, "

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=qwen3_tokenizer,
            think_prefix="Okay, ",
            add_generation_prompt=True,
            model_info=_CUSTOM_INFO,
        )

        assert result == expected

    def test_override_tag_applied_on_top(self, qwen3_tokenizer) -> None:
        """override_tag replaces the tag of the custom model_info."""
        base = _base(qwen3_tokenizer, [{"role": "user", "content": "q"}])
        expected = base.rstrip("\n") + "\n[THINK]\nOkay, "

        result = apply_chat_template(
            conversation=[{"role": "user", "content": "q"}],
            tokenizer=qwen3_tokenizer,
            think_prefix="Okay, ",
            override_tag="[THINK]",
            add_generation_prompt=True,
            model_info=_CUSTOM_INFO,
        )

        assert result == expected

    def test_custom_stripping_flags_used(self, qwen3_tokenizer) -> None:
        """The custom stripping flags are trusted over what detection would find.

        Qwen3 strips history reasoning, so detection would protect it when
        add_history_reasoning=True. A custom model_info saying it does not strip is
        used as given, so the template strips the reasoning.
        """
        result = apply_chat_template(
            conversation=_history_conversation(),
            tokenizer=qwen3_tokenizer,
            add_history_reasoning=True,
            add_generation_prompt=True,
            model_info=ModelInfo(
                prefixed=False,
                strips_history_think_tags=False,
            ),
        )

        assert _HISTORY_REASONING not in result

    def test_batching(self, qwen3_tokenizer) -> None:
        """apply_chat_templates passes the custom model_info to every conversation."""
        convs = [
            [{"role": "user", "content": "What is 1+1?"}],
            [{"role": "user", "content": "What is 2+2?"}],
        ]
        expected = [
            _base(qwen3_tokenizer, c).rstrip("\n") + "\n<reasoning>\nOkay, "
            for c in convs
        ]

        result = apply_chat_templates(
            conversations=convs,
            tokenizer=qwen3_tokenizer,
            think_prefix="Okay, ",
            add_generation_prompt=True,
            model_info=_CUSTOM_INFO,
        )

        assert result == expected
