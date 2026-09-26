"""Model template style detection from tokenizer chat templates."""

import dataclasses
import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


_logger = logging.getLogger(__name__)


class _Tokenizer(Protocol):
    """Minimal protocol for a HuggingFace-compatible tokenizer."""

    chat_template: str | None

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = ...,
        return_offsets_mapping: bool = ...,
    ) -> Any: ...

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool = ...,
        **kwargs: object,
    ) -> str | list[int]: ...

    def encode(
        self,
        text: str,
        add_special_tokens: bool = ...,
        truncation: bool = ...,
        max_length: int = ...,
    ) -> list[int]: ...

    def decode(
        self,
        token_ids: list[int],
    ) -> str: ...


def _unwrap_tokenizer(tokenizer: _Tokenizer) -> _Tokenizer:
    """
    Return the text tokenizer, unwrapping it from a multimodal processor if needed.

    Multimodal models (e.g. Qwen3.5) are often loaded as a processor, for example by
    AutoProcessor or unsloth. A processor wraps the text tokenizer but has no encode()
    method, so thinkpack uses the inner tokenizer for everything.

    Returns the tokenizer to use for templating and tokenization.
    """
    # a processor has no encode() method but exposes the text tokenizer as .tokenizer
    inner = getattr(tokenizer, "tokenizer", None)
    if not hasattr(tokenizer, "encode") and inner is not None:
        return inner
    return tokenizer


class TagStyle(StrEnum):
    """
    The formatting style used to wrap reasoning block content.

    HTML    — standard xml-style tags: <think>...</think>. Used by most models.
    BRACKET — bracket-style tags: [THINK]...[/THINK]. Used by some models (e.g. Mistral).
    """

    HTML = "html"
    BRACKET = "bracket"


@dataclass
class ModelInfo:
    """Detected template properties for a model's reasoning block handling.

    Produced by detect_model() and consumed by chat() and mask() to handle
    model-specific formatting. Use with_tag() to produce a copy with an
    overridden tag without re-running detection.
    """

    # true if the template injects the opening reasoning tag into the generation prompt
    prefixed: bool

    # the name inside the reasoning tag, e.g. "think", "reasoning", "THINK"
    tag_content: str = "think"

    # controls whether tags are formatted as <tag>...</tag> or [tag]...[/tag]
    tag_style: TagStyle = TagStyle.HTML

    # true if the template strips the reasoning block from the final assistant message
    # (the one after the last user message) when rendering, e.g. DeepSeek-R1
    strips_think_tags: bool = False

    # true if the template strips the reasoning block from earlier assistant messages
    # (those before the last user message) when rendering, e.g. Qwen3 and DeepSeek-R1
    strips_history_think_tags: bool = False

    # reserved for future use — always None from automatic detection
    reasoning_key: str | None = None

    @property
    def open_tag(self) -> str:
        """The opening reasoning tag, e.g. <think> or [THINK]."""
        if self.tag_style == TagStyle.HTML:
            return f"<{self.tag_content}>"
        return f"[{self.tag_content}]"

    @property
    def close_tag(self) -> str:
        """The closing reasoning tag, e.g. </think> or [/THINK]."""
        if self.tag_style == TagStyle.HTML:
            return f"</{self.tag_content}>"
        return f"[/{self.tag_content}]"

    @property
    def tag_regex(self) -> tuple[re.Pattern[str], re.Pattern[str]]:
        """The compiled open/close regex patterns for this model's reasoning tags.

        Returns a (open_re, close_re) tuple with a capturing group around the tag name.
        """
        escaped = re.escape(self.tag_content)
        if self.tag_style == TagStyle.BRACKET:
            return (
                re.compile(rf"\[({escaped})\]", re.IGNORECASE),
                re.compile(rf"\[/({escaped})\]", re.IGNORECASE),
            )
        return (
            re.compile(rf"<({escaped})>", re.IGNORECASE),
            re.compile(rf"</({escaped})>", re.IGNORECASE),
        )

    def with_tag(
        self,
        tag: str,
    ) -> "ModelInfo":
        """
        Return a copy with tag content and style inferred from the tag string.

        Accepts a raw tag name ("think"), an HTML tag ("<think>"), or a bracket tag
        ("[THINK]"). Style is inferred from the format; a raw name keeps the existing style.

        Returns a new ModelInfo with all other fields unchanged.
        """
        if tag.startswith("<") and tag.endswith(">"):
            # html-style tag: extract name from angle brackets
            return dataclasses.replace(
                self,
                tag_content=tag[1:-1],
                tag_style=TagStyle.HTML,
            )
        if tag.startswith("[") and tag.endswith("]"):
            # bracket-style tag: extract name from square brackets
            return dataclasses.replace(
                self,
                tag_content=tag[1:-1],
                tag_style=TagStyle.BRACKET,
            )
        # raw name — keep the existing style
        return dataclasses.replace(self, tag_content=tag)


# bracket-style is checked first as it is more distinctive than html tags
_REASONING_TAG_NAMES = ["think", "thinking", "thought", "reasoning"]

_KNOWN_TAGS: list[tuple[str, str, TagStyle]] = [
    *[
        (f"[{name.upper()}]", name.upper(), TagStyle.BRACKET)
        for name in _REASONING_TAG_NAMES
    ],
    *[(f"<{name}>", name, TagStyle.HTML) for name in _REASONING_TAG_NAMES],
]

# unique text placed inside a test reasoning block during detection — searching for this
# (rather than the tag itself) avoids being fooled by tags in a default system prompt
_DETECTION_MARKER = "thinkpack-detection-marker"

# plain text used to check that the tokenizer can encode and decode without losing
# information (e.g. spaces), which catches broken tokenizer versions
_ROUND_TRIP_TEXT = "thinkpack tokenizer check: hello world"


# keyed on chat_template string, which fully determines detection
_cache: dict[str, ModelInfo] = {}


def _template_text(tokenizer: _Tokenizer) -> str:
    """
    Return the tokenizer's chat template source as a single string.

    The template may be a string, a dict of named templates (e.g. "default" and
    "tool_use"), or None. A dict is joined into one string so it can be searched for
    reasoning tags and used as a cache key.

    Returns the template source, or an empty string if there is no template.
    """
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        return ""
    if isinstance(template, dict):
        # join all named templates so tag scanning sees every variant
        return "\n".join(str(t) for t in template.values())
    return str(template)


def _render(
    tokenizer: _Tokenizer,
    conversation: list[dict[str, str]],
    add_generation_prompt: bool,
) -> str:
    """
    Render a conversation with the tokenizer's chat template, without tokenizing.

    Returns the rendered template string.
    """
    rendered: str | list[int] = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if isinstance(rendered, list):
        # some tokenizers return token ids despite tokenize=False
        rendered = tokenizer.decode(rendered)
    return rendered


def _check_round_trip(tokenizer: _Tokenizer) -> None:
    """
    Warn if the tokenizer cannot encode and decode plain text without changing it.

    Some transformers versions (5.3 to 5.12) load certain byte-level tokenizers, such
    as DeepSeek-R1-Distill-Llama, with the wrong pre-tokenizer. Spaces are then
    silently dropped, so every token id is wrong. This check only logs a warning.
    """
    token_ids = tokenizer.encode(_ROUND_TRIP_TEXT, add_special_tokens=False)
    decoded = tokenizer.decode(token_ids)
    if decoded.strip() != _ROUND_TRIP_TEXT:
        _logger.warning(
            "Tokenizer %s does not round-trip plain text (%r was decoded as %r), so its "
            "token ids are likely wrong. This is a known bug in transformers 5.3 to 5.12 "
            "for some byte-level models (e.g. DeepSeek-R1-Distill) — upgrade transformers. "
            "See https://github.com/huggingface/transformers/issues/45488.",
            type(tokenizer).__name__,
            _ROUND_TRIP_TEXT,
            decoded,
        )


def detect_model(tokenizer: _Tokenizer) -> ModelInfo:
    """
    Detect how a tokenizer handles reasoning blocks from its chat template.

    Inspects the template in four steps (see inline comments): tag name detection,
    prefixed detection, and whether reasoning is stripped from the final and earlier
    assistant messages. Also checks the tokenizer round-trips plain text. Results are
    cached on the template string so repeated calls are free.

    Returns a ModelInfo with the detected properties.
    """
    tokenizer = _unwrap_tokenizer(tokenizer)
    template = _template_text(tokenizer)
    if cached := _cache.get(template):
        return cached

    # step 1: detect the reasoning tag by scanning the template source string
    # for known tag patterns — defaults to <think> if nothing matches
    tag_content = "think"
    tag_style = TagStyle.HTML
    for literal, content, style in _KNOWN_TAGS:
        if literal in template:
            tag_content = content
            tag_style = style
            break
    else:
        _logger.warning(
            "No known reasoning tag found in the chat template — "
            "defaulting to <think>. Use the override_tag= argument to override if needed."
        )

    # provisional info, so the open and close tag strings come from one place
    tags = ModelInfo(
        prefixed=False,
        tag_content=tag_content,
        tag_style=tag_style,
    )

    # step 2: detect prefixed by checking if the generation prompt ends with the
    # opening reasoning tag (trailing whitespace such as "\n" is ignored)
    gen_prompt = _render(
        tokenizer=tokenizer,
        conversation=[{"role": "user", "content": "hello"}],
        add_generation_prompt=True,
    )
    prefixed = gen_prompt.rstrip().endswith(tags.open_tag)

    # a test assistant message whose reasoning block contains a unique marker
    test_assistant = {
        "role": "assistant",
        "content": (
            f"{tags.open_tag}\n{_DETECTION_MARKER}\n{tags.close_tag}\ntest response"
        ),
    }

    # step 3: detect whether the template strips reasoning from the final assistant
    # message, by rendering it and checking whether the marker survives
    final_rendered = _render(
        tokenizer=tokenizer,
        conversation=[
            {"role": "user", "content": "hello"},
            test_assistant,
        ],
        add_generation_prompt=False,
    )
    strips_think_tags = _DETECTION_MARKER not in final_rendered

    # step 4: detect whether the template strips reasoning from earlier assistant
    # messages, by adding a later user message so the test message becomes history
    history_rendered = _render(
        tokenizer=tokenizer,
        conversation=[
            {"role": "user", "content": "hello"},
            test_assistant,
            {"role": "user", "content": "hello again"},
        ],
        add_generation_prompt=False,
    )
    strips_history_think_tags = _DETECTION_MARKER not in history_rendered

    # finally, warn if the tokenizer itself is broken (not cached separately, as
    # detection only runs once per template)
    _check_round_trip(tokenizer=tokenizer)

    result = ModelInfo(
        prefixed=prefixed,
        tag_content=tag_content,
        tag_style=tag_style,
        strips_think_tags=strips_think_tags,
        strips_history_think_tags=strips_history_think_tags,
    )
    _cache[template] = result
    return result


def get_model_info(
    tokenizer: _Tokenizer,
    override_tag: str | None = None,
) -> ModelInfo:
    """
    Detect model properties and apply an optional tag override in one call.

    Wraps detect_model() and calls with_tag() if a tag is supplied, so callers
    do not need to repeat the override pattern themselves. The detected ModelInfo
    is cached; the tag override (if any) is applied after cache lookup and is not
    cached itself, since it is cheap and callers may pass different tags.

    tag may be a raw name ("reasoning"), an HTML tag ("<reasoning>"), or a bracket
    tag ("[REASONING]") — style is inferred from the format by with_tag().

    Returns a ModelInfo with the detected properties and any tag override applied.
    """
    model_info = detect_model(tokenizer=tokenizer)
    if override_tag is not None:
        model_info = model_info.with_tag(override_tag)
    return model_info
