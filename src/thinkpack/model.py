"""Model template style detection from tokenizer chat templates."""

import dataclasses
import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


_logger = logging.getLogger(__name__)


class _Tokenizer(Protocol):
    """Minimal protocol for a HuggingFace-compatible tokenizer."""

    chat_template: str | None

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

    # true if the template strips think tags from assistant message content when rendering
    strips_think_tags: bool = False

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


# matches a trailing opening tag (html or bracket) — indicates a prefixed template
_TRAILING_TAG = re.compile(r"(?:<[a-zA-Z][a-zA-Z0-9_]*>|\[[A-Z][A-Z0-9_]*\])\s*$")

# bracket-style is checked first as it is more distinctive than html tags
_REASONING_TAG_NAMES = ["think", "thinking", "thought", "reasoning"]

_KNOWN_TAGS: list[tuple[str, str, TagStyle]] = [
    *[
        (f"[{name.upper()}]", name.upper(), TagStyle.BRACKET)
        for name in _REASONING_TAG_NAMES
    ],
    *[(f"<{name}>", name, TagStyle.HTML) for name in _REASONING_TAG_NAMES],
]


# keyed on chat_template string, which fully determines detection
_cache: dict[str, ModelInfo] = {}


def detect_model(tokenizer: _Tokenizer) -> ModelInfo:
    """
    Detect how a tokenizer handles reasoning blocks from its chat template.

    Inspects the template in three steps (see inline comments): prefixed detection,
    tag name detection, and strips_think_tags detection. Results are cached on the
    template string so repeated calls are free.

    Returns a ModelInfo with the detected properties.
    """
    template = tokenizer.chat_template or ""
    if cached := _cache.get(template):
        return cached

    # step 1: detect prefixed by checking if the generation prompt ends with an
    # opening reasoning tag — both html and bracket forms are checked
    gen_prompt: str | list[int] = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if isinstance(gen_prompt, list):
        # some tokenizers return token ids despite tokenize=False
        gen_prompt = tokenizer.decode(gen_prompt)

    prefixed = bool(_TRAILING_TAG.search(gen_prompt))

    # step 2: detect the reasoning tag by scanning the template source string
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
            "defaulting to <think>. Use the tag= argument to override if needed."
        )

    # step 3: detect whether the template strips think tags from assistant content
    # by rendering a test message with think tags embedded and checking if they survive
    if tag_style == TagStyle.BRACKET:
        open_tag = f"[{tag_content}]"
        close_tag = f"[/{tag_content}]"
    else:
        open_tag = f"<{tag_content}>"
        close_tag = f"</{tag_content}>"

    test_content = f"{open_tag}\ntest reasoning\n{close_tag}\ntest response"
    test_rendered: str | list[int] = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": test_content},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    if isinstance(test_rendered, list):
        test_rendered = tokenizer.decode(test_rendered)
    strips_think_tags = open_tag not in test_rendered

    result = ModelInfo(
        prefixed=prefixed,
        strips_think_tags=strips_think_tags,
        tag_content=tag_content,
        tag_style=tag_style,
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
