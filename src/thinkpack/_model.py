"""Model template style detection from tokenizer chat templates."""

import re
from dataclasses import dataclass
from enum import StrEnum


class TemplateStyle(StrEnum):
    """
    How a model's chat template handles reasoning blocks.

    INLINE   — standard: the model outputs <think>content</think> inline in its response.
               No special template support; tags are injected and parsed as plain text.
    NATIVE   — the template has a dedicated reasoning_content field (e.g. Qwen3).
               Reasoning is passed separately when building messages and rendered
               inside the think block by the template itself.
    PREFIXED — the template auto-injects an opening reasoning tag at the end of the
               generation prompt. The model's decoded output begins mid-reasoning
               (no opening tag visible), and always ends with a closing tag.
    """

    INLINE = "inline"
    NATIVE = "native"
    PREFIXED = "prefixed"


@dataclass
class ModelInfo:
    """Detected template style for a model's reasoning block handling.

    Detected once via detect_model() and used internally by mask() and steer()
    to handle model-specific formatting without exposing flags to the user.
    """

    style: TemplateStyle
    # the opening tag used by this model, e.g. "<think>", "<reasoning>", "<thought>"
    open_tag: str


# default opening tag used when the model has no known preference
_DEFAULT_OPEN_TAG = "<think>"

# sentinel injected into a test message to detect native reasoning_content support —
# if it appears in the rendered output, the template handles reasoning natively
_NATIVE_SENTINEL = "__thinkpack_detect__"

# matches any xml-like opening tag at the end of a string, e.g. <think>, <reasoning>
_TRAILING_TAG = re.compile(r"<([a-zA-Z][a-zA-Z0-9_]*)>\s*$")

# cache keyed on the chat_template string — the template fully determines detection,
# and is stable for the lifetime of any real tokenizer instance
_cache: dict[str, ModelInfo] = {}


def detect_model(tokenizer: object) -> ModelInfo:
    """
    Detect how a tokenizer handles reasoning blocks from its chat template.

    Checks for native reasoning_content support (NATIVE), a generation prompt
    that auto-injects an opening reasoning tag (PREFIXED), or neither (INLINE).
    Detection is fully behaviour-based — no template source scanning.

    Returns a ModelInfo with the detected TemplateStyle and open_tag.
    """
    template = getattr(tokenizer, "chat_template", "") or ""
    if cached := _cache.get(template):
        return cached

    # test for native reasoning_content support by rendering an assistant message
    # with a sentinel value — if the sentinel appears in output, the template
    # handles reasoning as a dedicated field rather than inline tags (e.g. Qwen3)
    try:
        out = tokenizer.apply_chat_template(  # type: ignore
            [
                {"role": "user", "content": ""},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": _NATIVE_SENTINEL,
                },
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        if isinstance(out, list):
            out = tokenizer.decode(out)  # type: ignore
        if _NATIVE_SENTINEL in out:
            # extract the actual tag the template wraps reasoning in, e.g. <think>
            tag_match = re.search(
                r"<([a-zA-Z][a-zA-Z0-9_]*)>[^<]*" + re.escape(_NATIVE_SENTINEL),
                out,
            )
            native_tag = f"<{tag_match.group(1)}>" if tag_match else _DEFAULT_OPEN_TAG
            result = ModelInfo(style=TemplateStyle.NATIVE, open_tag=native_tag)
            _cache[template] = result
            return result
    except Exception:
        pass  # template doesn't support this message structure — move on

    # apply with add_generation_prompt=True and check if any xml-like opening tag
    # was appended — if so, this is a PREFIXED model and we capture the tag name
    gen_prompt = tokenizer.apply_chat_template(  # type: ignore
        [{"role": "user", "content": ""}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if isinstance(gen_prompt, list):
        # some tokenizers return token ids despite tokenize=False — decode them
        gen_prompt = tokenizer.decode(gen_prompt)  # type: ignore

    match = _TRAILING_TAG.search(gen_prompt)
    if match:
        result = ModelInfo(
            style=TemplateStyle.PREFIXED,
            open_tag=f"<{match.group(1)}>",
        )
    else:
        result = ModelInfo(
            style=TemplateStyle.INLINE,
            open_tag=_DEFAULT_OPEN_TAG,
        )

    _cache[template] = result
    return result
