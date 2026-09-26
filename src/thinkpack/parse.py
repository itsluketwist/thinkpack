"""Parsing of model responses into reasoning and answer components."""

import dataclasses
from dataclasses import dataclass
from typing import cast, overload

from thinkpack.model import (
    ModelInfo,
    _resolve_model_info,
    _Tokenizer,
    _unwrap_tokenizer,
)


@dataclass
class ParsedResponse:
    """A model response split into reasoning and answer components.

    Exactly one of has_valid_reasoning, has_empty_reasoning, has_truncated_reasoning,
    and has_missing_reasoning is true for each response.
    """

    # text after the closing reasoning tag, or the full response if no block was found
    answer: str

    # content inside the reasoning block; empty string if no block was present
    reasoning: str

    # the reasoning tag name, e.g. "think" (None if no reasoning block was found)
    reasoning_tag: str | None

    # true if the reasoning block was completed and non-blank
    has_valid_reasoning: bool

    # true if an opening tag was found but no closing tag followed
    has_truncated_reasoning: bool

    # true if a reasoning block was opened and closed but its content was blank
    has_empty_reasoning: bool

    # true if no reasoning block structure could be found in the response
    has_missing_reasoning: bool

    # token count of the reasoning content; None unless parsed with calculate_tokens=True
    reasoning_token_count: int | None = None

    # token count of the answer content; None unless parsed with calculate_tokens=True
    answer_token_count: int | None = None

    @property
    def has_invalid_reasoning(self) -> bool:
        """True if reasoning is missing, truncated, or empty."""
        return not self.has_valid_reasoning

    @property
    def has_answer(self) -> bool:
        """True if the answer is non-empty and non-whitespace."""
        return bool(self.answer.strip())


def _prompt_leaves_block_open(
    prompt: str,
    model_info: ModelInfo,
) -> bool:
    """
    Check whether a generation prompt ends inside an open reasoning block.

    This is when the last opening tag comes after the last closing tag, e.g. a prompt
    ending "<think>\n" or "<think>\nLet me think". A prompt with no tags, or ending
    with a closed block such as "<think>\n\n</think>" (Qwen3 with enable_thinking=False),
    does not. Only the last tags are compared, so tags earlier in the prompt (e.g. in a
    system prompt) do not matter.

    Returns True if the model's output will continue inside the reasoning block.
    """
    open_re, close_re = model_info.tag_regex
    open_matches = list(open_re.finditer(prompt))
    close_matches = list(close_re.finditer(prompt))
    if not open_matches:
        return False
    if not close_matches:
        return True
    return open_matches[-1].start() > close_matches[-1].start()


def _parse_single(
    response: str,
    model_info: ModelInfo,
    tokenizer: _Tokenizer | None,
    calculate_tokens: bool,
) -> ParsedResponse:
    """Parse one response string, using an already-detected model_info.

    Returns the ParsedResponse.
    """
    prefixed = model_info.prefixed
    open_re, close_re = model_info.tag_regex
    close_match = close_re.search(response)

    if close_match:
        # a closing tag was found, so the reasoning block is complete — the reasoning
        # is everything before it, minus the opening tag if the output has one
        before_close = response[: close_match.start()]
        reasoning = open_re.sub("", before_close, count=1).strip()
        answer = response[close_match.end() :].strip()
        has_valid_reasoning = bool(reasoning)
        result = ParsedResponse(
            answer=answer,
            reasoning=reasoning,
            reasoning_tag=model_info.tag_content,
            has_valid_reasoning=has_valid_reasoning,
            has_truncated_reasoning=False,
            has_empty_reasoning=not has_valid_reasoning,
            has_missing_reasoning=False,
        )

    elif open_match := open_re.search(response):
        # model started reasoning but output was cut off before the close tag
        result = ParsedResponse(
            answer="",
            reasoning=response[open_match.end() :].strip(),
            reasoning_tag=model_info.tag_content,
            has_valid_reasoning=False,
            has_truncated_reasoning=True,
            has_empty_reasoning=False,
            has_missing_reasoning=False,
        )

    elif prefixed:
        # the prompt opened the reasoning block, so the output starts inside it — with
        # no closing tag, the reasoning was never finished
        result = ParsedResponse(
            answer="",
            reasoning=response.strip(),
            reasoning_tag=model_info.tag_content,
            has_valid_reasoning=False,
            has_truncated_reasoning=True,
            has_empty_reasoning=False,
            has_missing_reasoning=False,
        )

    else:
        # no reasoning tags at all — plain response with no think block
        result = ParsedResponse(
            answer=response,
            reasoning="",
            reasoning_tag=None,
            has_valid_reasoning=False,
            has_truncated_reasoning=False,
            has_empty_reasoning=False,
            has_missing_reasoning=True,
        )

    # populate token counts if requested and a tokenizer is available
    if calculate_tokens and tokenizer is not None:
        result = dataclasses.replace(
            result,
            reasoning_token_count=len(
                tokenizer.encode(result.reasoning, add_special_tokens=False)
            ),
            answer_token_count=len(
                tokenizer.encode(result.answer, add_special_tokens=False)
            ),
        )

    return result


@overload
def parse(
    response: str,
    tokenizer: _Tokenizer | None = ...,
    override_tag: str | None = ...,
    model_info: ModelInfo | None = ...,
    calculate_tokens: bool = ...,
    prompt: str | list[str] | None = ...,
    add_generation_reasoning: bool | None = ...,
) -> ParsedResponse: ...


@overload
def parse(
    response: list[str],
    tokenizer: _Tokenizer | None = ...,
    override_tag: str | None = ...,
    model_info: ModelInfo | None = ...,
    calculate_tokens: bool = ...,
    prompt: str | list[str] | None = ...,
    add_generation_reasoning: bool | None = ...,
) -> list[ParsedResponse]: ...


@overload
def parse(
    response: list[list[str]],
    tokenizer: _Tokenizer | None = ...,
    override_tag: str | None = ...,
    model_info: ModelInfo | None = ...,
    calculate_tokens: bool = ...,
    prompt: str | list[str] | None = ...,
    add_generation_reasoning: bool | None = ...,
) -> list[list[ParsedResponse]]: ...


def parse(
    response: str | list[str] | list[list[str]],
    tokenizer: _Tokenizer | None = None,
    override_tag: str | None = None,
    model_info: ModelInfo | None = None,
    calculate_tokens: bool = False,
    prompt: str | list[str] | None = None,
    add_generation_reasoning: bool | None = None,
) -> ParsedResponse | list[ParsedResponse] | list[list[ParsedResponse]]:
    """Parse one or more model responses into reasoning and answer components.

    Accepts a single string, a flat list of strings, or a nested [task][sample] list.
    Handles standard (<think>content</think>answer), prefixed (content</think>answer),
    truncated standard (<think>content...), and truncated prefixed (content...) formats.

    Pass a tokenizer to detect the model's reasoning format, or pass model_info directly
    to skip detection (model_info wins if both are given, and the tokenizer is then only
    used for token counts). override_tag replaces the reasoning tag, e.g. "<reasoning>".

    Pass the generation prompt(s) as prompt, so parse() knows whether the output starts
    inside an open reasoning block (e.g. the prompt ends with "<think>") or not (e.g. no
    tags, or a closed block from enable_thinking=False). Only the first prompt is
    checked, so all prompts should be built the same way. Alternatively, set
    add_generation_reasoning to True or False to state this directly, as it was passed
    to apply_chat_template().

    Token counts are added when calculate_tokens=True and a tokenizer is given.

    Returns a ParsedResponse, list[ParsedResponse], or list[list[ParsedResponse]]
    matching the shape of the input.
    """
    # multimodal processors wrap the text tokenizer — use the tokenizer directly
    if tokenizer is not None:
        tokenizer = _unwrap_tokenizer(tokenizer)

    # an explicit model_info takes precedence over detection from the tokenizer
    model_info = _resolve_model_info(
        tokenizer=tokenizer,
        model_info=model_info,
        override_tag=override_tag,
    )

    # decide whether the output starts inside an open reasoning block: an explicit
    # add_generation_reasoning wins, otherwise check the prompt if one was given
    if add_generation_reasoning is not None:
        model_info = dataclasses.replace(
            model_info,
            prefixed=add_generation_reasoning,
        )
    elif prompt is not None:
        # only the first prompt is checked, assuming the batch was built the same way
        _sample: str | None = (
            prompt if isinstance(prompt, str) else (prompt[0] if prompt else None)
        )
        if _sample:
            model_info = dataclasses.replace(
                model_info,
                prefixed=_prompt_leaves_block_open(
                    prompt=_sample,
                    model_info=model_info,
                ),
            )

    if isinstance(response, str):
        return _parse_single(
            response=response,
            model_info=model_info,
            tokenizer=tokenizer,
            calculate_tokens=calculate_tokens,
        )

    if response and isinstance(response[0], list):
        # nested [task][sample] list
        return [
            [
                _parse_single(
                    response=r,
                    model_info=model_info,
                    tokenizer=tokenizer,
                    calculate_tokens=calculate_tokens,
                )
                for r in batch
            ]
            for batch in cast(list[list[str]], response)
        ]

    # flat list of strings
    return [
        _parse_single(
            response=r,
            model_info=model_info,
            tokenizer=tokenizer,
            calculate_tokens=calculate_tokens,
        )
        for r in cast(list[str], response)
    ]
