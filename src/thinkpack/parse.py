"""Parsing of model responses into reasoning and answer components."""

import dataclasses
from dataclasses import dataclass
from typing import cast, overload

from thinkpack.model import ModelInfo, _Tokenizer, get_model_info


@dataclass
class ParsedResponse:
    """A model response split into reasoning and answer components.

    Reasoning is classified as valid or invalid. The three invalid sub-types
    (has_truncated_reasoning, has_empty_reasoning, has_missing_reasoning) are mutually
    exclusive, and together with has_valid_reasoning they sum to 1.
    """

    # text after the closing reasoning tag, or the full response if no block was found
    answer: str

    # content inside the reasoning block; empty string if no block was present
    reasoning: str

    # tag name as reported by model_info, e.g. "think" (None if no tag found)
    reasoning_tag: str | None

    # true if the reasoning block was completed and non-blank
    has_valid_reasoning: bool

    # true if an opening tag was found but no closing tag followed
    has_truncated_reasoning: bool

    # true if a reasoning block was opened and closed but its content was blank
    has_empty_reasoning: bool

    # true if no reasoning block structure could be found in the response
    has_missing_reasoning: bool

    # token count of the reasoning content; None if not calculated
    reasoning_token_count: int | None = None

    # token count of the answer content; None if not calculated
    answer_token_count: int | None = None

    @property
    def has_invalid_reasoning(self) -> bool:
        """True if reasoning is absent, truncated, or empty — not usable for analysis."""
        return not self.has_valid_reasoning

    @property
    def extracted_answer(self) -> bool:
        """True if the answer is non-empty and non-whitespace."""
        return bool(self.answer.strip())


def _parse_single(
    response: str,
    model_info: ModelInfo,
    tokenizer: _Tokenizer | None,
    calculate_tokens: bool,
) -> ParsedResponse:
    """Parse one response string using pre-resolved model_info."""
    prefixed = model_info.prefixed
    open_re, close_re = model_info.tag_regex
    close_match = close_re.search(response)

    if close_match:
        # strip the open tag (prefixed models have none in decoded output, sub is a no-op)
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
            reasoning=response[open_match.end() :],
            reasoning_tag=model_info.tag_content,
            has_valid_reasoning=False,
            has_truncated_reasoning=True,
            has_empty_reasoning=False,
            has_missing_reasoning=False,
        )

    elif prefixed:
        # prefixed models inject the open tag via the chat template — it never appears
        # in decoded output, so a missing close tag means truncation, not absence
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
) -> ParsedResponse: ...


@overload
def parse(
    response: list[str],
    tokenizer: _Tokenizer | None = ...,
    override_tag: str | None = ...,
    model_info: ModelInfo | None = ...,
    calculate_tokens: bool = ...,
) -> list[ParsedResponse]: ...


@overload
def parse(
    response: list[list[str]],
    tokenizer: _Tokenizer | None = ...,
    override_tag: str | None = ...,
    model_info: ModelInfo | None = ...,
    calculate_tokens: bool = ...,
) -> list[list[ParsedResponse]]: ...


def parse(
    response: str | list[str] | list[list[str]],
    tokenizer: _Tokenizer | None = None,
    override_tag: str | None = None,
    model_info: ModelInfo | None = None,
    calculate_tokens: bool = False,
) -> ParsedResponse | list[ParsedResponse] | list[list[ParsedResponse]]:
    """Parse one or more model responses into reasoning and answer components.

    Accepts a single string, a flat list of strings, or a nested [task][sample] list.
    Handles standard (<think>content</think>answer), prefixed (content</think>answer),
    truncated standard (<think>content...), and truncated prefixed (content...) formats.

    Pass tokenizer to auto-detect model_info; override_tag overrides the detected tag.
    Pass model_info directly to skip detection (e.g. when reusing across a batch).
    At least one of tokenizer or model_info must be provided.

    Token counts are populated when calculate_tokens=True and a tokenizer is provided.

    Returns a ParsedResponse, list[ParsedResponse], or list[list[ParsedResponse]]
    matching the shape of the input.
    """
    if tokenizer is not None:
        # detect model properties from the tokenizer; override_tag replaces the detected tag
        model_info = get_model_info(tokenizer=tokenizer, override_tag=override_tag)
    elif model_info is None:
        raise ValueError("One of tokenizer or model_info must be provided.")

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
