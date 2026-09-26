"""Chat templating that works across reasoning models, with optional thought-steering."""

from thinkpack.model import ModelInfo, _Tokenizer, _unwrap_tokenizer, get_model_info


def _inject_prefixes(
    prompt: str,
    model_info: ModelInfo,
    think_prefix: str | None = None,
    response_prefix: str | None = None,
    include_reasoning: bool | None = True,
) -> str:
    """
    Add the think and response prefixes to an already-templated prompt.

    include_reasoning controls the opening reasoning tag at the end of the prompt:
      - True  : make sure it is there, adding it if needed.
      - False : make sure it is not there, removing it if the template added one.
      - None  : leave the prompt as the template produced it.

    think_prefix seeds the model's reasoning after the opening tag.
    response_prefix seeds the model's response. If the prompt ends with an opening tag,
    it is closed first (or removed, when include_reasoning is False).

    Returns the prompt string ready for generation.
    """
    open_tag = model_info.open_tag
    close_tag = model_info.close_tag
    _prompt = prompt.rstrip("\n")
    already_open = _prompt.endswith(open_tag)

    if think_prefix is None and response_prefix is None:
        # no prefixes, so only add or remove the opening tag if needed

        if include_reasoning is None:
            # no changes needed, return as-is
            return prompt
        elif include_reasoning is False and already_open:
            # remove the opening tag added by the template
            return _prompt[: -len(open_tag)]
        elif include_reasoning is True and not already_open:
            # add the missing opening tag
            return _prompt + f"\n{open_tag}"

        # the prompt already matches what was asked for
        return _prompt

    if think_prefix is not None:
        # add open tag if not present
        if not already_open:
            _prompt += f"\n{open_tag}"

        # add the thought prefix inside the block
        _prompt += f"\n{think_prefix}"

    if response_prefix is not None:
        if think_prefix is not None:
            # think_prefix was just added — close the block before seeding the response
            _prompt += f"\n{close_tag}"
        elif already_open:
            if include_reasoning is False:
                # remove the opening tag added by the template
                _prompt = _prompt[: -len(open_tag)]
            else:
                # close the block the template opened
                _prompt += f"\n{close_tag}"

        _prompt += f"\n{response_prefix}"

    return _prompt


# placeholder for a message whose reasoning the template would strip — the template is
# rendered with the placeholder, which is then swapped for the real reasoning and content
_THINK_SENTINEL = "___THINK_INJECT_{idx}___"


def _prepare_messages(
    messages: list[dict[str, str]],
    model_info: ModelInfo,
    add_history_reasoning: bool | None = None,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """
    Embed each assistant message's 'reasoning' key into its content as a think block.

    Messages without a 'reasoning' key are unchanged. A blank 'reasoning' becomes an
    empty think block, and non-blank reasoning is wrapped in the reasoning tags.

    Assistant messages before the last user message are "history", and
    add_history_reasoning controls their reasoning:
      - None  : embed it and let the template decide whether to keep it.
      - True  : always keep it, even if the template would strip it.
      - False : always drop it.
    The final assistant message always keeps its reasoning.

    If the template would strip reasoning that should be kept, the message content is
    replaced by a placeholder, to be swapped back after the template is rendered.

    Returns (prepared_messages, sentinel_map), where sentinel_map maps each placeholder
    to its replacement text (empty when no placeholders are needed).
    """
    prepared = []
    sentinel_map: dict[str, str] = {}

    # find the last user message — assistant messages before it are history
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    last_user_idx = user_indices[-1] if user_indices else -1

    for idx, m in enumerate(messages):
        if "reasoning" not in m:
            # no reasoning key — pass the message through as-is
            prepared.append(m)
            continue

        reasoning = m["reasoning"]
        base = {k: v for k, v in m.items() if k != "reasoning"}
        content = base.get("content", "")
        is_history = idx < last_user_idx

        if is_history and add_history_reasoning is False:
            # reasoning is forced out of history, so drop it and keep only the content
            prepared.append(base)
            continue

        # decide whether the template would strip reasoning that should be kept
        if is_history:
            # history reasoning is only kept against the template when asked for
            use_sentinel = (
                add_history_reasoning is True and model_info.strips_history_think_tags
            )
        else:
            # the final assistant turn always keeps its reasoning
            use_sentinel = model_info.strips_think_tags

        if reasoning:
            # non-blank reasoning: wrap in open/close tags
            think_block = (
                f"{model_info.open_tag}\n{reasoning}\n{model_info.close_tag}\n"
            )
        else:
            # blank reasoning key: produce an empty think block
            think_block = f"{model_info.open_tag}\n{model_info.close_tag}\n"

        if use_sentinel:
            # the template would strip the think block, so render a placeholder instead
            # and record the real text to swap back in afterwards
            sentinel = _THINK_SENTINEL.format(idx=idx)
            sentinel_map[sentinel] = think_block + content
            prepared.append({**base, "content": sentinel})
        else:
            # embed the tags directly in the content
            prepared.append({**base, "content": think_block + content})

    return prepared, sentinel_map


def apply_chat_template(
    conversation: list[dict[str, str]],
    tokenizer: _Tokenizer,
    think_prefix: str | None = None,
    response_prefix: str | None = None,
    override_tag: str | None = None,
    add_generation_reasoning: bool | None = None,
    add_history_reasoning: bool | None = None,
    add_generation_prompt: bool | None = None,
    **kwargs: object,
) -> str:
    """
    Apply the chat template to a single conversation, with optional thought-steering.

    Works like the tokenizer's own apply_chat_template(), but handles each model's
    reasoning format automatically. Always returns a string, not token ids.

    add_generation_prompt is passed to the tokenizer. When None (default) the
    tokenizer's own default is used, which is False for HuggingFace tokenizers — so
    pass True when building a prompt for generation.

    Assistant messages may include a 'reasoning' key alongside 'role' and 'content',
    which is embedded as a think block (an empty block if the reasoning is blank).

    add_generation_reasoning controls the opening reasoning tag in the generation prompt:
      - None  : leave the template output unchanged (default).
      - True  : make sure the opening tag is there, adding it if needed.
      - False : make sure it is not there, removing it if the template added one.

    add_history_reasoning controls reasoning on assistant messages before the last user
    message, which some templates (e.g. Qwen3) strip:
      - None  : let the template decide (default).
      - True  : always keep the reasoning, even if the template would strip it.
      - False : always drop the reasoning.

    think_prefix seeds the model's reasoning inside an open reasoning block.
    response_prefix seeds the response, closing any open reasoning block first.
    override_tag replaces the detected reasoning tag, e.g. "<reasoning>".
    Any other kwargs are passed to tokenizer.apply_chat_template().

    Returns the templated prompt string.
    """
    if add_generation_reasoning is False and think_prefix is not None:
        raise ValueError(
            "add_generation_reasoning=False cannot be combined with think_prefix, "
            "as a think prefix needs an open reasoning block"
        )
    if add_generation_prompt is False and (
        think_prefix is not None
        or response_prefix is not None
        or add_generation_reasoning is True
    ):
        raise ValueError(
            "think_prefix, response_prefix or add_generation_reasoning=True require "
            "add_generation_prompt=True"
        )

    # multimodal processors wrap the text tokenizer — use the tokenizer directly
    tokenizer = _unwrap_tokenizer(tokenizer)

    # detect the model's reasoning format, then embed any reasoning into the messages
    model_info = get_model_info(
        tokenizer=tokenizer,
        override_tag=override_tag,
    )
    prepared, sentinel_map = _prepare_messages(
        messages=conversation,
        model_info=model_info,
        add_history_reasoning=add_history_reasoning,
    )

    # only forward add_generation_prompt when explicitly set; None defers to the tokenizer
    if add_generation_prompt is not None:
        kwargs = {**kwargs, "add_generation_prompt": add_generation_prompt}
    templated: str | list[int] = tokenizer.apply_chat_template(
        prepared,
        tokenize=False,
        **kwargs,
    )
    if isinstance(templated, list):
        # some tokenizers return token ids despite tokenize=False
        templated = tokenizer.decode(templated)

    # swap placeholders back for the reasoning the template would have stripped
    for sentinel, replacement in sentinel_map.items():
        templated = templated.replace(sentinel, replacement)

    return _inject_prefixes(
        prompt=templated,
        model_info=model_info,
        think_prefix=think_prefix,
        response_prefix=response_prefix,
        include_reasoning=add_generation_reasoning,
    )


def _resolve_prefix(
    prefix: str | list[str] | None,
    n: int,
    name: str,
) -> list[str | None]:
    """Expand a prefix argument into one prefix per conversation.

    A string or None is repeated n times; a list must already have length n.

    Returns a list of n prefixes.
    """
    if isinstance(prefix, list):
        if len(prefix) != n:
            raise ValueError(
                f"{name} list length ({len(prefix)}) "
                f"must match conversations length ({n})"
            )
        return list(prefix)
    return [prefix] * n


def apply_chat_templates(
    conversations: list[list[dict[str, str]]],
    tokenizer: _Tokenizer,
    think_prefix: str | list[str] | None = None,
    response_prefix: str | list[str] | None = None,
    override_tag: str | None = None,
    add_generation_reasoning: bool | None = None,
    add_history_reasoning: bool | None = None,
    add_generation_prompt: bool | None = None,
    **kwargs: object,
) -> list[str]:
    """
    Apply the chat template to a list of conversations.

    Calls apply_chat_template() on each conversation. think_prefix and response_prefix
    may each be a single string (used for every conversation) or a list with one entry
    per conversation. All other arguments are passed through unchanged.

    Returns a list of prompt strings, one per conversation.
    """
    n = len(conversations)
    think_prefixes = _resolve_prefix(
        prefix=think_prefix,
        n=n,
        name="think_prefix",
    )
    response_prefixes = _resolve_prefix(
        prefix=response_prefix,
        n=n,
        name="response_prefix",
    )

    return [
        apply_chat_template(
            conversation=conv,
            tokenizer=tokenizer,
            think_prefix=think_prefixes[i],
            response_prefix=response_prefixes[i],
            override_tag=override_tag,
            add_generation_reasoning=add_generation_reasoning,
            add_history_reasoning=add_history_reasoning,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        for i, conv in enumerate(conversations)
    ]
