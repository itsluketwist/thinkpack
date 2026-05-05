"""Inference-time chat templating with thought-steering and response prefix injection."""

from thinkpack.model import ModelInfo, _Tokenizer, get_model_info


def _inject_prefixes(
    prompt: str,
    model_info: ModelInfo,
    think_prefix: str | None = None,
    response_prefix: str | None = None,
    include_reasoning: bool | None = True,
) -> str:
    """
    Inject thought-steering and response prefix into an already-templated prompt string.

    include_reasoning controls the open reasoning tag:
      - True  : enforce its presence — add the tag for non-prefixed models, keep for prefixed.
      - False : enforce its absence — strip the tag if a prefixed template injected one.
      - None  : passive — leave the template output exactly as-is.

    think_prefix seeds the model's reasoning after the open tag.
    response_prefix seeds the model's response; for include_reasoning=False the tag is
    stripped first (if present), for True/None an existing open tag is closed first.

    Returns the prompt string ready for generation.
    """
    open_tag = model_info.open_tag
    close_tag = model_info.close_tag
    _prompt = prompt.rstrip("\n")
    already_open = _prompt.endswith(open_tag)

    if think_prefix is None and response_prefix is None:
        # no prefixes, just enforce tag presence or absence if needed

        if include_reasoning is None:
            # no changes needed, return as-is
            return prompt
        elif include_reasoning is False and already_open:
            # ensure no open tag, stripping it if a prefixed template injected one
            return _prompt[: -len(open_tag)]
        elif include_reasoning is True and not already_open:
            # ensure the open tag is present, adding it if needed
            return _prompt + f"\n{open_tag}\n"

        # include_reasoning=True + already_open, or False + not already_open: nothing to do
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
                # remove the tag from a prefixed template
                _prompt = _prompt[: -len(open_tag)]
            else:
                # close the block (True: explicitly; None: closes template-injected block)
                _prompt += f"\n{close_tag}"

        _prompt += f"\n{response_prefix}"

    return _prompt


def _prepare_messages(
    messages: list[dict[str, str]],
    model_info: ModelInfo,
) -> list[dict[str, str]]:
    """
    Embed reasoning into assistant messages as literal tags prepended to content.

    The 'reasoning' key controls behaviour: if absent, the message passes through
    unchanged; if present and blank, an empty think block is prepended; if present
    and non-blank, a complete block wrapping the reasoning text is prepended.
    This approach works uniformly for all models regardless of template style.
    """
    prepared = []
    for m in messages:
        if "reasoning" not in m:
            # no reasoning key — pass the message through as-is
            prepared.append(m)
            continue

        reasoning = m["reasoning"]
        base = {k: v for k, v in m.items() if k != "reasoning"}

        if reasoning:
            # non-blank reasoning: wrap in open/close tags
            tagged = f"{model_info.open_tag}\n{reasoning}\n{model_info.close_tag}\n"
        else:
            # blank reasoning key: produce an empty think block
            tagged = f"{model_info.open_tag}\n{model_info.close_tag}\n"

        prepared.append({**base, "content": tagged + base.get("content", "")})

    return prepared


def apply_chat_template(
    conversation: list[dict[str, str]],
    tokenizer: _Tokenizer,
    think_prefix: str | None = None,
    response_prefix: str | None = None,
    override_tag: str | None = None,
    add_generation_reasoning: bool | None = True,
    add_generation_prompt: bool = True,
    **kwargs: object,
) -> str:
    """
    Apply the chat template to a single conversation with optional thought-steering.

    Mirrors the tokenizer's apply_chat_template signature. Template style is detected
    automatically from the tokenizer. Always returns a prompt string — tokenization is
    left to the caller.

    Assistant messages may include a 'reasoning' key alongside 'role' and 'content'.
    If absent the message is unchanged. If present and blank, an empty think block is
    embedded. If present and non-blank, a complete block wrapping the reasoning is
    embedded. This works for any model regardless of template style.

    add_generation_reasoning controls the reasoning tag in the generation prompt:
      - True  : ensure the open tag is present, adding it if needed (default).
      - False : ensure no open tag, stripping it if a prefixed template injected one.
      - None  : leave the template output unchanged.

    think_prefix seeds the model's reasoning inside the open block.
    response_prefix seeds the response after the block closes.
    add_generation_prompt controls whether there is an assistant turn opener.

    Additional kwargs are forwarded to tokenizer.apply_chat_template(), allowing
    model-specific parameters to be passed through.

    Returns a prompt string ready to pass directly to a generation function.
    """
    if add_generation_reasoning is False and think_prefix is not None:
        raise ValueError(
            "add_generation_reasoning=False cannot be combined with think_prefix — "
            "reasoning block absence contradicts injecting a thought prefix directly"
        )
    if not add_generation_prompt and (
        think_prefix is not None
        or response_prefix is not None
        or add_generation_reasoning is True
    ):
        raise ValueError(
            "think_prefix, response_prefix or add_generation_reasoning=True require "
            "add_generation_prompt=True"
        )

    model_info = get_model_info(
        tokenizer=tokenizer,
        override_tag=override_tag,
    )
    prepared = _prepare_messages(
        messages=conversation,
        model_info=model_info,
    )

    templated: str | list[int] = tokenizer.apply_chat_template(
        prepared,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )
    if isinstance(templated, list):
        # some tokenizers return token ids despite tokenize=False
        templated = tokenizer.decode(templated)

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
    """Expand a prefix argument into a per-conversation list of length n.

    A plain string is broadcast to all n conversations; a list is validated
    and returned as-is; None becomes a list of None values.
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
    add_generation_reasoning: bool | None = True,
    add_generation_prompt: bool = True,
    **kwargs: object,
) -> list[str]:
    """
    Apply the chat template to a list of conversations.

    Convenience wrapper around apply_chat_template for batched use. think_prefix and
    response_prefix may each be a plain string (applied to every conversation) or a
    list with one entry per conversation. All other arguments are forwarded unchanged.

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
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        for i, conv in enumerate(conversations)
    ]
