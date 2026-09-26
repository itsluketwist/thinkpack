"""Inference-time chat templating with thought-steering and response prefix injection."""

from thinkpack.model import ModelInfo, _Tokenizer, _unwrap_tokenizer, get_model_info


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
            return _prompt + f"\n{open_tag}"

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


# sentinel format used to mark think-injection points for strips_think_tags models;
# template rendering is done with these placeholders, then replaced with the real blocks
_THINK_SENTINEL = "___THINK_INJECT_{idx}___"


def _prepare_messages(
    messages: list[dict[str, str]],
    model_info: ModelInfo,
    add_history_reasoning: bool | None = None,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """
    Embed reasoning into assistant messages as literal tags prepended to content.

    The 'reasoning' key controls behaviour: if absent, the message passes through
    unchanged; if present and blank, an empty think block is prepended; if present
    and non-blank, a complete block wrapping the reasoning text is prepended.

    Messages before the last user message are "history". Some templates (e.g. Qwen3)
    strip reasoning from history, so add_history_reasoning controls these messages:
      - None  : embed the tags and let the template decide what to keep.
      - True  : always keep the reasoning, even if the template would strip it.
      - False : always drop the reasoning.
    Messages after the last user message (the final assistant turn) always keep their
    reasoning, as this is required for training.

    Where the template would strip a block that should be kept, a sentinel placeholder
    replaces the assistant content during template rendering, and the think+content
    block is recorded for substitution after the template runs.

    Returns (prepared_messages, sentinel_map) where sentinel_map maps each sentinel string
    to its think+content replacement. Empty when no sentinels are needed.
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

        # decide whether the template would strip this block when we want to keep it
        if is_history:
            # history reasoning is only forced back in when explicitly requested
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
            # template would strip the think block, so use a sentinel instead;
            # the real think+content is recorded and re-injected after rendering
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
    Apply the chat template to a single conversation with optional thought-steering.

    Mirrors the tokenizer's apply_chat_template signature. Template style is detected
    automatically from the tokenizer. Always returns a prompt string — tokenization is
    left to the caller.

    Assistant messages may include a 'reasoning' key alongside 'role' and 'content'.
    If absent the message is unchanged. If present and blank, an empty think block is
    embedded. If present and non-blank, a complete block wrapping the reasoning is
    embedded. This works for any model regardless of template style.

    add_generation_reasoning controls the open reasoning tag in the generation prompt:
      - None  : leave the template output unchanged (default).
      - True  : ensure the open tag is present, adding it for non-prefixed models.
      - False : strip the open tag if a prefixed template injected one.

    add_history_reasoning controls reasoning on assistant messages before the last user
    message, which some templates (e.g. Qwen3) strip:
      - None  : leave it to the template (default).
      - True  : always keep the reasoning, even if the template would strip it.
      - False : always drop the reasoning.

    think_prefix seeds the model's reasoning inside the open block.
    response_prefix seeds the response after the block closes.
    add_generation_prompt is forwarded directly to the tokenizer; when None (default)
    the tokenizer's own default is used.

    Additional kwargs are forwarded to tokenizer.apply_chat_template(), allowing
    model-specific parameters to be passed through.

    Returns a prompt string ready to pass directly to a generation function.
    """
    if add_generation_reasoning is False and think_prefix is not None:
        raise ValueError(
            "add_generation_reasoning=False cannot be combined with think_prefix — "
            "reasoning block absence contradicts injecting a thought prefix directly"
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

    # re-inject think blocks bypassed via sentinels (strips_think_tags models)
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
    add_generation_reasoning: bool | None = None,
    add_history_reasoning: bool | None = None,
    add_generation_prompt: bool | None = None,
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
            add_history_reasoning=add_history_reasoning,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        for i, conv in enumerate(conversations)
    ]
