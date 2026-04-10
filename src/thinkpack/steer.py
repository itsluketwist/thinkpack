"""Inference-time thought-steering prefix injection."""

from enum import StrEnum

from thinkpack._model import detect_model


class SimplePrefix(StrEnum):
    """
    A small set of basic steering prefixes for common use cases.

    These are provided as convenient starting points — pass any string to
    steer() to use a custom prefix instead.
    """

    # minimal opening; lets the model continue naturally with a slight nudge
    BRIEF = "Okay, "
    # explicit step-by-step framing
    STEPS = "Okay, let me think this through step by step."
    # step-by-step framing with a reminder to stay concise and produce an answer
    CONCISE = (
        "Okay, let me think this through, "
        "but I need to be concise and make sure I also provide an answer."
    )


def steer(
    prompts: list[str],
    tokenizer: object,
    prefix: SimplePrefix | str | None = None,
    tag: str | None = None,
) -> list[str]:
    """Inject a thought-steering prefix into chat-templated prompt strings.

    Ensures each prompt ends with an open reasoning block, optionally seeded
    with a custom prefix to guide the model's thinking before it generates.

    - prefix=None:   ensures the opening tag is present (already the case for
                     PREFIXED templates; appended for INLINE/NATIVE templates)
    - prefix=<str>:  the string is placed immediately after the opening tag,
                     seeding the model's thought. Use SimplePrefix for common
                     presets, or pass any string for a custom prefix.

    For INLINE models that use a non-default tag (e.g. <reasoning> instead of
    <think>), pass tag="reasoning" to override the detected default. PREFIXED
    and NATIVE models always use the tag extracted from the template.

    The prompts should already be chat-templated strings (e.g. as returned by
    tokenizer.apply_chat_template with add_generation_prompt=True). Template style
    (INLINE, NATIVE, PREFIXED) is detected automatically from the tokenizer.

    Returns a list of prompt strings ready to pass directly to a generation function.
    """
    model_info = detect_model(tokenizer=tokenizer)
    # user-supplied tag overrides the detected default (useful for INLINE models
    # whose tag differs from the <think> default, e.g. <reasoning>)
    open_tag = f"<{tag}>" if tag is not None else model_info.open_tag

    steered = []
    for prompt in prompts:
        # check for any recognised reasoning tag, not just <think>
        already_open = prompt.rstrip("\n").endswith(open_tag)

        if prefix is None:
            # just ensure the prompt ends with an open reasoning tag
            if already_open:
                steered.append(prompt)
            else:
                steered.append(prompt + open_tag + "\n")
        else:
            # inject the prefix as the beginning of the thought content
            if already_open:
                # PREFIXED template already appended the tag — add only the body
                steered.append(prompt + "\n" + str(prefix))
            else:
                steered.append(prompt + open_tag + "\n" + str(prefix))

    return steered


def apply_steer_template(
    conversations: list[list[dict]],
    tokenizer: object,
    prefix: SimplePrefix | str | None = None,
    tag: str | None = None,
) -> list[str]:
    """Apply the chat template and inject a thought-steering prefix in one step.

    Convenience wrapper that combines tokenizer.apply_chat_template() and steer()
    into a single call. Accepts a list of conversations (each a list of message dicts
    with "role" and "content" keys) and returns steered prompt strings ready for
    generation.

    - prefix=None:   ensures the opening tag is present without seeding the thought
    - prefix=<str>:  seeds the model's thought with the given string
    - tag:           override the reasoning tag for INLINE models (see steer())

    Returns a list of steered prompt strings, one per conversation.
    """
    # apply the chat template to each conversation
    templated = []
    for messages in conversations:
        result = tokenizer.apply_chat_template(  # type: ignore
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if isinstance(result, list):
            # some tokenizers return token ids despite tokenize=False — decode them
            result = tokenizer.decode(result)  # type: ignore
        templated.append(result)

    return steer(
        prompts=templated,
        tokenizer=tokenizer,
        prefix=prefix,
        tag=tag,
    )
