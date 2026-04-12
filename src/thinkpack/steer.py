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
    close: bool = False,
) -> list[str]:
    """Inject a thought-steering prefix into chat-templated prompt strings.

    Ensures each prompt ends with an open reasoning block, optionally seeded
    with a prefix string to guide the model's thinking. Use SimplePrefix for
    common presets, or pass any string for a custom prefix. Template style
    (INLINE, NATIVE, PREFIXED) is detected automatically from the tokenizer.

    The prompts should already be chat-templated strings (e.g. as returned by
    tokenizer.apply_chat_template with add_generation_prompt=True).

    When close=True, the reasoning block is closed after the prefix, producing
    a complete <think>...</think> block. The model then generates its response
    after the closed block. This is useful as a universal interface for injecting
    a fixed reasoning block rather than steering an open-ended thought.

    Returns a list of prompt strings ready to pass directly to a generation function.
    """
    model_info = detect_model(tokenizer=tokenizer)
    # user-supplied tag overrides the detected default (useful for INLINE models
    # whose tag differs from the <think> default, e.g. <reasoning>)
    open_tag = f"<{tag}>" if tag is not None else model_info.open_tag
    # close tag is derived by inserting "/" after the opening "<"
    close_tag = open_tag.replace("<", "</", 1)

    steered = []
    for prompt in prompts:
        # check for any recognised reasoning tag, not just <think>
        already_open = prompt.rstrip("\n").endswith(open_tag)

        if prefix is None:
            if close:
                # inject an empty reasoning block — model responds after it
                if already_open:
                    steered.append(prompt + close_tag + "\n")
                else:
                    steered.append(prompt + open_tag + "\n" + close_tag + "\n")
            else:
                # just ensure the prompt ends with an open reasoning tag
                if already_open:
                    steered.append(prompt)
                else:
                    steered.append(prompt + open_tag + "\n")
        else:
            # inject the prefix as the beginning of the thought content
            if already_open:
                # PREFIXED template already appended the tag — add only the body
                if close:
                    steered.append(
                        prompt + "\n" + str(prefix) + "\n" + close_tag + "\n"
                    )
                else:
                    steered.append(prompt + "\n" + str(prefix))
            else:
                if close:
                    steered.append(
                        prompt
                        + open_tag
                        + "\n"
                        + str(prefix)
                        + "\n"
                        + close_tag
                        + "\n",
                    )
                else:
                    steered.append(prompt + open_tag + "\n" + str(prefix))

    return steered


def apply_steer_template(
    conversations: list[list[dict[str, str]]],
    tokenizer: object,
    prefix: SimplePrefix | str | None = None,
    tag: str | None = None,
    close: bool = False,
) -> list[str]:
    """Apply the chat template and inject a thought-steering prefix in one step.

    Convenience wrapper that combines tokenizer.apply_chat_template() and steer()
    into a single call. Accepts a list of conversations (each a list of message dicts
    with "role" and "content" keys) and returns steered prompt strings ready for
    generation. Pass close=True to produce a complete closed reasoning block.

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
        close=close,
    )
