# ***`ThinkPack`***

[![lint](https://github.com/itsluketwist/thinkpack/actions/workflows/lint.yaml/badge.svg)](https://github.com/itsluketwist/thinkpack/actions/workflows/lint.yaml)
[![test](https://github.com/itsluketwist/thinkpack/actions/workflows/test.yaml/badge.svg)](https://github.com/itsluketwist/thinkpack/actions/workflows/test.yaml)
[![release](https://github.com/itsluketwist/thinkpack/actions/workflows/release.yaml/badge.svg)](https://github.com/itsluketwist/thinkpack/actions/workflows/release.yaml)

A lightweight framework for reasoning-aware training, parsing, and evaluation of explicit reasoning language models.
Focussed on the characterisation and mitigation of **reasoning-trace collapse**.

***`ThinkPack`*** provides four focused modules:

- 💬 **[Chat templating](#thinkpackchat--chat-templating)** (`thinkpack.chat`) — applies chat templates with optional thought-steering and reasoning history embedding.
- 🔍 **[Response parsing](#thinkpackparse--response-parsing)** (`thinkpack.parse`) — splits raw model output into reasoning and answer components, with flags for presence, validity, and truncation.
- 📊 **[Statistics](#thinkpackstats--response-statistics)** (`thinkpack.stats`) — aggregates parsed responses into VR, ER, TR, MR, and Rpass@1, making reasoning-trace collapse measurable.
- 🎭 **[Loss masking](#thinkpackmask--training-time-loss-masking)** (`thinkpack.mask`) — the core method; prevents reasoning-trace collapse during fine-tuning by masking think blocks from the loss.

---

## *reasoning-trace collapse*

**Reasoning collapse** is the progressive loss of a model's ability to produce valid reasoning traces during fine-tuning. A model may still answer correctly, but stop producing a complete reasoning trace:

```text
before fine-tuning:  x → <think> reasoning </think> answer
after naive SFT:     x → <think> </think> answer
or simply:           x → answer
```

This can happen when a reasoning model is adapted with ordinary instruction–response data that contains final answers, but no explicit reasoning traces.
Standard supervised fine-tuning then gives the model a clear signal to produce the answer, but no signal to preserve the reasoning structure it learned during post-training.

`ThinkPack` helps make this behaviour visible by parsing outputs into reasoning and answer segments, then tracking whether reasoning is:

- valid: complete, non-empty, and extractable
- empty: delimiters are present, but contain no reasoning
- truncated: reasoning starts, but does not close
- missing: no reasoning trace can be extracted

It also supports reasoning-aware loss masking, so you can fine-tune on non-reasoning data without directly rewarding the model for producing empty or missing reasoning.

---

## *installation*

Requires [Python 3.11+](https://www.python.org/), install directly from [PyPI](https://pypi.org/project/thinkpack/):

```bash
pip install thinkpack
```

---

## *modules*

### `thinkpack.chat` — Chat templating

A model-aware wrapper around `tokenizer.apply_chat_template()`. Handles reasoning tag injection, thought-steering, and reasoning history embedding uniformly across all model types — no manual per-model configuration needed.

```python
# basic usage — applies the correct template for the model automatically
prompt = thinkpack.apply_chat_template(
    conversation=conversation,  # list of {"role": ..., "content": ...} dicts
    tokenizer=tokenizer,
)

# thought-steering — seed the model's reasoning before generation
prompt = thinkpack.apply_chat_template(
    conversation=conversation,
    tokenizer=tokenizer,
    think_prefix="Let me break this down step by step.",  # seeds reasoning block
    response_prefix="The answer is",                       # seeds final response
)

# embed reasoning into assistant messages for multi-turn conversations
conversation = [
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "reasoning": "2 + 2 = 4", "content": "4"},
    {"role": "user", "content": "And 3 + 3?"},
]
prompt = thinkpack.apply_chat_template(conversation=conversation, tokenizer=tokenizer)

# batch variant accepts a list of conversations
prompts = thinkpack.apply_chat_templates(conversations=conversations, tokenizer=tokenizer)
```

The `add_generation_reasoning` parameter controls the reasoning tag in the generation prompt:

| Value | Effect |
|---|---|
| `True` (default) | Ensure the opening reasoning tag is present — add it if needed |
| `False` | Ensure no opening tag — strip it if a prefixed template injected one |
| `None` | Leave the template output unchanged |

See [examples/notebooks/apply_chat.ipynb](examples/notebooks/apply_chat.ipynb) for interactive examples.

---

### `thinkpack.parse` — Response parsing

Parse raw model outputs into structured components. Each `ParsedResponse` carries flags that directly support reasoning-trace collapse analysis.

```python
# single response
parsed = thinkpack.parse(response=raw_text, tokenizer=tokenizer)
parsed.answer                   # str — text after the closing reasoning tag
parsed.reasoning                # str — content of the reasoning block
parsed.has_valid_reasoning      # bool — non-empty, completed reasoning block (→ VR)
parsed.has_missing_reasoning    # bool — no reasoning block found at all
parsed.has_truncated_reasoning  # bool — reasoning block opened but never closed
parsed.has_empty_reasoning      # bool — reasoning block opened and closed, but blank

# batch of responses (list accepted directly)
parsed_list = thinkpack.parse(response=responses, tokenizer=tokenizer)
```

Handles all four output formats:

| Format | Example |
|---|---|
| Standard | `<think>reasoning</think>answer` |
| Prefixed template | `reasoning</think>answer` (opening tag injected by template) |
| Truncated standard | `<think>reasoning...` (no closing tag) |
| Truncated prefixed | `reasoning...` (detected automatically for prefixed models) |

Recognises tag variants: `think`, `thinking`, `reasoning`, `thought` (case-insensitive).

See [examples/notebooks/parse_and_stats.ipynb](examples/notebooks/parse_and_stats.ipynb) for interactive examples.

---

### `thinkpack.stats` — Response statistics

Aggregates a batch of parsed responses into counts, exposing the structural metrics used to measure reasoning-trace collapse.

```python
parsed_list = thinkpack.parse(response=responses, tokenizer=tokenizer)
s = thinkpack.compute_stats(responses=parsed_list)

# reasoning-trace collapse metrics — all rates in [0, 1]
s.valid_reasoning_rate     # float — VR: fraction with complete, non-blank reasoning
s.missing_reasoning_rate   # float — MR: fraction with no reasoning block at all
s.total                    # int — total responses

# additional breakdown
s.truncated_reasoning_rate  # float — TR: block opened but never closed
s.empty_reasoning_rate      # float — ER: block opened and closed, but blank
s.answer_rate               # float — fraction with a non-blank answer
```

`valid_reasoning_rate` and the three invalid sub-types (`ER`, `TR`, `MR`) sum to 1.

**Key metrics for the paper:**

| Metric | Definition | Interpretation |
|---|---|---|
| **VR** | `valid_reasoning_rate` | Fraction with structurally valid reasoning (primary structural metric) |
| **ER** | `empty_reasoning_rate` | Fraction with an empty reasoning block (delimiters present, no content) |
| **TR** | `truncated_reasoning_rate` | Fraction where reasoning starts but is never closed |
| **MR** | `missing_reasoning_rate` | Fraction with no reasoning block at all |
| **pass@1** | accuracy on first sample | Standard answer correctness |
| **Rpass@1** | accuracy among VR=True samples | Accuracy conditioned on valid reasoning |

Reasoning collapse is observable as VR → 0 over training steps or data size.

---

### `thinkpack.mask` — Training-time loss masking

The core method. When fine-tuning a reasoning model, `apply_mask()` formats training records into a pretokenized HuggingFace dataset with selected sections excluded from the loss. Masking the think block prevents the model from learning to skip it.

```python
import thinkpack

# masking-based SFT — prevents reasoning-trace collapse
dataset = thinkpack.apply_mask(
    conversations=conversations,  # list of conversation dicts with "role" and "content" keys
    tokenizer=tokenizer,
    masked=thinkpack.MaskType.THINK,  # mask the think block from the loss
)

# naive SFT — causes reasoning-trace collapse (use as baseline)
naive_dataset = thinkpack.apply_mask(
    conversations=conversations,
    tokenizer=tokenizer,
    masked=None,  # no masking — all tokens contribute to the loss
)
```

The `masked` parameter is a composable flag — combine sections with `|`:

| Value | Effect |
|---|---|
| `MaskType.THINK` | Think block hidden from loss; model trains on prompt + response |
| `MaskType.PROMPT \| MaskType.THINK` | Train on response only |
| `None` | No masking; all tokens contribute to the loss (naive baseline) |

See [examples/notebooks/loss_masking.ipynb](examples/notebooks/loss_masking.ipynb) for interactive examples.

---

## *agent skill*

`thinkpack` ships with an `llms.txt` file and a CLI command to install it as an agent skill in your project.
This gives AI coding assistants (Claude Code, Cursor, Windsurf) immediate, accurate context about the library.

Install the skill for your preferred tool from your project root:

```bash
thinkpack skill --tool claude     # .claude/commands/thinkpack.md
thinkpack skill --tool cursor     # .cursor/rules/thinkpack.mdc
thinkpack skill --tool windsurf   # .windsurf/rules/thinkpack.md
```

Or print the raw `llms.txt` content to stdout:

```bash
thinkpack skill
```

---

## *contributing*

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get involved, and [DEVELOPMENT.md](DEVELOPMENT.md) for environment setup.
