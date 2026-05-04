# `ThinkPack`

[![lint](https://github.com/itsluketwist/thinkpack/actions/workflows/lint.yaml/badge.svg)](https://github.com/itsluketwist/thinkpack/actions/workflows/lint.yaml)
[![test](https://github.com/itsluketwist/thinkpack/actions/workflows/test.yaml/badge.svg)](https://github.com/itsluketwist/thinkpack/actions/workflows/test.yaml)
[![release](https://github.com/itsluketwist/thinkpack/actions/workflows/release.yaml/badge.svg)](https://github.com/itsluketwist/thinkpack/actions/workflows/release.yaml)

A framework for training, parsing, and evaluating explicit reasoning models — centred on **reasoning collapse**.

`thinkpack` provides four focused modules:

- 💬 **[Chat templating](#thinkpackchat--chat-templating)** (`thinkpack.chat`) — applies chat templates with optional thought-steering and reasoning history embedding.
- 🔍 **[Response parsing](#thinkpackparse--response-parsing)** (`thinkpack.parse`) — splits raw model output into reasoning and answer components, with flags for presence, validity, and truncation.
- 📊 **[Statistics](#thinkpackstats--response-statistics)** (`thinkpack.stats`) — aggregates parsed responses into AR and VR rates, making reasoning collapse measurable.
- 🎭 **[Loss masking](#thinkpackmask--training-time-loss-masking)** (`thinkpack.mask`) — the core method; prevents reasoning collapse during fine-tuning by masking think blocks from the loss.

---

## *reasoning collapse*

**Reasoning collapse** is a failure mode that occurs when fine-tuning reasoning-enabled models on standard instruction–response data:

> The model learns to skip its reasoning block entirely — producing answers directly without a `<think>` trace.

This happens because the response alone is sufficient to minimise cross-entropy loss. The reasoning block provides no training signal and becomes an obstacle the model learns to avoid.

```
before fine-tuning:   x → <think>reasoning</think> answer
after naive SFT:      x → answer
```

ThinkPack makes this phenomenon **observable**, **measurable**, and **preventable**.

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

Parse raw model outputs into structured components. Each `ParsedResponse` carries flags that directly support reasoning collapse analysis.

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

---

### `thinkpack.stats` — Response statistics

Aggregates a batch of parsed responses into counts, exposing the **AR** and **VR** rates used to measure reasoning collapse.

```python
parsed_list = thinkpack.parse(response=responses, tokenizer=tokenizer)
s = thinkpack.compute_stats(responses=parsed_list)

# reasoning collapse metrics — all rates in [0, 1]
s.valid_reasoning_rate     # float — VR: fraction with complete, non-blank reasoning
s.missing_reasoning_rate   # float — fraction with no reasoning block at all
s.total                    # int — total responses

# AR: fraction with any reasoning structure (valid + truncated + empty)
ar = 1 - s.missing_reasoning_rate

# additional breakdown
s.truncated_reasoning_rate  # float — block opened but never closed
s.empty_reasoning_rate      # float — block opened and closed, but blank
s.answer_rate               # float — fraction with a non-blank answer
```

`valid_reasoning_rate` and `invalid_reasoning_rate` sum to 1. The three invalid sub-types (`missing`, `truncated`, `empty`) sum to `invalid_reasoning_rate`.

**Key metrics for the paper:**

| Metric | Definition | Interpretation |
|---|---|---|
| **AR** | `1 - missing_reasoning_rate` | Fraction with any reasoning structure present |
| **VR** | `valid_reasoning_rate` | Fraction with structurally valid reasoning |
| **pass@1** | accuracy on first sample | Standard answer correctness |
| **Rpass@1** | accuracy among VR=True samples | Accuracy conditioned on valid reasoning |

Reasoning collapse is observable as VR → 0 over training steps or data size.

See [examples/scripts/inference.py](examples/scripts/inference.py) for a complete collapse measurement pipeline.

---

### `thinkpack.mask` — Training-time loss masking

The core method. When fine-tuning a reasoning model, `apply_mask()` formats training records into a pretokenized HuggingFace dataset with selected sections excluded from the loss. Masking the think block prevents the model from learning to skip it.

```python
import thinkpack

# masking-based SFT — prevents reasoning collapse
dataset = thinkpack.apply_mask(
    conversations=conversations,  # list of conversation dicts with "role" and "content" keys
    tokenizer=tokenizer,
    masked=thinkpack.MaskType.THINK,  # mask the think block from the loss
)

# naive SFT — causes reasoning collapse (use as baseline)
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

Model-specific template handling (Qwen3's native `reasoning_content` field, OLMo-3's auto-injected opening tag) is detected automatically from the tokenizer — no manual configuration needed.

See [examples/scripts/training.py](examples/scripts/training.py) for a complete comparison of naive vs masking-based SFT.

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
