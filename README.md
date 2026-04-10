# ThinkPack

![ThinkPack](assets/banner.png)

A lightweight toolkit for working with reasoning blocks in language models — preventing think collapse via los masking, steering reasoning at inference time, and parsing model outputs.

**Think collapse** is a failure mode where reasoning models stop using their `<think>...</think>` blocks during or after fine-tuning.
Without intervention, the model learns to skip reasoning entirely — producing answers directly and losing the chain-of-thought behaviour it was trained on.
ThinkPack provides three targeted tools to prevent this:

- **Loss masking** (`thinkpack.mask`) — keeps reasoning blocks in the training context while masking them from the loss, so the model doesn't learn to skip them.
- **Thought steering** (`thinkpack.steer`) — injects a short primer after the opening reasoning tag at inference time, nudging the model to reason before answering.
- **Response parsing** (`thinkpack.parse`) — splits raw model output into reasoning and answer components, with flags for truncation detection.

---

## Installation

```bash
pip install thinkpack
```

---

## Modules

### `thinkpack.mask` — Training-time loss masking

When fine-tuning a reasoning model, naively training on all tokens can cause the model to learn to skip its reasoning block entirely. `mask()` formats your training records into a pretokenized HuggingFace dataset with selected parts of the sequence excluded from the loss.

```python
import thinkpack

dataset = thinkpack.mask(
    records=records,    # list of dicts with "instruction" and "response" keys
    tokenizer=tokenizer,
    masked=thinkpack.Mask.THINK,  # mask only the think block (default)
)
```

The `masked` parameter is a composable flag — combine sections with `|`:

| Value | Effect |
|---|---|
| `Mask.THINK` | Think block hidden from loss; model trains on prompt + response |
| `Mask.PROMPT \| Mask.THINK` | Train on response only |
| `None` | No masking; all tokens contribute to the loss |

Model-specific template handling (Qwen3's native `reasoning_content` field, OLMo-3's auto-injected opening tag) is detected automatically from the tokenizer — no manual configuration needed.

See [examples/training.py](examples/training.py) for a complete training loop.

---

### `thinkpack.steer` — Inference-time thought steering

Think collapse can also be addressed at inference time by injecting a short prefix after the opening reasoning tag, seeding the model's reasoning before it generates its own thought content.

```python
# ensure the opening reasoning tag is present without seeding the thought
steered_prompts = thinkpack.steer(
    prompts=templated_prompts,  # already chat-templated strings
    tokenizer=tokenizer,
)

# seed the model's thought with a preset
steered_prompts = thinkpack.steer(
    prompts=templated_prompts,
    tokenizer=tokenizer,
    prefix=thinkpack.SimplePrefix.CONCISE,
)

# or pass any custom string
steered_prompts = thinkpack.steer(
    prompts=templated_prompts,
    tokenizer=tokenizer,
    prefix="Okay, this is a tricky one. Let me consider each part carefully.",
)
```

`SimplePrefix` provides a few basic presets:

| Preset | Text |
|---|---|
| `BRIEF` | `"Okay, "` |
| `STEPS` | `"Okay, let me think this through step by step."` |
| `CONCISE` | `"Okay, let me think this through, but I need to be concise and make sure I also provide an answer."` |

`steer()` handles the PREFIXED template quirk automatically: models like OLMo-3 whose chat template already ends with an opening reasoning tag do not get a duplicate tag injected.

See [examples/inference.py](examples/inference.py) for a complete inference loop.

---

### `thinkpack.parse` — Response parsing

Parse raw model outputs into structured components — useful for evaluation, analysis, and hybrid decoding pipelines.

```python
# single response
parsed = thinkpack.parse(response=raw_text)
parsed.answer                   # str — text after the closing reasoning tag
parsed.reasoning                # str — content of the reasoning block
parsed.has_valid_reasoning      # bool — non-empty, completed reasoning block
parsed.has_truncated_reasoning  # bool — reasoning block started but never closed

# directly from vLLM output objects (single output → list, list of outputs → list[list])
parsed = thinkpack.parse_output(output=outputs)
```

Handles all four output formats:

| Format | Example |
|---|---|
| Standard | `<think>reasoning</think>answer` |
| Prefixed template | `reasoning</think>answer` (opening tag injected by template) |
| Truncated standard | `<think>reasoning...` (no closing tag) |
| Truncated prefixed | `reasoning...` (pass `prefixed=True`) |

Recognises tag variants: `think`, `thinking`, `reasoning`, `thought` (case-insensitive).

---

## *development*

Clone the repository code:

```shell
git clone https://github.com/itsluketwist/thinkpack.git
```

We use [`uv`](https://astral.sh/blog/uv) for project management.
Once cloned, create a virtual environment and install the project with dev dependencies:

```shell
python -m venv .venv

. .venv/bin/activate

pip install uv

uv sync --extra dev
```

Use `make` commands to lint and test:

```shell
make lint

make test
```

Use `uv` to add new dependencies into the project:

```shell
uv add transformers
```

Or to upgrade dependencies:

```shell
uv sync --upgrade
```

Check typings with `ty`:

```shell
uv run --extra dev ty check src tests
```
