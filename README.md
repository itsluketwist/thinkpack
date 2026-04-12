# `thinkpack`

![ThinkPack](assets/banner.png)

A lightweight toolkit for working with reasoning blocks when training and evaluating language models

`thinkpack` provides five modules covering the full reasoning model workflow:

- **[Loss masking](#thinkpackmask--training-time-loss-masking)** (`thinkpack.mask`) — keeps reasoning blocks in the training context while masking them from the loss, so the model doesn't learn to skip them.
- **[Thought steering](#thinkpacksteer--inference-time-thought-steering)** (`thinkpack.steer`) — injects a short primer after the opening reasoning tag at inference time, nudging the model to reason before answering.
- **[Response parsing](#thinkpackparse--response-parsing)** (`thinkpack.parse`) — splits raw model output into reasoning and answer components, with flags for truncation detection.
- **[Reasoning distillation](#thinkpackdistill--distillation-prompt-building-and-reasoning-extraction)** (`thinkpack.distill`) — builds prompts for a teacher model to generate reasoning traces, then extracts and writes them back into your records.
- **[Hybrid decoding](#thinkpackhybrid--hybrid-decoding)** (`thinkpack.hybrid`) — separates reasoning and answering across a base model and a fine-tuned adapter for improved output quality.

---

## *installation*

Install directly from PyPI:

```bash
pip install thinkpack
```

---

## *modules*

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

### `thinkpack.distill` — Distillation prompt building and reasoning extraction

When training data lacks reasoning traces, `distill` helps construct them. It builds prompts that ask a teacher model to produce a reasoning trace given a question and its known answer, then extracts and writes those traces back into your records.

```python
import thinkpack

# build prompts for a teacher model to generate reasoning traces
prompts = thinkpack.build_prompts(
    records=records,  # list of dicts with "instruction" and "response" keys
)

# after generating responses from the teacher model, extract the traces
traces = thinkpack.extract_reasoning(text=responses, tag="reasoning_trace")

# or write traces back into records in one step
records = thinkpack.update_records(
    records=records,
    responses=responses,
    field="reasoning",  # key to write extracted traces into
)
```

`extract_reasoning` accepts a single string or a list, and returns `None` where extraction fails (blank or no tag found):

```python
# single response — returns str | None
trace = thinkpack.extract_reasoning(text=response)

# list of responses — returns list[str | None]
traces = thinkpack.extract_reasoning(text=responses)
```

---

### `thinkpack.hybrid` — Hybrid decoding

Hybrid decoding separates reasoning from answering across two model variants: the base model generates the reasoning block freely (without fine-tuning influence), and the fine-tuned adapter generates the final answer conditioned on that reasoning. This can improve answer quality when the adapter has partially collapsed.

Requires vLLM with `enable_lora=True`.

```python
from thinkpack import hybrid_generate, SimplePrefix

# steered_prompts = prompts already ending with an open reasoning tag (from steer())
results = thinkpack.hybrid_generate(
    prompts=steered_prompts,
    llm=llm,                        # vLLM LLM loaded with enable_lora=True
    lora_request=lora_request,      # adapter used for phase 2
    sampling_params=sampling_params,
)

for r in results:
    r.reasoning  # str — reasoning produced by the base model
    r.answer     # str — answer produced by the fine-tuned model
    r.raw        # str — full combined string for convenience
```

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

uv sync
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
