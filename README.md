# ***`ThinkPack`***

[![lint](https://github.com/itsluketwist/thinkpack/actions/workflows/lint.yaml/badge.svg)](https://github.com/itsluketwist/thinkpack/actions/workflows/lint.yaml)
[![test](https://github.com/itsluketwist/thinkpack/actions/workflows/test.yaml/badge.svg)](https://github.com/itsluketwist/thinkpack/actions/workflows/test.yaml)
[![release](https://github.com/itsluketwist/thinkpack/actions/workflows/release.yaml/badge.svg)](https://github.com/itsluketwist/thinkpack/actions/workflows/release.yaml)

A lightweight framework for reasoning-aware training, parsing, and evaluation of explicit reasoning language models.
Focussed on the characterisation and mitigation of **reasoning-trace collapse**.

***`ThinkPack`*** provides four core modules, plus distillation helpers:

- 💬 **[Chat templating](#thinkpackchat--chat-templating)** (`thinkpack.chat`) — applies chat templates with optional thought-steering and reasoning history embedding.
- 🔍 **[Response parsing](#thinkpackparse--response-parsing)** (`thinkpack.parse`) — splits raw model output into reasoning and answer components, with flags for presence, validity, and truncation.
- 📊 **[Statistics](#thinkpackstats--response-statistics)** (`thinkpack.stats`) — aggregates parsed responses into VR, ER, TR, MR, and Rpass@1, making reasoning-trace collapse measurable.
- 🎭 **[Loss masking](#thinkpackmask--training-time-loss-masking)** (`thinkpack.mask`) — masks think blocks from the loss during fine-tuning, a simple mitigation that can help preserve reasoning traces.
- 🧪 **[Distillation](#thinkpackdistill--reasoning-distillation)** (`thinkpack.distill`) — uses a teacher model to add reasoning traces to instruction–response data.

> 📄 Accompanies the paper [**Reasoning-Trace Collapse: Evaluating the Loss of Explicit Reasoning During Fine-Tuning**](https://arxiv.org/abs/2605.21127), accepted to the NeurIPS 2026 Evaluations and Datasets Track — see [*citation*](#citation).

---

## *reasoning-trace collapse*

**Reasoning-trace collapse** is the progressive loss of a model's ability to produce valid reasoning traces during fine-tuning. A model may still answer correctly, but stop producing a complete reasoning trace:

```text
before fine-tuning:  x → <think> reasoning </think> answer
after naive SFT:     x → <think> </think> answer
or simply:           x → answer
```

This can happen when a reasoning model is fine-tuned on ordinary instruction–response data that contains final answers, but no reasoning traces.
Standard supervised fine-tuning then trains the model to produce the answer, but gives it no reason to keep the reasoning it learned during post-training.

`ThinkPack` makes this visible by parsing outputs into reasoning and answer segments, then tracking whether reasoning is:

- valid: complete, non-empty, and extractable
- empty: delimiters are present, but contain no reasoning
- truncated: reasoning starts, but does not close
- missing: no reasoning trace can be extracted

It also supports reasoning-aware loss masking, so you can fine-tune on non-reasoning data without directly training the model to produce empty or missing reasoning.
Masking can mitigate collapse, but its effect is model- and task-dependent (in the paper, it only partially helps OLMo-3), so measure VR after fine-tuning rather than assuming reasoning has been preserved.

---

## *installation*

Requires [Python 3.11+](https://www.python.org/), install directly from [PyPI](https://pypi.org/project/thinkpack/):

```bash
pip install thinkpack
```

**Compatibility:** tested with `transformers` 4.57 and 5.x, and with Qwen3, Qwen3.5, DeepSeek-R1-Distill, OLMo-3, and Ministral-3 reasoning models.
`transformers` 5.3 to 5.12 are excluded, as they load some tokenizers (e.g. DeepSeek-R1-Distill) incorrectly and silently produce wrong token ids ([transformers#45488](https://github.com/huggingface/transformers/issues/45488)).
`thinkpack` logs a warning if a tokenizer cannot round-trip plain text.
Multimodal processors (e.g. Qwen3.5 loaded via `AutoProcessor` or unsloth) can be passed wherever a tokenizer is expected.

---

## *modules*

Every function detects the model's reasoning format from the tokenizer's chat template: the reasoning tag, whether the template opens the reasoning block in the generation prompt, and whether it strips reasoning from earlier messages.
So the same code works across models, with no per-model configuration.
Call `thinkpack.detect_model(tokenizer)` to see what was detected, and pass `override_tag=` (e.g. `"<reasoning>"`) to any function if the wrong tag is found.
For full control, pass a custom `thinkpack.ModelInfo` as `model_info=` to any function to skip detection entirely — `parse()` then needs no tokenizer at all.

### `thinkpack.chat` — Chat templating

A drop-in replacement for `tokenizer.apply_chat_template()` that handles reasoning tags, thought-steering, and reasoning history the same way across all models.

```python
# build a prompt for generation
prompt = thinkpack.apply_chat_template(
    conversation=conversation,  # list of {"role": ..., "content": ...} dicts
    tokenizer=tokenizer,
    add_generation_prompt=True,  # open the assistant turn, ready for generation
)

# thought-steering — the model continues its reasoning from the given text
prompt = thinkpack.apply_chat_template(
    conversation=conversation,
    tokenizer=tokenizer,
    add_generation_prompt=True,
    think_prefix="Let me break this down step by step.",
)

# add reasoning to earlier assistant messages in a multi-turn conversation
conversation = [
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "reasoning": "2 + 2 = 4", "content": "4"},
    {"role": "user", "content": "And 3 + 3?"},
]
prompt = thinkpack.apply_chat_template(
    conversation=conversation,
    tokenizer=tokenizer,
    add_generation_prompt=True,
    add_history_reasoning=True,  # keep the reasoning, even if the template strips it
)

# batch version, taking a list of conversations
prompts = thinkpack.apply_chat_templates(
    conversations=conversations,
    tokenizer=tokenizer,
    add_generation_prompt=True,
)
```

As with the tokenizer, `add_generation_prompt` defaults to `False`, so pass `True` when building prompts for generation.
`response_prefix=` seeds the start of the final response, closing any open reasoning block first.

The `add_generation_reasoning` parameter controls the opening reasoning tag in the generation prompt:

| Value | Effect |
|---|---|
| `None` (default) | Leave the template output unchanged |
| `True` | Make sure the opening tag is present, adding it if needed |
| `False` | Make sure there is no opening tag, removing it if the template added one |

The `add_history_reasoning` parameter controls reasoning on assistant messages *before the last user message*, which some templates (e.g. Qwen3, Qwen3.5, DeepSeek-R1) strip:

| Value | Effect |
|---|---|
| `None` (default) | Embed the reasoning and let the template decide whether to keep it |
| `True` | Always keep the reasoning, even if the template would strip it |
| `False` | Always drop the reasoning |

Reasoning on the final assistant message (after the last user message) is always kept, as it is needed for training.

See [examples/notebooks/apply_chat.ipynb](examples/notebooks/apply_chat.ipynb) for interactive examples.

---

### `thinkpack.parse` — Response parsing

Parse raw model outputs into reasoning and answer components, with flags that classify the reasoning.

```python
# single response
parsed = thinkpack.parse(response=raw_text, tokenizer=tokenizer)
parsed.answer                   # str — text after the closing reasoning tag
parsed.reasoning                # str — content of the reasoning block
parsed.has_valid_reasoning      # bool — non-empty, completed reasoning block (→ VR)
parsed.has_empty_reasoning      # bool — reasoning block opened and closed, but blank
parsed.has_truncated_reasoning  # bool — reasoning block opened but never closed
parsed.has_missing_reasoning    # bool — no reasoning block found at all

# batch of responses, passing the prompts they were generated from
parsed_list = thinkpack.parse(response=responses, tokenizer=tokenizer, prompt=prompts)
```

Handles all four output formats:

| Format | Example |
|---|---|
| Standard | `<think>reasoning</think>answer` |
| Prefixed template | `reasoning</think>answer` (opening tag added by the template) |
| Truncated standard | `<think>reasoning...` (no closing tag) |
| Truncated prefixed | `reasoning...` (detected automatically for prefixed models) |

Recognises tag variants: `think`, `thinking`, `reasoning`, `thought` (case-insensitive).

Pass the generation prompt as `prompt=` so `parse` knows whether the output starts inside an open reasoning block — for example, Qwen3.5 opens `<think>` by default, but closes it in the prompt when called with `enable_thinking=False`.
For prefixed templates, output with no closing tag is classed as truncated, whether generation hit the token limit or the model stopped early.
Pass `calculate_tokens=True` to also count the reasoning and answer tokens.

See [examples/notebooks/parse_and_stats.ipynb](examples/notebooks/parse_and_stats.ipynb) for interactive examples.

---

### `thinkpack.stats` — Response statistics

Aggregates a batch of parsed responses into the metrics used to measure reasoning-trace collapse.

```python
s = thinkpack.compute_stats(
    responses=parsed_list,
    results=correct,  # optional — one bool per response, marking correct answers
)

# all rates are fractions in [0, 1]
s.valid_reasoning_rate      # VR
s.empty_reasoning_rate      # ER
s.truncated_reasoning_rate  # TR
s.missing_reasoning_rate    # MR
s.pass_at_1                 # pass@1 (None without results)
s.rpass_at_1                # Rpass@1 (None without results)
s.total                     # number of responses
```

The rates are also available by their short names: `s.vr`, `s.er`, `s.tr`, and `s.mr`.

| Metric | Definition | Interpretation |
|---|---|---|
| **VR** | `valid_reasoning_rate` | Fraction with structurally valid reasoning (primary structural metric) |
| **ER** | `empty_reasoning_rate` | Fraction with an empty reasoning block (delimiters present, no content) |
| **TR** | `truncated_reasoning_rate` | Fraction where reasoning starts but is never closed |
| **MR** | `missing_reasoning_rate` | Fraction with no reasoning block at all |
| **pass@1** | `pass_at_1` | Standard answer correctness |
| **Rpass@1** | `rpass_at_1` | Accuracy among responses with valid reasoning |

VR, ER, TR, and MR sum to 1.
Reasoning-trace collapse shows up as VR → 0 over training steps or data size.

For nested `[task][sample]` input, all rates are averaged across tasks, so each task counts equally (tasks with no samples are left out).
Rpass@1 is averaged only over tasks with at least one valid-reasoning sample, since it is undefined for the rest.

---

### `thinkpack.mask` — Training-time loss masking

`apply_mask()` tokenizes training conversations into a HuggingFace dataset, with selected sections excluded from the loss.
Masking the think block means the model is not directly trained to produce empty or missing reasoning, which can help preserve its reasoning traces.
How much it helps is model- and task-dependent, so check VR with [`thinkpack.stats`](#thinkpackstats--response-statistics) after training.

```python
import thinkpack

# masking-based SFT — can help mitigate reasoning-trace collapse
dataset = thinkpack.apply_mask(
    conversations=conversations,  # each ending with the assistant message to train on
    tokenizer=tokenizer,
    masked=thinkpack.MaskType.THINK,  # mask the think block from the loss
)

# naive SFT — baseline, can lead to reasoning-trace collapse
naive_dataset = thinkpack.apply_mask(
    conversations=conversations,
    tokenizer=tokenizer,
    masked=None,  # no masking — all tokens contribute to the loss
)
```

The `masked` parameter is a flag, so sections can be combined with `|`:

| Value | Effect |
|---|---|
| `MaskType.THINK` | Think block hidden from the loss; model trains on prompt + response |
| `MaskType.PROMPT \| MaskType.THINK` | Train on the response only |
| `None` | No masking; all tokens contribute to the loss (naive baseline) |

When masking, a final assistant message with no `"reasoning"` key gets an empty think block, matching what the model sees at inference.
The dataset is not padded, so train with a collator that pads `labels` with `-100`, such as `transformers.DataCollatorForSeq2Seq(tokenizer=tokenizer)`.
A fast tokenizer is required.

See [examples/notebooks/loss_masking.ipynb](examples/notebooks/loss_masking.ipynb) for interactive examples.

---

### `thinkpack.distill` — Reasoning distillation

Helpers for adding reasoning traces to instruction–response data, by asking a teacher model to explain how each known answer is reached.

```python
records = [{"instruction": "What is 2 + 2?", "response": "4"}]

# plain-text prompts asking for the reasoning inside <reasoning_steps> tags
prompts = thinkpack.build_prompts(records=records)

# generate with any model, ideally with "</reasoning_steps>" as a stop sequence
responses = ...

# add each extracted trace to its record as "reasoning", then build conversations
records = thinkpack.update_records(records=records, responses=responses)
conversations = thinkpack.to_conversations(records=records)
```

The conversations are ready for `apply_mask()` or `apply_chat_template()`.
Records where no reasoning could be extracted are left without a `"reasoning"` key.

---

## *agent skill*

`thinkpack` ships with an `llms.txt` file and a CLI command to install it as an agent skill in your project.
This gives AI coding assistants (Claude Code, Cursor, Windsurf) accurate context about the library.

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

---

## *citation*

This work was part of the paper [**Reasoning-Trace Collapse: Evaluating the Loss of Explicit Reasoning During Fine-Tuning**](https://arxiv.org/abs/2605.21127).
We hope you find ***`ThinkPack`*** useful! If it helps your research, please consider citing the paper:

**Reference:**

```
Lukas Twist, Helen Yannakoudakis, and Jie M. Zhang. 2026. Reasoning-Trace Collapse: Evaluating the Loss of Explicit Reasoning During Fine-Tuning. In Advances in Neural Information Processing Systems: Evaluations and Datasets Track, Sydney, Australia.
```

**BibTeX:**

```
@inproceedings{twistReasoningCollapse2026,
  title = {{Reasoning-Trace Collapse: Evaluating the Loss of Explicit Reasoning During Fine-Tuning}},
  author = {Twist, Lukas and Yannakoudakis, Helen and Zhang, Jie M.},
  booktitle = {Advances in Neural Information Processing Systems: Evaluations and Datasets Track},
  location = {Sydney, Australia},
  year = {2026},
  month = {December},
  url = {https://arxiv.org/abs/2605.21127},
}
```

---

## *licence*

***`ThinkPack`*** is released under the [MIT License](LICENSE).
The datasets in [`study/data`](study/data) are derived from public benchmarks, and keep the licences of their original sources.
