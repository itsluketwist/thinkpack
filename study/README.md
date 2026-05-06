# ***Reasoning-Trace Collapse: Empirical Study Replication Package***

This package reproduces the experiments from the research paper ***"Reasoning-Trace Collapse: Evaluating the Loss of Explicit Reasoning During Fine-Tuning"***.
It contains the training and evaluation code needed to observe the reasoning collapse phenomenon - where models that reason by default (using `<think>...</think>` blocks) stop reasoning entirely after fine-tuning on standard instruction-response data - and to compare strategies that prevent it.

Two scripts cover the full experimental workflow:

- **`train.py`** — fine-tunes a base model on chemistry QA data using a specified reasoning strategy
- **`evaluate.py`** — evaluates a model (base or fine-tuned) on a dataset and reports accuracy plus reasoning collapse metrics

The ThinkPack library ([companion repository](../)) provides utilities to allow model-agnostic reasoning-aware training and evaluation.

---

## *setup*

**Requirements:** Python 3.11+, CUDA, one or more NVIDIA GPUs with at least 80GB VRAM (for full-precision LoRA).

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

***Note:** The experiments were run on NVIDIA H200 GPUs with Python 3.11 and PyTorch 2.5.1 (CUDA 12.1). Other configurations may work but are untested.*

---

## *datasets*

The `data/` directory contains all datasets used:

| File | Keys | Description |
|------|------|-------------|
| `chemistry_train.jsonl` | `instruction`, `response` | ~2,600 chemistry multiple-choice QA pairs for training — from the [SciKnowEval](https://huggingface.co/datasets/hicai-zju/SciKnowEval) Chemistry L-3 subset |
| `chemistry_256.jsonl` | `prompt`, `answer` | 256 held-out chemistry questions for evaluation |
| `gsm8k_256.jsonl` | `prompt`, `answer` | 256 arithmetic reasoning questions from [GSM8K](https://huggingface.co/datasets/openai/gsm8k) test split |
| `evalplus_256.jsonl` | `prompt`, `entry_point`, `test_code` | 256 code generation problems sampled from [HumanEval+](https://huggingface.co/datasets/evalplus/humanevalplus) and [MBPP+](https://huggingface.co/datasets/evalplus/mbppplus) |

---

## *configuration*

The `config/` directory contains the hyperparameters used in the experiments:

- **`train.yaml`** — LoRA fine-tuning hyperparameters (learning rate, epochs, batch size, LoRA rank, etc.). The `default` profile is used unless overridden with `--profile`.
- **`evaluate.yaml`** — vLLM generation parameters (temperature, max tokens, etc.). The `greedy` profile sets temperature to 0 for deterministic evaluation.

Both files can be edited directly to adjust hyperparameters, or new named profiles can be added and selected with `--profile`.

---

## *training*

Fine-tune a model using `train.py`. The primary variables are the **model**, the **reasoning strategy**, and the **learning rate**.

### *reasoning strategies*

The `--strategy` flag controls how the training data is prepared:

| Strategy | Description |
|----------|-------------|
| `default` | Use the model's native chat template behaviour with no modifications |
| `bare` | Alias for `default`; the name reflects that the model trains on its "bare" template output |
| `empty` | Explicitly inject an empty `<think></think>` block into every training sample, included in the loss |
| `mask` | Inject an empty `<think></think>` block and mask it from the loss |
| `respond` | Inject an empty `<think></think>` block and mask both it and the instruction; training only on response tokens |


### *example commands*

```bash
# baseline — model's default chat template behaviour (collapse expected)
python train.py --model Qwen/Qwen3-8B --strategy default --lr 1e-5

# core prevention method — mask think block from loss
python train.py --model Qwen/Qwen3-8B --strategy mask --lr 1e-5

# alternative prevention method — train on response only
python train.py --model Qwen/Qwen3-8B --strategy respond --lr 1e-5

# train with an empty think block included in loss
python train.py --model Qwen/Qwen3-8B --strategy empty --lr 1e-5
```

Training artefacts (the LoRA adapter) are saved to `output/<model>-<strategy>/adapter/` by default. Pass `--output` to override.

### *all arguments*

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | required | HuggingFace model name or local path |
| `--strategy` | `default` | Reasoning strategy (see table above) |
| `--lr` | `1e-5` | Learning rate — the primary experimental variable |
| `--seed` | `42` | Random seed |
| `--output` | `output/<model>-<strategy>/` | Directory to save the adapter |
| `--profile` | `default` | Hyperparameter profile in `config/train.yaml` |

---

## *evaluation*

Evaluate a model using `evaluate.py`. Pass `--adapter` to load a fine-tuned LoRA adapter on top of the base model; omit it to evaluate the base model directly.

```bash
# evaluate the base model (no fine-tuning)
python evaluate.py --model Qwen/Qwen3-8B --dataset chemistry

# evaluate a fine-tuned adapter
python evaluate.py --model Qwen/Qwen3-8B --adapter output/qwen3-8b-mask/adapter --dataset chemistry

# evaluate on other datasets
python evaluate.py --model Qwen/Qwen3-8B --adapter output/qwen3-8b-mask/adapter --dataset gsm8k
python evaluate.py --model Qwen/Qwen3-8B --adapter output/qwen3-8b-mask/adapter --dataset evalplus

# save results to a file
python evaluate.py --model Qwen/Qwen3-8B --adapter output/qwen3-8b-mask/adapter --dataset chemistry --output results/mask_chemistry.json
```

### *output fields*

| Field | Description |
|-------|-------------|
| `pass_at_1` | Fraction of questions answered correctly |
| `vr` | Valid reasoning rate — fraction of responses with a complete, non-empty `<think>` block |
| `ir` | Invalid reasoning rate — the collapse indicator; `vr + ir = 1` |
| `mr` | Missing reasoning rate — no `<think>` block at all |
| `tr` | Truncated reasoning rate — `<think>` opened but never closed (hit max tokens) |
| `er` | Empty reasoning rate — `<think></think>` present but blank |

### *all arguments*

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | required | HuggingFace model name or local path |
| `--dataset` | required | `chemistry`, `gsm8k`, `evalplus`, or path to a JSONL file |
| `--adapter` | `None` | Path to a LoRA adapter directory |
| `--output` | `None` | Path to save results JSON (prints to stdout if omitted) |
| `--profile` | `greedy` | Generation profile in `config/evaluate.yaml` |

---

## *paper results*

The experiments consist of two parts: a learning rate sweep on the `default` strategy, and a strategy comparison at a fixed learning rate.

### *learning rate sweep (default strategy)*

Three learning rates are tested to show how collapse severity scales with training intensity.

**`qwen3-8b`** (`Qwen/Qwen3-8B`):

```bash
python train.py --model Qwen/Qwen3-8B --strategy default --lr 2e-5
python train.py --model Qwen/Qwen3-8B --strategy default --lr 1e-5
python train.py --model Qwen/Qwen3-8B --strategy default --lr 5e-6
```

**`olmo-3-7b`** (`allenai/OLMo-3-7B-Think`):

```bash
python train.py --model allenai/OLMo-3-7B-Think --strategy default --lr 2e-5
python train.py --model allenai/OLMo-3-7B-Think --strategy default --lr 1e-5
python train.py --model allenai/OLMo-3-7B-Think --strategy default --lr 5e-6
```

### *strategy comparison (lr = 1e-5)*

Four strategies are compared at a fixed learning rate to evaluate which approaches prevent collapse.

**`qwen3-8b`** (`Qwen/Qwen3-8B`):

```bash
python train.py --model Qwen/Qwen3-8B --strategy bare    --lr 1e-5
python train.py --model Qwen/Qwen3-8B --strategy empty   --lr 1e-5
python train.py --model Qwen/Qwen3-8B --strategy mask    --lr 1e-5
python train.py --model Qwen/Qwen3-8B --strategy respond --lr 1e-5
```

**`olmo-3-7b`** (`allenai/OLMo-3-7B-Think`):

```bash
python train.py --model allenai/OLMo-3-7B-Think --strategy bare    --lr 1e-5
python train.py --model allenai/OLMo-3-7B-Think --strategy empty   --lr 1e-5
python train.py --model allenai/OLMo-3-7B-Think --strategy mask    --lr 1e-5
python train.py --model allenai/OLMo-3-7B-Think --strategy respond --lr 1e-5
```

### *evaluation*

In both cases, each saved checkpoint should be evaluated against all three benchmarks.

```bash
python evaluate.py --model Qwen/Qwen3-8B --adapter output/qwen3-8b-mask/adapter --dataset chemistry
python evaluate.py --model Qwen/Qwen3-8B --adapter output/qwen3-8b-mask/adapter --dataset gsm8k
python evaluate.py --model Qwen/Qwen3-8B --adapter output/qwen3-8b-mask/adapter --dataset evalplus
```
