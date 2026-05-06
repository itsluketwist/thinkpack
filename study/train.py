"""Fine-tune a language model on chemistry QA data with a configurable reasoning strategy."""

import unsloth  # noqa: F401 # isort: skip

import argparse
from pathlib import Path

import torch
import yaml
from llm_cgr import load_jsonl
from transformers import (
    DataCollatorForSeq2Seq,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

import thinkpack


# paths relative to this script
_CONFIG_FILE = Path(__file__).parent / "config" / "train.yaml"
_DEFAULT_DATA_FILE = Path(__file__).parent / "data" / "chemistry_train.jsonl"


class _CheckpointCallback(TrainerCallback):
    """Save the LoRA adapter to a numbered subdirectory at regular step intervals."""

    def __init__(self, model: object, save_steps: int, output_dir: Path) -> None:
        self._model = model
        self._save_steps = save_steps
        self._output_dir = output_dir

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        if state.global_step % self._save_steps == 0:
            path = self._output_dir / "checkpoints" / f"step_{state.global_step}"
            self._model.save_pretrained(
                str(path),
                save_in_float16=False,
                save_in_bfloat16=True,
            )
            print(f"checkpoint saved: {path}")


def _load_config(profile: str) -> dict:
    """Load hyperparameters from training.yaml, merging the named profile on top of defaults.

    Returns a flat dict of hyperparameter values.
    """
    with open(_CONFIG_FILE) as f:
        config = yaml.safe_load(f)
    result = dict(config.get("default", {}))
    if profile != "default":
        result.update(config.get(profile, {}))
    return result


def _build_conversations(records: list[dict], strategy: str) -> list[list[dict]]:
    """Convert instruction/response records into conversation dicts for ThinkPack.

    If a record contains a 'reasoning' field, it is included in the assistant message
    so ThinkPack uses it as the think block content. For the empty strategy, an empty
    reasoning key is added explicitly so the model trains on empty <think></think> tokens.

    Returns a list of two-turn conversations in chat format.
    """
    conversations = []
    for r in records:
        assistant: dict = {"role": "assistant", "content": r["response"]}
        if "reasoning" in r:
            # use the distilled reasoning trace from the data (e.g. from a construct dataset)
            assistant["reasoning"] = r["reasoning"]
        elif strategy == "empty":
            # inject an empty reasoning key so think tokens appear in the training sequence
            assistant["reasoning"] = ""
        conversations.append(
            [
                {"role": "user", "content": r["instruction"]},
                assistant,
            ]
        )
    return conversations


def main() -> None:
    """Run fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a model on chemistry QA with a reasoning strategy.",
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument(
        "--strategy",
        default="default",
        choices=["default", "bare", "empty", "mask", "respond"],
        help=(
            "reasoning strategy: "
            "default/bare = use the model's native chat template behaviour (Qwen3 emits empty "
            "<think></think>, OLMo emits <think> prefix — collapse is model-dependent); "
            "empty = inject an empty <think></think> block and train on it with full loss; "
            "mask = inject an empty think block and mask it from loss (prevents collapse); "
            "respond = mask think block + prompt, train only on response tokens"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output directory (default: output/<model>-<strategy>/)",
    )
    parser.add_argument(
        "--profile", default="default", help="profile in config/training.yaml"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate (primary experimental variable)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="path to training JSONL (default: data/chemistry_train.jsonl); if records include a 'reasoning' field it will be used as the think block",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help="save a lora checkpoint every N steps (0 = final adapter only)",
    )
    args = parser.parse_args()

    # load config and set the primary experimental variables from args
    config = _load_config(args.profile)
    config["learning_rate"] = args.lr
    config["seed"] = args.seed

    # derive output directory from model name + strategy if not explicitly specified
    model_slug = args.model.split("/")[-1].lower()
    output_dir = (
        Path(args.output)
        if args.output
        else Path("output") / f"{model_slug}-{args.strategy}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"model:     {args.model}")
    print(f"strategy:  {args.strategy}")
    print(f"output:    {output_dir}")
    print(f"lr:        {config['learning_rate']}")
    print(f"seed:      {config['seed']}")

    # olmo3 requires sdpa attention — flex_attention breaks unsloth on this model family
    load_kwargs: dict = {}
    if "olmo" in args.model.lower():
        load_kwargs["attn_implementation"] = "sdpa"

    # disable dynamo — unsloth patches are not dynamo-traceable
    torch._dynamo.config.disable = True

    # load base model and tokenizer via unsloth (bf16 precision, optional 4-bit quantisation)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=config.get("max_seq_length", 32768),
        load_in_4bit=config.get("load_in_4bit", True),
        dtype=torch.bfloat16,
        **load_kwargs,
    )

    # ensure the tokenizer has a pad token — required by the data collator
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # apply lora adapters — only these small weight matrices are trained, not the full model
    lora = config.get("lora", {})
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora.get("r", 16),
        lora_alpha=lora.get("alpha", 16),
        lora_dropout=lora.get("dropout", 0),
        target_modules=lora.get(
            "target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )

    # map strategy name to the thinkpack MaskType argument.
    # default/bare: no masking, no think block injected — the model's native template decides
    #   whether a think block appears (e.g. Qwen3 emits empty <think></think>, OLMo emits
    #   an opening <think>). collapse behaviour is model-dependent.
    # empty: no masking, but empty reasoning key added in _build_conversations so the empty
    #   think block tokens are present in the training sequence and contribute to loss.
    # mask: mask the think block from loss (the core method for preventing collapse).
    # respond: mask think block + prompt, training only on response tokens.
    strategy_mask = {
        "default": None,
        "bare": None,
        "empty": None,
        "mask": thinkpack.MaskType.THINK,
        "respond": thinkpack.MaskType.THINK | thinkpack.MaskType.PROMPT,
    }[args.strategy]

    # load training data and build conversation dicts
    data_file = Path(args.data) if args.data else _DEFAULT_DATA_FILE
    records = load_jsonl(data_file)
    conversations = _build_conversations(records=records, strategy=args.strategy)

    # thinkpack tokenizes conversations and applies label masking according to the strategy.
    # for mask/respond, it automatically injects an empty reasoning key into conversations
    # that lack one, ensuring training context matches what the model sees at inference time.
    train_dataset = thinkpack.apply_mask(
        conversations=conversations,
        tokenizer=tokenizer,
        masked=strategy_mask,
        max_seq_length=config.get("max_seq_length", 32768),
    )

    print(f"training samples: {len(train_dataset)}")

    # attach checkpoint callback if requested — saves adapter to checkpoints/step_N/ at each interval
    callbacks = []
    if args.save_steps > 0:
        callbacks.append(
            _CheckpointCallback(
                model=model,
                save_steps=args.save_steps,
                output_dir=output_dir,
            )
        )

    # sft trainer with pre-tokenized dataset (skip_prepare_dataset avoids re-tokenisation errors)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        callbacks=callbacks,
        packing=False,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        ),
        args=SFTConfig(
            output_dir=str(output_dir),
            seed=config.get("seed", 42),
            num_train_epochs=config.get("num_epochs", 3),
            per_device_train_batch_size=config.get("batch_size", 2),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 2),
            learning_rate=float(config["learning_rate"]),
            warmup_ratio=config.get("warmup_ratio", 0.1),
            weight_decay=config.get("weight_decay", 0.01),
            logging_steps=config.get("logging_steps", 10),
            save_strategy="no",
            bf16=config.get("bf16", True),
            max_seq_length=config.get("max_seq_length", 32768),
            report_to="none",
            dataset_kwargs={"skip_prepare_dataset": True},
        ),
    )

    trainer.train()

    # save only the lora adapter in bf16 (the small weight matrices, ~75MB)
    adapter_path = output_dir / "adapter"
    model.save_pretrained(
        str(adapter_path), save_in_float16=False, save_in_bfloat16=True
    )
    tokenizer.save_pretrained(str(adapter_path))
    print(f"adapter saved to {adapter_path}")


if __name__ == "__main__":
    main()
