"""Example: Naive SFT vs masking-based SFT to prevent reasoning collapse.

Shows the single-line ThinkPack change that prevents reasoning collapse
during fine-tuning on standard instruction-response data.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

import thinkpack


# --- load model and tokenizer ---

model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# --- load training conversations ---
# standard instruction-response pairs in chat format; no reasoning field required.
# the model produces <think>...</think> traces at inference, but training data need not include them.

conversations = [
    [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4"},
    ],
    [
        {"role": "user", "content": "Write a haiku about the sea."},
        {"role": "assistant", "content": "Waves crash on the shore..."},
    ],
]

# --- naive SFT (causes reasoning collapse) ---
# all tokens contribute to the loss, including any generated <think> blocks.
# the model learns to skip reasoning, since the response alone minimises loss.
naive_dataset = thinkpack.apply_mask(
    conversations=conversations,
    tokenizer=tokenizer,
    masked=None,  # no masking — naive baseline
)

# --- masking-based SFT (prevents reasoning collapse) ---
# the think block is excluded from the loss; the model is not penalised for reasoning.
# this preserves reasoning behaviour while still training on the response.
# template style (INLINE, NATIVE, PREFIXED) is detected automatically from the tokenizer.
masked_dataset = thinkpack.apply_mask(
    conversations=conversations,
    tokenizer=tokenizer,
    masked=thinkpack.MaskType.THINK,  # mask the think block (core method)
)

# --- train ---
# swap naive_dataset for masked_dataset to compare collapse vs no collapse.
# all other training code is identical — this is the only change.

trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=masked_dataset,
    args=TrainingArguments(
        output_dir="output/model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
    ),
)

trainer.train()
