"""Example: Naive SFT vs masking-based SFT to mitigate reasoning collapse.

Shows the single-line ThinkPack change that can help mitigate reasoning collapse
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

# --- naive SFT (can cause reasoning collapse) ---
# all tokens contribute to the loss, including any generated <think> blocks.
# the model can learn to skip reasoning, since the response alone minimises loss.
naive_dataset = thinkpack.apply_mask(
    conversations=conversations,
    tokenizer=tokenizer,
    masked=None,  # no masking — naive baseline
)

# --- masking-based SFT (can help mitigate reasoning collapse) ---
# the think block is excluded from the loss; the model is not penalised for reasoning.
# this can help preserve reasoning behaviour, though the effect is model- and task-dependent.
# template style (INLINE, NATIVE, PREFIXED) is detected automatically from the tokenizer.
masked_dataset = thinkpack.apply_mask(
    conversations=conversations,
    tokenizer=tokenizer,
    masked=thinkpack.MaskType.THINK,  # mask the think block from the loss
)

# --- train ---
# swap between naive_dataset and masked_dataset to compare their effect on collapse.
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
