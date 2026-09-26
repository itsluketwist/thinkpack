"""Example: naive SFT vs masking-based SFT, to mitigate reasoning-trace collapse.

Shows the one-argument ThinkPack change that can help preserve reasoning traces when
fine-tuning on standard instruction-response data.
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

import thinkpack


# --- load model and tokenizer ---

model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# --- load training conversations ---
# standard instruction-response pairs in chat format, with no reasoning.
# the model produces <think>...</think> blocks at inference, but the data has none.

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

# --- naive SFT (can cause reasoning-trace collapse) ---
# all tokens contribute to the loss, including any empty think block the template adds.
# the model can learn to skip reasoning, as the training data never contains any.
naive_dataset = thinkpack.apply_mask(
    conversations=conversations,
    tokenizer=tokenizer,
    masked=None,  # no masking — naive baseline
)

# --- masking-based SFT (can help mitigate reasoning-trace collapse) ---
# an empty think block is added to each response and masked from the loss, so the
# model is not trained to produce empty reasoning. the effect is model- and
# task-dependent, so measure VR after training (see inference.py).
masked_dataset = thinkpack.apply_mask(
    conversations=conversations,
    tokenizer=tokenizer,
    masked=thinkpack.MaskType.THINK,  # mask the think block from the loss
)

# --- train ---
# swap between naive_dataset and masked_dataset to compare their effect on collapse.
# all other training code is identical.

trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=masked_dataset,
    # pads each batch, with padded labels set to -100 so they are ignored by the loss
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
    ),
    args=TrainingArguments(
        output_dir="output/model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
    ),
)

trainer.train()
