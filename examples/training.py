"""
Example: SFT training with think-block loss masking using ThinkPack.

This script shows exactly where to insert thinkpack.mask() into a standard
training loop. The rest is boilerplate transformers — the ThinkPack line
is marked with a comment.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

import thinkpack

# --- load model and tokenizer ---

model_name = "allenai/OLMo-3-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# --- load training records ---
# each record is a dict with "instruction" and "response" keys;
# an optional "reasoning" key provides think block content

records = [
    {"instruction": "What is 2 + 2?", "response": "4"},
    {"instruction": "Write a haiku about the sea.", "response": "Waves crash on the shore..."},
]

# --- ThinkPack: format records with reasoning tokens masked from the loss ---
# insert this line in place of your existing dataset formatting step.
# template style (INLINE, NATIVE, PREFIXED) is detected automatically.
# combine flags to mask multiple sections, e.g. Mask.PROMPT | Mask.THINK
# to train on the response only.
train_dataset = thinkpack.mask(
    records=records,
    tokenizer=tokenizer,
    masked=thinkpack.Mask.THINK,
)

# --- train ---

trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    args=TrainingArguments(
        output_dir="output/model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
    ),
)

trainer.train()
