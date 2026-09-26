"""Example: measuring reasoning-trace collapse with ThinkPack.

Generates responses, parses them into reasoning and answer, and computes the collapse
metrics. A model that has collapsed shows VR (valid reasoning) near zero, as it stops
producing valid <think> blocks.
"""

from vllm import LLM, SamplingParams

import thinkpack


# --- load model ---

llm = LLM(model="Qwen/Qwen3-8B")
tokenizer = llm.get_tokenizer()

# --- prepare conversations ---

conversations = [
    [{"role": "user", "content": "What is the time complexity of quicksort?"}],
    [{"role": "user", "content": "Explain gradient descent in one paragraph."}],
]

# --- apply chat template ---
# the model's reasoning format is detected from the tokenizer.
# add_generation_prompt=True opens the assistant turn, ready for generation.
prompts = thinkpack.apply_chat_templates(
    conversations=conversations,
    tokenizer=tokenizer,
    add_generation_prompt=True,
)

# --- generate ---

outputs = llm.generate(
    prompts=prompts,
    sampling_params=SamplingParams(
        temperature=0.6,
        max_tokens=2048,
    ),
)

# --- parse outputs into reasoning and answer ---
# vllm returns one list of samples per prompt (n=1 here, so one string per list).
# passing the prompts lets parse() see whether each output starts inside a think block.
texts = [[c.text for c in o.outputs] for o in outputs]
parsed = thinkpack.parse(
    response=texts,
    tokenizer=tokenizer,
    prompt=prompts,
)

# --- compute the collapse metrics ---
# all rates are fractions in [0, 1], and VR + MR + ER + TR = 1.
# collapse shows up as VR -> 0 over training steps or data size.
s = thinkpack.compute_stats(responses=parsed)
print(f"total responses:  {s.total}")
print(f"VR (valid):       {s.vr:.2%}")
print(f"MR (missing):     {s.mr:.2%}")
print(f"ER (empty):       {s.er:.2%}")
print(f"TR (truncated):   {s.tr:.2%}")
print(f"has answer:       {s.answer_rate:.2%}")
print()

# inspect individual responses
for task_parsed in parsed:
    # each task has one entry per sample (n=1 here, so index 0)
    p = task_parsed[0]
    print(f"answer:              {p.answer}")
    print(f"has_valid_reasoning: {p.has_valid_reasoning}")
    print()
