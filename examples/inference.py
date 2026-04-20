"""Example: Measuring reasoning collapse with ThinkPack.

After generation, parse outputs and compute AR (any reasoning) and VR (valid reasoning)
rates. A model exhibiting reasoning collapse will show VR near zero — it stops producing
valid <think> blocks despite being trained on a reasoning-enabled base model.
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
# apply_steer_template handles template detection automatically.
# pass prefix=None to just ensure <think> is open before generation.
# pass a SimplePrefix or custom string to seed the thought (optional).
prompts = thinkpack.apply_steer_template(
    conversations=conversations,
    tokenizer=tokenizer,
    prefix=None,
)

# --- generate ---

outputs = llm.generate(
    prompts=prompts,
    sampling_params=SamplingParams(
        temperature=0.6,
        max_tokens=2048,
    ),
)

# --- ThinkPack: parse all outputs into reasoning and answer components ---
# parse_output accepts the raw vLLM output list directly.
parsed = thinkpack.parse_output(output=outputs)

# --- ThinkPack: compute AR and VR rates to measure reasoning collapse ---
# AR (any reasoning): fraction of responses with any reasoning block present.
# VR (valid reasoning): fraction with a complete, non-blank reasoning block.
# collapse is observable as VR -> 0 over training steps or data size.
s = thinkpack.stats(responses=parsed)
print(f"total responses:      {s.total}")
print(f"AR (any reasoning):   {s.has_reasoning_block / s.total:.2%}")
print(f"VR (valid reasoning): {s.has_valid_reasoning / s.total:.2%}")
print(f"has answer:           {s.has_answer / s.total:.2%}")
print()

# inspect individual responses
for task_parsed in parsed:
    # task_parsed is a list with one entry per sample (n=1 here, so index 0)
    p = task_parsed[0]
    print(f"answer:              {p.answer}")
    print(f"has_valid_reasoning: {p.has_valid_reasoning}")
    print()
