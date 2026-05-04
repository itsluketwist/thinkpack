"""Example: Measuring reasoning collapse with ThinkPack.

After generation, parse outputs and compute VR (valid reasoning) and IR (invalid reasoning)
rates. A model exhibiting reasoning collapse will show VR near zero and IR near one — it
stops producing valid <think> blocks despite being trained on a reasoning-enabled base model.
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
# apply_chat_templates handles template detection automatically.
# think_prefix=None (default) leaves thought seeding to the model; add_generation_reasoning=True
# (default) ensures the <think> tag is open before generation starts.
prompts = thinkpack.apply_chat_templates(
    conversations=conversations,
    tokenizer=tokenizer,
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
# extract the generated text from each vLLM RequestOutput: one list per conversation,
# one string per sample (n=1 here, so each inner list has a single entry).
texts = [[c.text for c in o.outputs] for o in outputs]
parsed = thinkpack.parse(
    response=texts,
    tokenizer=tokenizer,
)

# --- ThinkPack: compute VR and IR rates to measure reasoning collapse ---
# VR (valid reasoning): fraction with a complete, non-blank reasoning block.
# IR (invalid reasoning): fraction where reasoning was absent, truncated, or empty.
# collapse is observable as VR -> 0 (and IR -> 1) over training steps or data size.
# all rate fields in ResponseStats are already fractions in [0, 1].
s = thinkpack.compute_stats(responses=parsed)
print(f"total responses:         {s.total}")
print(f"VR (valid reasoning):    {s.vr:.2%}")
print(f"IR (invalid reasoning):  {s.invalid_reasoning_rate:.2%}")
print(f"  missing:               {s.mr:.2%}")
print(f"  empty:                 {s.er:.2%}")
print(f"  truncated:             {s.tr:.2%}")
print(f"has answer:              {s.answer_rate:.2%}")
print()

# inspect individual responses
for task_parsed in parsed:
    # task_parsed is a list with one entry per sample (n=1 here, so index 0)
    p = task_parsed[0]
    print(f"answer:              {p.answer}")
    print(f"has_valid_reasoning: {p.has_valid_reasoning}")
    print()
