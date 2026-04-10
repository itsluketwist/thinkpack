"""Example: Inference with thought-steering and response parsing using ThinkPack.

This script shows exactly where to insert thinkpack functions into a standard
vLLM inference loop. The ThinkPack lines are marked with comments.
"""

from vllm import LLM, SamplingParams

import thinkpack

# --- load model ---

llm = LLM(model="allenai/OLMo-3-1B")
tokenizer = llm.get_tokenizer()

# --- prepare conversations ---

conversations = [
    [{"role": "user", "content": "What is the time complexity of quicksort?"}],
    [{"role": "user", "content": "Explain gradient descent in one paragraph."}],
]

# --- ThinkPack: apply chat template and inject a steering prefix in one step ---
# replaces the usual tokenizer.apply_chat_template() call.
# template style (INLINE, NATIVE, PREFIXED) is detected automatically.
# pass prefix=None to just ensure <think> is open, or a string to seed the thought.
steered_prompts = thinkpack.apply_steer_template(
    conversations=conversations,
    tokenizer=tokenizer,
    prefix=thinkpack.SimplePrefix.CONCISE,
)

# --- generate ---

outputs = llm.generate(
    prompts=steered_prompts,
    sampling_params=SamplingParams(
        temperature=0.6,
        max_tokens=2048,
    ),
)

# --- ThinkPack: parse all outputs into reasoning and answer components ---
# insert this line immediately after generation — accepts the output list directly.
parsed = thinkpack.parse_output(output=outputs)

for task_parsed in parsed:
    # task_parsed is a list with one entry per sample (n=1 here, so index 0)
    p = task_parsed[0]
    print(f"answer:              {p.answer}")
    print(f"has_valid_reasoning: {p.has_valid_reasoning}")
    print()
