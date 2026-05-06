"""Evaluate a model on chemistry, gsm8k, or evalplus using vLLM and ThinkPack."""

import argparse
import atexit
import json
import re
import shutil
import tempfile
from pathlib import Path

import torch
import yaml
from llm_cgr import load_jsonl, save_json
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import thinkpack


# paths relative to this script
_CONFIG_FILE = Path(__file__).parent / "config" / "evaluate.yaml"
_DATA_DIR = Path(__file__).parent / "data"

# shorthand dataset names used in the paper
_DATASETS = {
    "chemistry": _DATA_DIR / "chemistry_256.jsonl",
    "gsm8k": _DATA_DIR / "gsm8k_256.jsonl",
    "evalplus": _DATA_DIR / "evalplus_256.jsonl",
}

# models that need cpu-side lora merging before vllm inference.
# vllm 0.7.3 lacks native lora support for qwen3 and olmo3.
_MERGE_PREFIXES = ("qwen/qwen3", "allenai/olmo")

# olmo-3 uses sliding window attention, which is incompatible with vllm prefix caching
_NO_PREFIX_CACHE_PREFIXES = ("allenai/olmo-3",)

# regex patterns for extracting multiple-choice letter answers.
# applied to the uppercased full response text in order — first match wins.
_FULL_PATTERNS = [
    r"(?:ANSWER|CHOICE)\s*(?:IS|:)\s*([A-Z])\b",  # "the answer is A" or "answer: A"
    r"\\BOXED\{([A-Z])\}",  # \boxed{A}
    r"\b([A-Z])\s*[.\):]?\s*$",  # standalone letter at end of text
]

# tail patterns applied to only the last 200 characters to reduce false positives
_TAIL_PATTERNS = [
    r"\b([A-Z])[.:\)\]]\s+\S+.*$",  # "D. some text" or "D: some text"
    r"[\(\[]([A-Z])[\)\]]",  # "(A)" or "[A]"
]


def _load_config(profile: str) -> dict:
    """Load and return inference parameters from inference.yaml for the given profile.

    Returns a flat dict of generation parameters.
    """
    with open(_CONFIG_FILE) as f:
        config = yaml.safe_load(f)
    return dict(config.get(profile, {}))


def _extract_answer(text: str) -> str | None:
    """Extract a multiple-choice letter (A–Z) from a model response.

    Tries full-text patterns first, then tail-only patterns to reduce false positives.

    Returns the extracted letter (uppercased), or None if no match found.
    """
    upper = text.upper()
    tail = upper[-200:] if len(upper) > 200 else upper
    for pattern in _FULL_PATTERNS:
        if m := re.search(pattern, upper):
            return m.group(1)
    for pattern in _TAIL_PATTERNS:
        if m := re.search(pattern, tail):
            return m.group(1)
    return None


def _extract_code(text: str) -> str:
    """Extract Python code from a model response, stripping markdown fences if present.

    Returns the code block content if fenced, otherwise the full response text.
    """
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return match.group(1) if match else text


def _evaluate_code(answer: str, entry_point: str, test_code: str) -> bool:
    """Execute generated code against the provided test suite.

    Runs the model's answer and the test_code in a shared namespace, then calls
    check(entry_point_fn). A passing result returns True; any exception returns False.

    Returns True if all test cases pass, False otherwise.
    """
    code = _extract_code(answer)
    try:
        namespace: dict = {}
        exec(code, namespace)  # noqa: S102
        exec(test_code, namespace)  # noqa: S102
        namespace["check"](namespace[entry_point])
        return True
    except Exception:
        return False


def _needs_lora_merge(model_path: str) -> bool:
    """Check whether this model requires cpu-side lora merging before vllm inference.

    Returns True for model families where vllm 0.7.3 lacks native lora support.
    """
    return any(model_path.lower().startswith(p) for p in _MERGE_PREFIXES)


def _patch_olmo3_config() -> None:
    """Add the missing head_dim property to Olmo3Config for vllm 0.7.x compatibility.

    vllm accesses config.head_dim, but Olmo3Config does not define it; it can be
    derived from hidden_size // num_attention_heads.
    """
    try:
        from transformers import Olmo3Config

        if not hasattr(Olmo3Config, "head_dim"):
            Olmo3Config.head_dim = property(  # type: ignore[attr-defined]
                lambda self: self.hidden_size // self.num_attention_heads,
            )
    except ImportError:
        pass


def _merge_adapter(model_path: str, adapter_path: str) -> str:
    """Merge a lora adapter into the base model on cpu, saving to a temp directory.

    Qwen3 and OLMo3 require this because vllm 0.7.3 does not support native lora for
    these model families. The merged model is automatically cleaned up when the process
    exits, so only the small adapter (~75MB) is kept on disk permanently.

    Returns the path to the temporary merged model directory.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("  merging lora adapter into base model on cpu...")
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()

    # save to a temp dir — vllm needs files on disk; cleaned up on process exit
    tmp = tempfile.mkdtemp(prefix="study_merged_")
    merged.save_pretrained(tmp)
    AutoTokenizer.from_pretrained(adapter_path).save_pretrained(tmp)
    atexit.register(shutil.rmtree, tmp, True)

    del merged, base
    print(f"  merged model ready at {tmp}")
    return tmp


def main() -> None:
    """Run evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a model on a chemistry/gsm8k/evalplus dataset.",
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument(
        "--dataset",
        required=True,
        help="dataset name (chemistry, gsm8k, evalplus) or path to a JSONL file",
    )
    parser.add_argument(
        "--adapter", default=None, help="path to lora adapter directory"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="path to save results as json (default: print to stdout)",
    )
    parser.add_argument(
        "--profile", default="greedy", help="profile in config/inference.yaml"
    )
    args = parser.parse_args()

    config = _load_config(args.profile)

    # resolve dataset path: shorthand name or direct file path
    dataset_path = _DATASETS.get(args.dataset, Path(args.dataset))
    print(f"dataset:  {dataset_path}")
    records = load_jsonl(dataset_path)

    # build single-turn user conversations from prompt records
    conversations = [[{"role": "user", "content": r["prompt"]}] for r in records]

    # handle lora adapter loading — some models require cpu-side merging
    model_path = args.model
    lora_request = None
    use_native_lora = False

    _patch_olmo3_config()

    if args.adapter is not None:
        if _needs_lora_merge(args.model):
            # qwen3 and olmo3: merge adapter into base model before loading into vllm
            model_path = _merge_adapter(args.model, args.adapter)
        else:
            # other models: use vllm's native lora support
            use_native_lora = True
            lora_request = LoRARequest(
                lora_name="adapter",
                lora_int_id=1,
                lora_local_path=args.adapter,
            )

    # detect available gpus for tensor parallel inference
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    else:
        n_gpus = 1

    # olmo-3 uses sliding window attention, incompatible with vllm prefix caching
    path_lower = args.model.lower()
    use_prefix_caching = not any(
        path_lower.startswith(p) for p in _NO_PREFIX_CACHE_PREFIXES
    )

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        enable_prefix_caching=use_prefix_caching,
        enable_lora=use_native_lora,
        tensor_parallel_size=n_gpus,
        max_model_len=config.get("max_tokens", 32768),
        max_num_seqs=32,
        gpu_memory_utilization=0.82,
    )
    tokenizer = llm.get_tokenizer()

    # apply chat template using the model's default behaviour
    prompts = thinkpack.apply_chat_templates(
        conversations=conversations,
        tokenizer=tokenizer,
        add_generation_reasoning=None,
    )

    sampling_params = SamplingParams(
        temperature=config.get("temperature", 0.0),
        top_p=config.get("top_p", 1.0),
        top_k=config.get("top_k", -1),
        max_tokens=config.get("max_tokens", 32768),
        min_tokens=config.get("min_tokens", 1),
        seed=config.get("seed", 42),
        n=config.get("samples", 1),
    )

    # generate responses — lora_request is None for base model evaluation
    outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )

    # extract generated text — one list per prompt, one string per sample (n=1 for greedy)
    texts = [[c.text for c in o.outputs] for o in outputs]

    # parse each response into reasoning and answer components.
    # thinkpack handles <think>...</think> extraction for all supported model families.
    parsed = thinkpack.parse(response=texts, tokenizer=tokenizer)

    # detect dataset type from the first record — code datasets have entry_point + test_code
    is_code = "entry_point" in records[0] and "test_code" in records[0]

    # evaluate each response — code: execute against test suite; other: extract letter answer.
    # only the answer section (after </think>) is evaluated in both cases.
    results = []
    for task_parsed, record in zip(parsed, records):
        # use first sample (greedy decoding produces n=1)
        p = task_parsed[0]
        if is_code:
            correct = _evaluate_code(
                answer=p.answer,
                entry_point=record["entry_point"],
                test_code=record["test_code"],
            )
        else:
            expected = record["answer"].strip().upper()
            extracted = _extract_answer(p.answer) if p.answer.strip() else None
            correct = extracted == expected if extracted is not None else False
        results.append(correct)

    # compute reasoning statistics — the key collapse metrics for the paper.
    # results is passed as [[bool]] (one bool per sample, one sample per task).
    stats = thinkpack.compute_stats(parsed, results=[[r] for r in results])

    output = {
        "model": args.model,
        "adapter": args.adapter,
        "dataset": args.dataset,
        "seed": config.get("seed", 42),
        "total": len(records),
        "pass_at_1": stats.pass_at_1,  # fraction of questions answered correctly
        "vr": stats.vr,  # valid reasoning rate: fraction with complete <think> blocks
        "ir": stats.ir,  # invalid reasoning rate: collapse indicator (vr + ir = 1)
        "mr": stats.mr,  # missing reasoning rate: no <think> block at all
        "tr": stats.tr,  # truncated reasoning rate: <think> opened but not closed
        "er": stats.er,  # empty reasoning rate: <think></think> present but blank
    }

    if args.output:
        save_json(output, args.output)
        print(f"results saved to {args.output}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
