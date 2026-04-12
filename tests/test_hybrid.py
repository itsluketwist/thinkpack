"""Tests for thinkpack.hybrid — hybrid decoding with base model reasoning."""

from dataclasses import dataclass
from typing import Any

from thinkpack.hybrid import HybridResult, hybrid_generate


@dataclass
class MockCompletion:
    """Simulates a single vLLM completion output."""

    text: str
    finish_reason: str = "stop"


@dataclass
class MockRequestOutput:
    """Simulates a vLLM RequestOutput (one per prompt)."""

    outputs: list[MockCompletion]


class MockLLM:
    """Mock vLLM LLM that records which lora_request each generate() call received."""

    def __init__(
        self,
        phase1_texts: list[str],
        phase2_texts: list[str],
    ) -> None:
        # responses returned in order: first call = phase 1, second = phase 2
        self._responses = [phase1_texts, phase2_texts]
        self._call_index = 0
        # records (prompts, lora_request) for each generate() call
        self.calls: list[tuple[list[str], Any]] = []

    def generate(
        self,
        prompts: list[str],
        sampling_params: Any,
        lora_request: Any = None,
    ) -> list[MockRequestOutput]:
        """Record the call and return the next batch of mock responses."""
        self.calls.append((prompts, lora_request))
        texts = self._responses[self._call_index]
        self._call_index += 1
        return [MockRequestOutput(outputs=[MockCompletion(text=t)]) for t in texts]


class TestHybridGenerate:
    """Tests for hybrid_generate()."""

    def test_phase1_uses_no_adapter(self) -> None:
        """Phase 1 generate() call must have lora_request=None."""
        lora = object()  # sentinel for the adapter
        llm = MockLLM(
            phase1_texts=["<think>\nsome reasoning\n</think>\nthe answer"],
            phase2_texts=["final answer"],
        )
        hybrid_generate(
            prompts=["prompt"],
            llm=llm,
            lora_request=lora,
            sampling_params=None,
        )

        _, phase1_lora = llm.calls[0]
        assert phase1_lora is None

    def test_phase2_uses_adapter(self) -> None:
        """Phase 2 generate() call must have the provided lora_request."""
        lora = object()
        llm = MockLLM(
            phase1_texts=["<think>\nsome reasoning\n</think>\nthe answer"],
            phase2_texts=["final answer"],
        )
        hybrid_generate(
            prompts=["prompt"],
            llm=llm,
            lora_request=lora,
            sampling_params=None,
        )

        _, phase2_lora = llm.calls[1]
        assert phase2_lora is lora

    def test_reasoning_prepended_in_phase2_prompt(self) -> None:
        """Phase 2 prompt contains the reasoning block extracted from phase 1."""
        llm = MockLLM(
            phase1_texts=["<think>\nmy reasoning\n</think>\nsome answer"],
            phase2_texts=["final answer"],
        )
        hybrid_generate(
            prompts=["original prompt"],
            llm=llm,
            lora_request=None,
            sampling_params=None,
        )

        phase2_prompts, _ = llm.calls[1]
        assert "my reasoning" in phase2_prompts[0]
        assert phase2_prompts[0].startswith("original prompt")

    def test_result_fields(self) -> None:
        """HybridResult contains correct reasoning, answer, and raw fields."""
        llm = MockLLM(
            phase1_texts=["<think>\nmy reasoning\n</think>\nignored"],
            phase2_texts=["final answer"],
        )
        results = hybrid_generate(
            prompts=["prompt"],
            llm=llm,
            lora_request=None,
            sampling_params=None,
        )

        assert len(results) == 1
        r = results[0]
        assert isinstance(r, HybridResult)
        assert "my reasoning" in r.reasoning
        assert r.answer == "final answer"
        assert "my reasoning" in r.raw
        assert "final answer" in r.raw

    def test_no_reasoning_in_phase1(self) -> None:
        """If phase 1 produces no reasoning, phase 2 prompt is unmodified."""
        llm = MockLLM(
            phase1_texts=["plain response with no think tags"],
            phase2_texts=["final answer"],
        )
        results = hybrid_generate(
            prompts=["original prompt"],
            llm=llm,
            lora_request=None,
            sampling_params=None,
        )

        phase2_prompts, _ = llm.calls[1]
        # prompt should be unchanged — no reasoning prefix added
        assert phase2_prompts[0] == "original prompt"
        assert results[0].reasoning == ""
        assert results[0].raw == "final answer"

    def test_multiple_prompts(self) -> None:
        """Multiple prompts are handled correctly across both phases."""
        llm = MockLLM(
            phase1_texts=[
                "<think>\nreasoning one\n</think>\nanswer",
                "<think>\nreasoning two\n</think>\nanswer",
            ],
            phase2_texts=["answer one", "answer two"],
        )
        results = hybrid_generate(
            prompts=["prompt one", "prompt two"],
            llm=llm,
            lora_request=None,
            sampling_params=None,
        )

        assert len(results) == 2
        assert "reasoning one" in results[0].reasoning
        assert "reasoning two" in results[1].reasoning
