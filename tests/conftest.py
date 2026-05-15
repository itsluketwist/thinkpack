"""Pytest configuration and shared fixtures."""

import os

import pytest


# suppress transformers logging during tests
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --no-slow flag to skip slow tests that download real tokenizers."""
    parser.addoption(
        "--no-slow",
        action="store_true",
        default=False,
        help="skip slow tests that require downloading real HuggingFace tokenizers",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the slow marker."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (require HuggingFace tokenizer downloads). "
        "Skip with --no-slow.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip slow tests if --no-slow is passed."""
    if not config.getoption("--no-slow"):
        return
    skip_slow = pytest.mark.skip(reason="skipped by --no-slow")
    for item in items:
        if item.get_closest_marker("slow"):
            item.add_marker(skip_slow)


# ---------------------------------------------------------------------------
# real tokenizer fixtures — downloaded once per session and shared across all
# test files. each fixture is INLINE or PREFIXED as noted.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def qwen3_tokenizer():
    """Qwen/Qwen3-8B — not prefixed, <think> tags."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B",
        trust_remote_code=True,
    )


@pytest.fixture(scope="session")
def qwen35_tokenizer():
    """Qwen/Qwen3.5-9B — prefixed, <think> tags."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-9B",
        trust_remote_code=True,
    )


@pytest.fixture(scope="session")
def deepseek_r1_llama_tokenizer():
    """deepseek-ai/DeepSeek-R1-Distill-Llama-8B — prefixed, <think> tags, strips history think blocks."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        trust_remote_code=True,
    )


@pytest.fixture(scope="session")
def olmo3_tokenizer():
    """allenai/OLMo-3-7B-Think —  — prefixed, <think> tags."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        "allenai/OLMo-3-7B-Think",
        trust_remote_code=True,
    )


@pytest.fixture(scope="session")
def ministral_tokenizer():
    """mistralai/Ministral-3B-Instruct-2410 — not prefixed, [THINK] tags."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        "mistralai/Ministral-3-3B-Reasoning-2512",
        trust_remote_code=True,
    )
