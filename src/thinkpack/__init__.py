"""ThinkPack — a framework for training, parsing, and evaluating explicit reasoning models."""

from thinkpack.chat import apply_chat_template
from thinkpack.mask import MaskType, mask
from thinkpack.model import ModelInfo, TagStyle, detect_model, get_model_info
from thinkpack.parse import ParsedResponse, parse, parse_all, parse_output
from thinkpack.stats import ResponseStats, compute_stats


__all__ = [
    "ModelInfo",
    "TagStyle",
    "detect_model",
    "get_model_info",
    "MaskType",
    "mask",
    "ParsedResponse",
    "parse",
    "parse_all",
    "parse_output",
    "ResponseStats",
    "compute_stats",
    "apply_chat_template",
]
