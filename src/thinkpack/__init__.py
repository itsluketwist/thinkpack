"""ThinkPack — a framework for training, parsing, and evaluating explicit reasoning models."""

import logging

from thinkpack.chat import apply_chat_template, apply_chat_templates
from thinkpack.mask import MaskType, mask
from thinkpack.model import ModelInfo, TagStyle, detect_model, get_model_info
from thinkpack.parse import ParsedResponse, parse, parse_all, parse_output
from thinkpack.stats import ResponseStats, compute_stats


# standard library logging hygiene — lets users opt in by configuring their own handler
logging.getLogger("thinkpack").addHandler(logging.NullHandler())


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
    "apply_chat_templates",
]
