"""ThinkPack — a framework for training, parsing, and evaluating explicit reasoning models."""

from thinkpack._model import ModelInfo, TemplateStyle, detect_model
from thinkpack.mask import Mask, mask
from thinkpack.parse import ParsedResponse, parse, parse_all, parse_output
from thinkpack.stats import ResponseStats, stats
from thinkpack.steer import SimplePrefix, apply_steer_template, steer


__all__ = [
    "ModelInfo",
    "TemplateStyle",
    "detect_model",
    "Mask",
    "mask",
    "ParsedResponse",
    "parse",
    "parse_all",
    "parse_output",
    "ResponseStats",
    "stats",
    "SimplePrefix",
    "apply_steer_template",
    "steer",
]
