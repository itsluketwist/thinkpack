"""ThinkPack — tools for preventing think collapse in reasoning language models."""

from thinkpack._model import ModelInfo, TemplateStyle, detect_model
from thinkpack.distill import build_prompts, extract_reasoning, update_records
from thinkpack.hybrid import HybridResult, hybrid_generate
from thinkpack.mask import Mask, mask
from thinkpack.parse import ParsedResponse, parse, parse_all, parse_output
from thinkpack.stats import ResponseStats, stats
from thinkpack.steer import SimplePrefix, apply_steer_template, steer


__all__ = [
    "ModelInfo",
    "TemplateStyle",
    "detect_model",
    "build_prompts",
    "extract_reasoning",
    "update_records",
    "HybridResult",
    "hybrid_generate",
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
