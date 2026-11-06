"""CogOS · 认知空间 OS 骨架（最小可运行版）"""

from .axes import DIM, DIM_NAMES, SemanticFrame, get_domain_frame, accumulate_bias
from .lexicon import raw_scores
from .mapping import MappingResult, map_text, detect_domain, neighbors_of
from .memory import MemoryItem, MemorySpace, RecallResult
from .analysis import analyze, analyze_gua, analyze_yaos, analyze_index
from .resonance import build as build_resonance
from .pipeline import CogOS, Turn

__all__ = [
    "DIM",
    "DIM_NAMES",
    "SemanticFrame",
    "get_domain_frame",
    "accumulate_bias",
    "raw_scores",
    "MappingResult",
    "map_text",
    "detect_domain",
    "neighbors_of",
    "MemoryItem",
    "MemorySpace",
    "RecallResult",
    "analyze",
    "analyze_gua",
    "analyze_yaos",
    "analyze_index",
    "build_resonance",
    "CogOS",
    "Turn",
]
