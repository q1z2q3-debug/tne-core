"""TNE-Core v3.1 - 三元九维认知框架"""
__version__ = "3.1.0"

from .tne_projector_v5 import YuanbaoProjectorV4, LocalSemanticBackend, LLMBackend
from .tne_distiller import TNE_Distiller
from .tne_trajectory_collector import TrajectoryCollector

__all__ = ["YuanbaoProjectorV4", "LocalSemanticBackend", "LLMBackend",
           "TNE_Distiller", "TrajectoryCollector"]
