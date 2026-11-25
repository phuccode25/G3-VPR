"""
G3-VPR: Geometry-Grounded Gaussian Aggregation and Matryoshka Relational Distillation
for Universal Single-Query Visual Place Recognition.
"""

from .model import G3VPR, G3VPRConfig, build_g3vpr
from .losses import UnifiedG3VPRLoss
from .teachers import DualTeacherEngine

__all__ = [
    "G3VPR",
    "G3VPRConfig",
    "build_g3vpr",
    "UnifiedG3VPRLoss",
    "DualTeacherEngine",
]
