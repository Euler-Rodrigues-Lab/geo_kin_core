"""Shared MuJoCo overlays for teleoperation demos (``viz`` extra: mujoco).

Robot-agnostic and device-agnostic: everything draws from geo_kin_core types,
so each robot repo gets the same human/solver overlays without copying code.
"""

from . import capsules
from .human import HumanCapsuleViz
from .solver import draw_filtered_sew

__all__ = ["capsules", "HumanCapsuleViz", "draw_filtered_sew"]
