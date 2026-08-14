"""Typed messages shared by all solvers (licensed wheel, private reference, fallback).

Draft v0 — field set mirrors the proven duck-typed action dict produced by
XRRTCBodyPoseDevice._default_process_bones_to_action and consumed by
G1FullBodySEWSolver, made explicit. Any field may be None; None on an output
joint group means "keep previous goal" (existing controller contract).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SEWPose:
    """One arm's Shoulder/Elbow/Wrist targets in the mocap/world frame."""

    S: np.ndarray  # (3,)
    E: np.ndarray  # (3,)
    W: np.ndarray  # (3,)
    R_world_wrist: Optional[np.ndarray] = None  # (3,3)

    @classmethod
    def from_flat18(cls, flat: np.ndarray) -> "SEWPose":
        """From the legacy (18,) layout: [S(3), E(3), W(3), R_world_wrist.flatten()(9)]."""
        flat = np.asarray(flat, dtype=float)
        return cls(S=flat[0:3], E=flat[3:6], W=flat[6:9], R_world_wrist=flat[9:18].reshape(3, 3))

    def to_flat18(self) -> np.ndarray:
        R = self.R_world_wrist if self.R_world_wrist is not None else np.eye(3)
        return np.concatenate([self.S, self.E, self.W, R.reshape(9)])


@dataclass
class PreprocessConfig:
    """Human->robot input conditioning, applied inside the solver session."""

    shoulder_width_scale: float = 1.0
    hip_width_scale: float = 1.0
    mocap_cartesian_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    mocap_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    R_mjworld_human: Optional[np.ndarray] = None  # (3,3)


@dataclass
class RetargetFrame:
    """One frame of human input. All poses in the mocap/world frame."""

    left_sew: Optional[SEWPose] = None
    right_sew: Optional[SEWPose] = None
    R_world_upper_body: Optional[np.ndarray] = None  # (3,3) shoulder-center rotation
    p_world_upper_body: Optional[np.ndarray] = None  # (3,)
    head_rotation: Optional[np.ndarray] = None  # (3,3), in upper-body frame
    R_lower_upper: Optional[np.ndarray] = None  # (3,3) torso: lower-to-upper-body rotation
    left_fingers: Optional[dict] = None  # per-finger keypoint dicts (body-centric)
    right_fingers: Optional[dict] = None
    left_hka: Optional[np.ndarray] = None  # legs (hip/knee/ankle); unused until leg support
    right_hka: Optional[np.ndarray] = None
    left_gripper_val: Optional[float] = None
    right_gripper_val: Optional[float] = None
    extras: dict = field(default_factory=dict)  # tags, centroids, device-specific


@dataclass
class RetargetDiagnostics:
    """Per-solve diagnostics for viz/logging; never required for control."""

    filtered_sew: Optional[np.ndarray] = None  # post-XPBD SEW (capsule overlay viz)
    is_least_squares: dict = field(default_factory=dict)  # per limb
    solution_branch: dict = field(default_factory=dict)  # per limb
    singularity: dict = field(default_factory=dict)  # per limb metrics
    timing_ms: dict = field(default_factory=dict)
    # Post-solve robot-side FK in the ROBOT BASE frame (sessions that support
    # it, e.g. Vega with robot_fk=True):
    #   {'right': {...}|None, 'left': {...}|None,
    #    'R_base_body': (3,3), 'p_base_body': (3,)}
    # Per-side dict carries the G1-style FK keys (S, S_end, E, E_end, W,
    # W_end, T, se, ew, R_0_7, S_capsule, E_capsule, W_capsule, T_capsule)
    # plus, on hand-equipped sessions, palmbox_origin (3,) / palmbox_R (3,3).
    robot_fk: Optional[dict] = None


@dataclass
class RetargetOutput:
    """Joint goals. None = keep previous goal for that group."""

    q_goal_right: Optional[np.ndarray] = None  # (7,)
    q_goal_left: Optional[np.ndarray] = None  # (7,)
    q_goal_torso: Optional[np.ndarray] = None  # robot-specific DOF (G1: 3-DOF waist)
    q_goal_right_hand: Optional[np.ndarray] = None
    q_goal_left_hand: Optional[np.ndarray] = None
    q_goal_right_leg: Optional[np.ndarray] = None  # reserved; None until leg support
    q_goal_left_leg: Optional[np.ndarray] = None
    q_goal_head: Optional[np.ndarray] = None
    left_gripper_val: Optional[float] = None  # pass-through
    right_gripper_val: Optional[float] = None
    diag: RetargetDiagnostics = field(default_factory=RetargetDiagnostics)
