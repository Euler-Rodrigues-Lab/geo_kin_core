"""Human-skeleton capsule overlay, drawn from a typed :class:`RetargetFrame`.

Device-independent by construction: it reads the same frame the solver reads,
so any device (XR, MediaPipe, a recorded frame stream) gets the same overlay
and no device SDK is needed to draw it. Needs the ``viz`` extra (mujoco).

Frames and conventions (as produced by the device adapters):

* ``*_sew`` S/E/W points and ``*_fingers`` keypoints are in the **upper-body**
  frame, placed in the world by ``p_world_upper_body`` / ``R_world_upper_body``.
* ``*_hka`` leg points are in the **lower-body** frame, whose origin is the hip
  centre (``hip_center_world``) and whose orientation is
  ``R_world_upper_body @ R_lower_upper.T`` (verified against the recorded
  ``A_world``; the residual is the ankle-joint vs heel definition).
* :meth:`set_base_offset` applies ``R_offset @ (p_world - p_offset)``, matching
  the retargeting session's mocap transform, so the human is drawn in the same
  frame as the robot: ``viz.set_base_offset(session.p_mocap_world,
  session.R_mocap_world.T)``.

Usage::

    viz = HumanCapsuleViz(viewer)
    ...
    capsules.clear(viewer)
    viz.set_base_offset(session.p_mocap_world, session.R_mocap_world.T)
    viz.draw(frame)
    viewer.sync()
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..types import RetargetFrame
from . import capsules

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_JOINTS = ("mcp", "pip", "tip")


class HumanCapsuleViz:
    """Capsule skeleton overlay for one human input frame."""

    def __init__(self, viewer, color=(0.2, 0.6, 1.0, 0.5), size: float = 0.06,
                 finger_size: float = 0.015, head_size: float = 0.09,
                 head_offset=(0.0, 0.0, 0.22), show_fingers: bool = True,
                 show_legs: bool = True, show_head: bool = True):
        """
        Args:
            viewer: MuJoCo passive viewer (anything with ``user_scn``).
            color: RGBA for every segment.
            size / finger_size / head_size: capsule radii (m).
            head_offset: head centre in the upper-body frame, above mid-shoulder.
            show_fingers / show_legs / show_head: per-part toggles.
        """
        self.viewer = viewer
        self.color = tuple(color)
        self.size = float(size)
        self.finger_size = float(finger_size)
        self.head_size = float(head_size)
        self.head_offset = np.asarray(head_offset, dtype=float)
        self.show_fingers = show_fingers
        self.show_legs = show_legs
        self.show_head = show_head
        self.offset_position = np.zeros(3)
        self.offset_rotation = np.eye(3)

    def set_base_offset(self, position=None, rotation=None) -> None:
        """Set the view transform ``R_offset @ (p_world - p_offset)``."""
        self.offset_position = np.zeros(3) if position is None else np.asarray(position, float).reshape(3)
        self.offset_rotation = np.eye(3) if rotation is None else np.asarray(rotation, float).reshape(3, 3)

    # -- coordinate helpers -------------------------------------------------
    def _to_view(self, p_world) -> np.ndarray:
        return self.offset_rotation @ (np.asarray(p_world, float) - self.offset_position)

    def _body_to_view(self, p_body, frame: RetargetFrame) -> Optional[np.ndarray]:
        if p_body is None or frame.R_world_upper_body is None or frame.p_world_upper_body is None:
            return None
        p_world = frame.p_world_upper_body + frame.R_world_upper_body @ np.asarray(p_body, float)
        return self._to_view(p_world)

    def _leg_to_view(self, p_lower, leg: dict, frame: RetargetFrame) -> Optional[np.ndarray]:
        hip_center = leg.get("hip_center_world")
        if p_lower is None or hip_center is None or frame.R_world_upper_body is None:
            return None
        R_world_lower = frame.R_world_upper_body
        if frame.R_lower_upper is not None:
            R_world_lower = R_world_lower @ frame.R_lower_upper.T
        return self._to_view(np.asarray(hip_center, float) + R_world_lower @ np.asarray(p_lower, float))

    # -- drawing ------------------------------------------------------------
    def draw(self, frame: Optional[RetargetFrame]) -> int:
        """Draw the skeleton for `frame`. Returns the number of geoms added.

        Does not clear the scene — call :func:`capsules.clear` once per rendered
        frame so several overlays can share it.
        """
        if frame is None or self.viewer is None:
            return 0
        n = 0
        shoulders = []
        for side in ("left", "right"):
            sew = getattr(frame, f"{side}_sew")
            if sew is None:
                continue
            S = self._body_to_view(sew.S, frame)
            E = self._body_to_view(sew.E, frame)
            W = self._body_to_view(sew.W, frame)
            n += capsules.add_chain(self.viewer, [S, E, W], self.size, self.color)
            if S is not None:
                shoulders.append(S)
            if self.show_fingers and W is not None:
                n += self._draw_hand(getattr(frame, f"{side}_fingers"), W, frame)

        mid_shoulder = np.mean(shoulders, axis=0) if len(shoulders) == 2 else (
            shoulders[0] if shoulders else None)
        if len(shoulders) == 2:
            n += capsules.add_capsule(self.viewer, shoulders[0], shoulders[1], self.size, self.color)

        if self.show_head and mid_shoulder is not None and frame.R_world_upper_body is not None:
            head = mid_shoulder + self.offset_rotation @ frame.R_world_upper_body @ self.head_offset
            n += capsules.add_sphere(self.viewer, head, self.head_size, self.color)

        if self.show_legs:
            n += self._draw_legs(frame, mid_shoulder)
        return n

    def _draw_hand(self, fingers: Optional[dict], wrist_view: np.ndarray,
                   frame: RetargetFrame) -> int:
        if not fingers:
            return 0
        n = 0
        for finger in FINGER_NAMES:
            sub = fingers.get(finger)
            if not sub:
                continue
            chain = [wrist_view]
            for joint in FINGER_JOINTS:
                point = self._body_to_view(sub.get(f"{finger}_{joint}"), frame)
                if point is not None:
                    chain.append(point)
            n += capsules.add_chain(self.viewer, chain, self.finger_size, self.color)
        return n

    def _draw_legs(self, frame: RetargetFrame, mid_shoulder) -> int:
        n = 0
        hip_centers = []
        for side in ("left", "right"):
            leg = getattr(frame, f"{side}_hka")
            if not leg:
                continue
            chain = [self._leg_to_view(leg.get(k), leg, frame) for k in ("H", "K", "A")]
            n += capsules.add_chain(self.viewer, chain, self.size, self.color)
            hip_center = leg.get("hip_center_world")
            if hip_center is not None:
                hip_centers.append(self._to_view(hip_center))
        if hip_centers and mid_shoulder is not None:
            n += capsules.add_capsule(self.viewer, mid_shoulder,
                                      np.mean(hip_centers, axis=0), self.size, self.color)
        return n
