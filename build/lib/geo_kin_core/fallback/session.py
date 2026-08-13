"""Public mink differential-IK fallback session.

A plain differential-IK wrist tracker built on `mink <https://github.com/kevinzakka/mink>`_:
per-arm FrameTask on the wrist pose taken from ``RetargetFrame.*_sew`` (the W
position and R_world_wrist block), an optional torso orientation task from
``RetargetFrame.R_lower_upper``, a posture regularizer, and hard joint limits.

This is deliberately NOT the SEW-geometric solver: there is no elbow / swivel
tracking and no hand IK. It exists so public robot repos and their CI can run
teleop end-to-end without a license; the licensed ``geo_kin`` wheel (or the
private reference) is a drop-in upgrade via
:func:`geo_kin_core.session.resolve_session`.

Frame convention: with the default :class:`~geo_kin_core.types.PreprocessConfig`,
wrist targets from the frame are interpreted directly in the MJCF world frame.
``R_mjworld_human`` / ``mocap_cartesian_scale`` / ``mocap_offset`` remap them
(``p = R @ (W * scale) + offset``, ``R_target = R @ R_world_wrist``).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..types import (
    PreprocessConfig,
    RetargetDiagnostics,
    RetargetFrame,
    RetargetOutput,
)

try:
    import mujoco
    import mink
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The mink differential-IK fallback needs the 'fallback' extra: "
        "pip install 'geo-kin-core[fallback]'  (mink, mujoco, quadprog, scipy)"
    ) from exc

from scipy.spatial.transform import Rotation


def _se3(R: np.ndarray, p: np.ndarray) -> "mink.SE3":
    return mink.SE3.from_rotation_and_translation(
        rotation=mink.SO3.from_matrix(np.asarray(R, dtype=float).reshape(3, 3)),
        translation=np.asarray(p, dtype=float).reshape(3),
    )


class MinkFallbackSession:
    """Differential-IK fallback implementing the RetargetingSolver protocol.

    Args:
        model_xml: Path to the robot MJCF (e.g. the ``g1_29dof.xml`` shipped in
            the public robot repo's assets).
        sides: Config dict naming the bodies/joints the solver touches::

                {
                  "right": {"wrist_body": <body>, "joints": [<7 names in output order>]},
                  "left":  {"wrist_body": <body>, "joints": [...]},
                  "torso": {"body": <body>, "joints": [<waist names in output order>]},
                }

            An optional per-arm ``"palm_body"`` overrides which body the
            FrameTask tracks (the joint list still defines the readback).
            See :mod:`geo_kin_core.fallback.presets` for the shipped G1 preset.
        preprocess: :class:`~geo_kin_core.types.PreprocessConfig` or kwargs dict.
            Only ``R_mjworld_human`` / ``mocap_cartesian_scale`` /
            ``mocap_offset`` are honored (there is no SEW machinery to scale).
        torso: Solve the waist from ``frame.R_lower_upper``.
        torso_mag: Rotation-vector magnitude scaling applied to R_lower_upper
            before it becomes the torso orientation target (same knob as the
            reference session).
        wrist_position_cost / wrist_orientation_cost / torso_orientation_cost /
            posture_cost / lm_damping: task weights.
        max_iters / ik_dt / ik_damping / qp_solver / pos_threshold /
            ori_threshold: inner differential-IK loop knobs.
        **_ignored: Any other kwarg (SEW filter cutoffs, collision-avoidance
            flags, ``robot=`` / ``hand=`` from resolve_session, ...) is accepted
            and silently ignored for drop-in compatibility.
    """

    def __init__(
        self,
        model_xml,
        sides: dict,
        preprocess=None,
        torso: bool = True,
        torso_mag=(0.8, 0.8, 1.0),
        wrist_position_cost: float = 4.0,
        wrist_orientation_cost: float = 1.0,
        torso_orientation_cost: float = 0.2,
        posture_cost: float = 1e-3,
        lm_damping: float = 1.0,
        max_iters: int = 20,
        ik_dt: float = 1.0 / 60.0,
        ik_damping: float = 3e-2,
        qp_solver: str = "quadprog",
        pos_threshold: float = 1e-4,
        ori_threshold: float = 1e-3,
        **_ignored,
    ):
        model_xml = Path(model_xml)
        if not model_xml.exists():
            raise FileNotFoundError(f"Robot MJCF not found: {model_xml}")
        self.model_xml = str(model_xml)

        if preprocess is None:
            preprocess = PreprocessConfig()
        elif isinstance(preprocess, dict):
            preprocess = PreprocessConfig(**preprocess)
        self.preprocess_config = preprocess

        self.torso_enabled = bool(torso)
        self.torso_mag = np.asarray(torso_mag, dtype=float)
        self.max_iters = int(max_iters)
        self.ik_dt = float(ik_dt)
        self.ik_damping = float(ik_damping)
        self.qp_solver = str(qp_solver)
        self.pos_threshold = float(pos_threshold)
        self.ori_threshold = float(ori_threshold)

        self._model = mujoco.MjModel.from_xml_path(self.model_xml)
        self._configuration = mink.Configuration(self._model)

        # --- joint addressing per group (output order == sides joint order) --
        for key in ("right", "left"):
            if key not in sides:
                raise ValueError(f"sides config is missing the {key!r} arm entry")
        self._arm_qposadr = {}
        self._arm_dofadr = {}
        self._arm_task = {}
        controlled_dofs: list[int] = []
        for side in ("right", "left"):
            cfg = sides[side]
            self._arm_qposadr[side] = self._qposadr(cfg["joints"])
            self._arm_dofadr[side] = self._dofadr(cfg["joints"])
            controlled_dofs.extend(self._arm_dofadr[side].tolist())
            self._arm_task[side] = mink.FrameTask(
                frame_name=cfg.get("palm_body") or cfg["wrist_body"],
                frame_type="body",
                position_cost=wrist_position_cost,
                orientation_cost=wrist_orientation_cost,
                lm_damping=lm_damping,
            )

        torso_cfg = sides.get("torso")
        self._torso_task = None
        self._torso_qposadr = None
        if torso_cfg is not None:
            self._torso_qposadr = self._qposadr(torso_cfg["joints"])
            self._torso_dofadr = self._dofadr(torso_cfg["joints"])
            if self.torso_enabled:
                controlled_dofs.extend(self._torso_dofadr.tolist())
                self._torso_task = mink.FrameTask(
                    frame_name=torso_cfg["body"],
                    frame_type="body",
                    position_cost=0.0,
                    orientation_cost=torso_orientation_cost,
                    lm_damping=lm_damping,
                )

        # --- posture regularizer over the controlled DOFs only ---------------
        cost = np.zeros(self._model.nv)
        cost[np.asarray(controlled_dofs, dtype=int)] = float(posture_cost)
        self._posture_task = mink.PostureTask(self._model, cost=cost)

        # --- freeze everything the fallback does not own ---------------------
        # (legs, floating base, any hand joints in the MJCF, and the waist when
        # torso=False). Enforced as an exact equality constraint in the QP.
        frozen = sorted(set(range(self._model.nv)) - set(controlled_dofs))
        self._constraints = (
            [mink.DofFreezingTask(self._model, dof_indices=frozen)] if frozen else []
        )
        self._limits = [mink.ConfigurationLimit(self._model)]

        # --- seed state -------------------------------------------------------
        self._configuration.update()
        self._q_home = self._configuration.data.qpos.copy()
        self._posture_task.set_target_from_configuration(self._configuration)
        if self._torso_task is not None:
            torso_body = self._configuration.data.body(torso_cfg["body"])
            self._R_torso_home = torso_body.xmat.copy().reshape(3, 3)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_preset(cls, name: str, model_xml, **kwargs) -> "MinkFallbackSession":
        """Build a session from a shipped robot preset (e.g. ``'g1'``)."""
        from .presets import get_preset

        return cls(model_xml=model_xml, sides=get_preset(name), **kwargs)

    def _qposadr(self, joint_names) -> np.ndarray:
        return np.array(
            [self._model.jnt_qposadr[self._model.joint(n).id] for n in joint_names],
            dtype=int,
        )

    def _dofadr(self, joint_names) -> np.ndarray:
        return np.array(
            [self._model.jnt_dofadr[self._model.joint(n).id] for n in joint_names],
            dtype=int,
        )

    # ------------------------------------------------------------------
    # Session protocol
    # ------------------------------------------------------------------

    def reset(
        self,
        q_init_right: Optional[np.ndarray] = None,
        q_init_left: Optional[np.ndarray] = None,
    ) -> None:
        """Re-seed the internal configuration (home pose + optional arm inits)."""
        qpos = self._configuration.data.qpos
        qpos[:] = self._q_home
        if q_init_right is not None:
            adr = self._arm_qposadr["right"]
            qpos[adr] = np.asarray(q_init_right, dtype=float).reshape(-1)[: adr.size]
        if q_init_left is not None:
            adr = self._arm_qposadr["left"]
            qpos[adr] = np.asarray(q_init_left, dtype=float).reshape(-1)[: adr.size]
        self._configuration.update()
        self._posture_task.set_target_from_configuration(self._configuration)

    def solve(
        self,
        frame: RetargetFrame,
        engaged: bool = True,
        q_current_right: Optional[np.ndarray] = None,
        q_current_left: Optional[np.ndarray] = None,
        **_ignored,
    ) -> RetargetOutput:
        """Track wrist (+ torso) targets with a few differential-IK steps.

        Returns a RetargetOutput; None on a joint group means "keep the
        previous goal". Hands are never solved here (q_goal_*_hand stay None);
        gripper values pass through untouched.
        """
        if not engaged:
            # Freeze: do not advance the internal configuration, emit no goals.
            return RetargetOutput()

        t0 = time.perf_counter()
        qpos = self._configuration.data.qpos

        # Warm start from the measured robot state when provided.
        if q_current_right is not None:
            adr = self._arm_qposadr["right"]
            qpos[adr] = np.asarray(q_current_right, dtype=float).reshape(-1)[: adr.size]
        if q_current_left is not None:
            adr = self._arm_qposadr["left"]
            qpos[adr] = np.asarray(q_current_left, dtype=float).reshape(-1)[: adr.size]
        self._configuration.update()

        tasks = [self._posture_task]
        arm_tasks = {}
        for side in ("right", "left"):
            sew = frame.right_sew if side == "right" else frame.left_sew
            if sew is None or sew.W is None:
                continue
            task = self._arm_task[side]
            task.set_target(_se3(*self._wrist_target(side, sew)))
            tasks.append(task)
            arm_tasks[side] = task

        torso_active = (
            self._torso_task is not None and frame.R_lower_upper is not None
        )
        if torso_active:
            self._torso_task.set_target(
                _se3(self._torso_target(frame.R_lower_upper), np.zeros(3))
            )
            tasks.append(self._torso_task)

        diag = RetargetDiagnostics()
        out = RetargetOutput(diag=diag)
        out.left_gripper_val = frame.left_gripper_val
        out.right_gripper_val = frame.right_gripper_val

        if not arm_tasks and not torso_active:
            return out  # nothing to track this frame

        for _ in range(self.max_iters):
            vel = mink.solve_ik(
                self._configuration,
                tasks,
                self.ik_dt,
                self.qp_solver,
                self.ik_damping,
                limits=self._limits,
                constraints=self._constraints,
            )
            self._configuration.integrate_inplace(vel, self.ik_dt)
            if self._converged(arm_tasks):
                break

        if "right" in arm_tasks:
            out.q_goal_right = qpos[self._arm_qposadr["right"]].copy()
        if "left" in arm_tasks:
            out.q_goal_left = qpos[self._arm_qposadr["left"]].copy()
        if torso_active:
            out.q_goal_torso = qpos[self._torso_qposadr].copy()

        diag.timing_ms["solve"] = (time.perf_counter() - t0) * 1e3
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _map_point(self, p: np.ndarray) -> np.ndarray:
        """Apply the preprocess mapping (scale, rotate, offset) to a position."""
        cfg = self.preprocess_config
        p = np.asarray(p, dtype=float).reshape(3) * np.asarray(
            cfg.mocap_cartesian_scale, dtype=float
        )
        if cfg.R_mjworld_human is not None:
            p = np.asarray(cfg.R_mjworld_human, dtype=float).reshape(3, 3) @ p
        return p + np.asarray(cfg.mocap_offset, dtype=float)

    def _map_rotation(self, R: np.ndarray) -> np.ndarray:
        cfg = self.preprocess_config
        R = np.asarray(R, dtype=float).reshape(3, 3)
        if cfg.R_mjworld_human is not None:
            R = np.asarray(cfg.R_mjworld_human, dtype=float).reshape(3, 3) @ R
        return R

    def _wrist_target(self, side: str, sew) -> tuple:
        """(R, p) world-frame target for one arm's wrist FrameTask."""
        p = self._map_point(sew.W)
        if sew.R_world_wrist is not None:
            R = self._map_rotation(sew.R_world_wrist)
        else:
            # Position-only frame: hold the current wrist orientation.
            body_name = self._arm_task[side].frame_name
            R = self._configuration.data.body(body_name).xmat.copy().reshape(3, 3)
        return R, p

    def _torso_target(self, R_lower_upper: np.ndarray) -> np.ndarray:
        """World-frame torso orientation target from R_lower_upper."""
        rotvec = Rotation.from_matrix(
            np.asarray(R_lower_upper, dtype=float).reshape(3, 3)
        ).as_rotvec()
        R_cmd = Rotation.from_rotvec(rotvec * self.torso_mag).as_matrix()
        return R_cmd @ self._R_torso_home

    def _converged(self, arm_tasks: dict) -> bool:
        for task in arm_tasks.values():
            err = task.compute_error(self._configuration)
            if np.linalg.norm(err[:3]) > self.pos_threshold:
                return False
            if np.linalg.norm(err[3:]) > self.ori_threshold:
                return False
        return bool(arm_tasks)
