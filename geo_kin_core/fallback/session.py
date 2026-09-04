"""Public mink differential-IK fallback session.

A differential-IK baseline built on `mink <https://github.com/kevinzakka/mink>`_:
per-arm wrist/palm FrameTasks, optional torso/elbow/head tasks, a posture
regularizer, frozen unowned DOFs, and optional hard joint limits. The RBY1
preset ports the independent task/frame behavior of the frozen WARP MINK
baseline without importing protected analytic SEW or XHand kernels.

This is deliberately NOT the SEW-geometric solver: elbow tracking is a standard
MINK position task (not a SEW swivel task), and there is no hand IK. It exists
so public robot repos and their CI can run
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
        elbow_position_cost: float = 0.0,
        head_orientation_cost: float = 0.0,
        posture_cost: float = 1e-3,
        lm_damping: float = 1.0,
        max_iters: int = 20,
        ik_dt: float = 1.0 / 60.0,
        ik_damping: float = 3e-2,
        qp_solver: str = "quadprog",
        pos_threshold: float = 1e-4,
        ori_threshold: float = 1e-3,
        enable_joint_limits: bool = True,
        targets_in_upper_body_frame: bool = False,
        align_base_to_upper_body: bool = False,
        torso_target_mode: str = "lower_upper",
        wrist_target_mode: str = "wrist",
        wrist_rotation_offsets: Optional[dict] = None,
        mobile_base: bool = False,
        mobile_base_alpha: float = 1.0,
        **_ignored,
    ):
        # Frozen WARP converter names accepted as aliases, so its recorded
        # MINK configs can drive this public fallback without translation.
        wrist_position_cost = _ignored.pop("hand_position_cost", wrist_position_cost)
        wrist_orientation_cost = _ignored.pop("hand_orientation_cost", wrist_orientation_cost)
        elbow_position_cost = _ignored.pop("elbow_angle_cost", elbow_position_cost)
        posture_cost = _ignored.pop("posture_body_cost", posture_cost)
        max_iters = _ignored.pop("mink_max_iters", max_iters)
        ik_dt = _ignored.pop("mink_dt", ik_dt)
        ik_damping = _ignored.pop("mink_damping", ik_damping)
        pos_threshold = _ignored.pop("mink_pos_threshold", pos_threshold)
        ori_threshold = _ignored.pop("mink_ori_threshold", ori_threshold)
        qp_solver = _ignored.pop("mink_solver_name", qp_solver)
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
        if torso_target_mode not in ("lower_upper", "upper_body"):
            raise ValueError("torso_target_mode must be 'lower_upper' or 'upper_body'")
        if wrist_target_mode not in ("wrist", "palm"):
            raise ValueError("wrist_target_mode must be 'wrist' or 'palm'")
        self.targets_in_upper_body_frame = bool(targets_in_upper_body_frame)
        self.align_base_to_upper_body = bool(align_base_to_upper_body)
        self.torso_target_mode = torso_target_mode
        self.wrist_target_mode = wrist_target_mode
        self.mobile_base = bool(mobile_base)
        self.mobile_base_alpha = float(mobile_base_alpha)
        if not 0.0 <= self.mobile_base_alpha <= 1.0:
            raise ValueError("mobile_base_alpha must be in [0, 1]")
        self._wrist_rotation_offsets = {
            side: np.asarray((wrist_rotation_offsets or {}).get(side, np.eye(3)), dtype=float).reshape(3, 3)
            for side in ("right", "left")
        }
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
        self._elbow_task = {}
        self._p_wrist_to_palm = {}
        controlled_dofs: list[int] = []
        for side in ("right", "left"):
            cfg = sides[side]
            self._arm_qposadr[side] = self._qposadr(cfg["joints"])
            self._arm_dofadr[side] = self._dofadr(cfg["joints"])
            controlled_dofs.extend(self._arm_dofadr[side].tolist())
            self._arm_task[side] = mink.FrameTask(
                frame_name=cfg["wrist_body"],
                frame_type="body",
                position_cost=wrist_position_cost,
                orientation_cost=wrist_orientation_cost,
                lm_damping=lm_damping,
            )
            if cfg.get("elbow_body") and elbow_position_cost > 0.0:
                self._elbow_task[side] = mink.FrameTask(
                    frame_name=cfg["elbow_body"], frame_type="body",
                    position_cost=elbow_position_cost, orientation_cost=0.0,
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

        head_cfg = sides.get("head")
        self._head_task = None
        self._head_qposadr = None
        if head_cfg is not None:
            self._head_qposadr = self._qposadr(head_cfg["joints"])
            self._head_dofadr = self._dofadr(head_cfg["joints"])
            if head_orientation_cost > 0.0:
                controlled_dofs.extend(self._head_dofadr.tolist())
                self._head_task = mink.FrameTask(
                    frame_name=head_cfg["body"], frame_type="body",
                    position_cost=0.0, orientation_cost=head_orientation_cost,
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
        if frozen and hasattr(mink, "DofFreezingTask"):
            self._constraints = [mink.DofFreezingTask(self._model, dof_indices=frozen)]
        else:
            # Compatibility with the MINK version used for the submitted WARP
            # baseline, which predates DofFreezingTask. A large posture weight
            # is the original solver's mechanism for locking wheels/hands and
            # other unowned DOFs.
            self._constraints = []
            if frozen:
                cost[np.asarray(frozen, dtype=int)] = 1e3
                self._posture_task = mink.PostureTask(self._model, cost=cost)
        self._limits = [mink.ConfigurationLimit(self._model)] if enable_joint_limits else []

        # --- seed state -------------------------------------------------------
        self._configuration.update()
        self._q_home = self._configuration.data.qpos.copy()
        self._posture_task.set_target_from_configuration(self._configuration)
        if self._torso_task is not None:
            torso_body = self._configuration.data.body(torso_cfg["body"])
            self._R_torso_home = torso_body.xmat.copy().reshape(3, 3)
        self._base_cfg = sides.get("base")
        self.R_world_base = None
        self.p_world_base = None
        self._R_world_base_initial = None
        self._p_world_base_initial = None
        if self.align_base_to_upper_body:
            if self._base_cfg is None:
                raise ValueError("align_base_to_upper_body requires sides['base']")
            shoulders = [sides[s].get("shoulder_body") for s in ("right", "left")]
            if any(name is None for name in shoulders):
                raise ValueError("base alignment requires shoulder_body for both arms")
            data = self._configuration.data
            base = data.body(self._base_cfg["body"])
            R_model_base = base.xmat.copy().reshape(3, 3)
            p_shoulders = 0.5 * (data.body(shoulders[0]).xpos + data.body(shoulders[1]).xpos)
            self._p_base_shoulder_home = R_model_base.T @ (p_shoulders - base.xpos)

        # Fixed robot wrist->palm translations are derived from the public MJCF
        # rather than protected XHand kinematics.
        data = self._configuration.data
        for side in ("right", "left"):
            cfg = sides[side]
            palm_name = cfg.get("palm_body") if self.wrist_target_mode == "palm" else None
            if palm_name:
                wrist = data.body(cfg["wrist_body"])
                palm = data.body(palm_name)
                R_wrist = wrist.xmat.copy().reshape(3, 3)
                self._p_wrist_to_palm[side] = R_wrist.T @ (palm.xpos - wrist.xpos)
            else:
                self._p_wrist_to_palm[side] = np.zeros(3)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_preset(cls, name: str, model_xml, **kwargs) -> "MinkFallbackSession":
        """Build a session from a shipped robot preset (e.g. ``'g1'``)."""
        from .presets import get_preset, get_preset_options

        options = get_preset_options(name)
        options.update(kwargs)
        return cls(model_xml=model_xml, sides=get_preset(name), **options)

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
        self.R_world_base = None
        self.p_world_base = None
        self._R_world_base_initial = None
        self._p_world_base_initial = None

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

        R_task_upper, p_task_upper = self._upper_body_pose(frame)
        tasks = [self._posture_task]
        arm_tasks = {}
        for side in ("right", "left"):
            sew = frame.right_sew if side == "right" else frame.left_sew
            if sew is None or sew.W is None:
                continue
            task = self._arm_task[side]
            task.set_target(_se3(*self._wrist_target(side, sew, frame, R_task_upper, p_task_upper)))
            tasks.append(task)
            arm_tasks[side] = task
            elbow = self._elbow_task.get(side)
            if elbow is not None and sew.E is not None:
                R_elbow = self._configuration.data.body(elbow.frame_name).xmat.copy().reshape(3, 3)
                elbow.set_target(_se3(R_elbow, self._point_target(sew.E, R_task_upper, p_task_upper)))
                tasks.append(elbow)

        torso_active = (
            self._torso_task is not None
            and (frame.R_world_upper_body is not None if self.torso_target_mode == "upper_body"
                 else frame.R_lower_upper is not None)
        )
        if torso_active:
            R_torso = R_task_upper if self.torso_target_mode == "upper_body" else self._torso_target(frame.R_lower_upper)
            self._torso_task.set_target(_se3(R_torso, np.zeros(3)))
            tasks.append(self._torso_task)

        head_active = self._head_task is not None and frame.head_rotation is not None
        if head_active:
            self._head_task.set_target(_se3(R_task_upper @ np.asarray(frame.head_rotation), np.zeros(3)))
            tasks.append(self._head_task)

        diag = RetargetDiagnostics()
        out = RetargetOutput(diag=diag)
        out.left_gripper_val = frame.left_gripper_val
        out.right_gripper_val = frame.right_gripper_val

        if not arm_tasks and not torso_active and not head_active:
            return out  # nothing to track this frame

        for _ in range(self.max_iters):
            solve_kwargs = {"limits": self._limits}
            if self._constraints:
                solve_kwargs["constraints"] = self._constraints
            vel = mink.solve_ik(
                self._configuration, tasks, self.ik_dt,
                self.qp_solver, self.ik_damping, **solve_kwargs,
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
        if head_active:
            out.q_goal_head = qpos[self._head_qposadr].copy()
        elif self._head_qposadr is not None and frame.head_rotation is not None:
            euler = Rotation.from_matrix(np.asarray(frame.head_rotation)).as_euler("ZYX")
            out.q_goal_head = self._clip_group(self._head_qposadr, np.array([euler[0], euler[1]]))
        if self.align_base_to_upper_body:
            out.R_world_base = np.asarray(self.R_world_base).copy()
            out.p_world_base = np.asarray(self.p_world_base).copy()

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

    def _point_target(self, point, R_task_upper, p_task_upper) -> np.ndarray:
        p = self._map_point(point)
        if self.targets_in_upper_body_frame:
            return p_task_upper + R_task_upper @ p
        return p

    def _wrist_target(self, side: str, sew, frame, R_task_upper, p_task_upper) -> tuple:
        """Return a task-frame wrist target using the frozen baseline mapping."""
        local_wrist = np.asarray(sew.W, dtype=float)
        if sew.R_world_wrist is not None:
            R_local = self._map_rotation(sew.R_world_wrist)
            R = (R_task_upper @ R_local if self.targets_in_upper_body_frame else R_local)
            R = R @ self._wrist_rotation_offsets[side]
        else:
            # Position-only frame: hold the current wrist orientation.
            body_name = self._arm_task[side].frame_name
            R = self._configuration.data.body(body_name).xmat.copy().reshape(3, 3)
        if self.wrist_target_mode == "palm" and sew.R_world_wrist is not None:
            mcp = (frame.extras or {}).get(f"{side}_finger_mcp_centroid")
            local_palm = local_wrist if mcp is None else local_wrist + np.asarray(sew.R_world_wrist) @ np.asarray(mcp)
            p_palm = self._point_target(local_palm, R_task_upper, p_task_upper)
            p = p_palm - R @ self._p_wrist_to_palm[side]
        else:
            p = self._point_target(local_wrist, R_task_upper, p_task_upper)
        return R, p

    @staticmethod
    def _yaw_rotation(R: np.ndarray) -> np.ndarray:
        x = np.asarray(R, dtype=float).reshape(3, 3)[:, 0].copy()
        x[2] = 0.0
        x = np.array([1.0, 0.0, 0.0]) if np.linalg.norm(x) < 1e-8 else x / np.linalg.norm(x)
        return Rotation.from_euler("Z", np.arctan2(x[1], x[0])).as_matrix()

    def _upper_body_pose(self, frame: RetargetFrame) -> tuple[np.ndarray, np.ndarray]:
        R_world_upper = np.eye(3) if frame.R_world_upper_body is None else np.asarray(frame.R_world_upper_body, dtype=float)
        p_world_upper = np.zeros(3) if frame.p_world_upper_body is None else np.asarray(frame.p_world_upper_body, dtype=float)
        if not self.align_base_to_upper_body:
            return R_world_upper, p_world_upper
        R_target = self._yaw_rotation(R_world_upper)
        p_target = p_world_upper - R_target @ self._p_base_shoulder_home
        p_target = np.asarray(p_target, dtype=float)
        p_target[2] = 0.0
        if self.R_world_base is None or self.p_world_base is None:
            self.R_world_base, self.p_world_base = R_target, p_target
            self._R_world_base_initial, self._p_world_base_initial = R_target.copy(), p_target.copy()
        elif self.mobile_base:
            a = self.mobile_base_alpha
            self.p_world_base[:2] = a * p_target[:2] + (1.0 - a) * self.p_world_base[:2]
            yaw_t = np.arctan2(R_target[1, 0], R_target[0, 0])
            yaw_c = np.arctan2(self.R_world_base[1, 0], self.R_world_base[0, 0])
            dyaw = (yaw_t - yaw_c + np.pi) % (2.0 * np.pi) - np.pi
            self.R_world_base = Rotation.from_euler("Z", yaw_c + a * dyaw).as_matrix()
        else:
            self.R_world_base = self._R_world_base_initial.copy()
            self.p_world_base = self._p_world_base_initial.copy()
        R_task_upper = self.R_world_base.T @ R_world_upper
        p_task_upper = self.R_world_base.T @ (p_world_upper - self.p_world_base)
        return R_task_upper, p_task_upper

    def _clip_group(self, addresses: np.ndarray, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float).copy()
        for i, adr in enumerate(addresses):
            joint_id = int(np.flatnonzero(self._model.jnt_qposadr == adr)[0])
            if self._model.jnt_limited[joint_id]:
                values[i] = np.clip(values[i], *self._model.jnt_range[joint_id])
        return values

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
