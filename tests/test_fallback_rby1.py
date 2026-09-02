"""RBY1 integration validation for the mature MINK fallback preset."""

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("mink")

from geo_kin_core.frames import load_frames
from geo_kin_core.session import resolve_session


DEFAULT_RBY1 = Path(__file__).resolve().parents[2] / "rby1_teleop"
RBY1_ROOT = Path(os.environ.get("RBY1_TELEOP_ROOT", DEFAULT_RBY1))
ASSETS = RBY1_ROOT / "rby1_teleop/assets"
MODEL = ASSETS / "rby1_with_xhand/model_v1.3_xhand_act.xml"
MOTION = ASSETS / "sample_motion/ipman_roll.npz"

pytestmark = pytest.mark.skipif(
    not MODEL.exists() or not MOTION.exists(),
    reason="rby1_teleop assets are unavailable",
)

Q_RIGHT = np.array([-0.261799, -0.261799, 0.0, -1.894395, 0.0, 0.0, 0.0])
Q_LEFT = np.array([-0.261799, 0.261799, 0.0, -1.894395, 0.0, 0.0, 0.0])


def _session(**overrides):
    config = dict(
        backend="mink",
        model_xml=MODEL,
        wrist_target_mode="wrist",
        hand_position_cost=1.0,
        hand_orientation_cost=1.0,
        torso_orientation_cost=0.0,
        elbow_angle_cost=0.0,
        head_orientation_cost=0.0,
        mink_max_iters=40,
        enable_joint_limits=True,
    )
    config.update(overrides)
    session = resolve_session("rby1", **config)
    session.reset(Q_RIGHT, Q_LEFT)
    return session


def test_rby1_mink_eef_tracks_frame_stream():
    session = _session()
    stream = load_frames(MOTION)
    errors_mm = []
    for i in range(20):
        out = session.solve(stream[i])
        assert out.q_goal_right.shape == (7,)
        assert out.q_goal_left.shape == (7,)
        assert out.q_goal_torso.shape == (6,)
        assert out.q_goal_head.shape == (2,)
        assert out.p_world_base.shape == (3,)
        assert out.R_world_base.shape == (3, 3)
        assert np.all(np.isfinite(out.q_goal_right))
        for side in ("right", "left"):
            error = session._arm_task[side].compute_error(session._configuration)
            errors_mm.append(1e3 * np.linalg.norm(error[:3]))
    assert np.mean(errors_mm[10:]) < 1.0
    assert np.max(errors_mm[10:]) < 2.0


def test_rby1_mink_torso_elbow_head_mode_is_finite():
    session = _session(
        torso_orientation_cost=0.5,
        elbow_angle_cost=0.5,
        head_orientation_cost=0.05,
        mink_max_iters=10,
    )
    out = session.solve(load_frames(MOTION)[0])
    assert set(session._elbow_task) == {"right", "left"}
    assert np.all(np.isfinite(out.q_goal_torso))
    assert np.all(np.isfinite(out.q_goal_head))
