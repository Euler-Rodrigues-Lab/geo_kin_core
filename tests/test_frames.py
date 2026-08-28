"""Frame-stream save/load round trip (schema geo_kin_core.frames/1)."""

import numpy as np
import pytest

from geo_kin_core.frames import DEFAULT_FINGER_KEYS, load_frames, save_frames
from geo_kin_core.types import RetargetFrame, SEWPose

RNG = np.random.default_rng(7)


def _rot():
    q, _ = np.linalg.qr(RNG.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _fingers(keys=DEFAULT_FINGER_KEYS):
    return {name: {keys[i][j]: RNG.normal(size=3) for j in range(3)}
            for i, name in enumerate(("thumb", "index", "middle", "ring", "pinky"))}


def _leg():
    return {"H": RNG.normal(size=3), "K": RNG.normal(size=3), "A": RNG.normal(size=3),
            "ankle_rot": _rot(), "A_world": RNG.normal(size=3),
            "hip_center_world": RNG.normal(size=3)}


def _frame(**overrides):
    base = dict(
        left_sew=SEWPose.from_flat18(RNG.normal(size=18)),
        right_sew=SEWPose.from_flat18(RNG.normal(size=18)),
        R_world_upper_body=_rot(), p_world_upper_body=RNG.normal(size=3),
        head_rotation=_rot(), R_lower_upper=_rot(),
        left_fingers=_fingers(), right_fingers=_fingers(),
        left_hka=_leg(), right_hka=_leg(),
        left_gripper_val=0.31, right_gripper_val=0.42,
        extras={"body_center": RNG.normal(size=3)},
    )
    base.update(overrides)
    return RetargetFrame(**base)


def _assert_frames_equal(a, b):
    for side in ("left", "right"):
        pa, pb = getattr(a, f"{side}_sew"), getattr(b, f"{side}_sew")
        if pa is None:
            assert pb is None
        else:
            np.testing.assert_array_equal(pa.to_flat18(), pb.to_flat18())
        fa, fb = getattr(a, f"{side}_fingers"), getattr(b, f"{side}_fingers")
        if not fa:
            assert not fb
        else:
            assert set(fa) == set(fb)
            for finger, joints in fa.items():
                assert set(joints) == set(fb[finger])
                for joint, value in joints.items():
                    np.testing.assert_array_equal(value, fb[finger][joint])
        la, lb = getattr(a, f"{side}_hka"), getattr(b, f"{side}_hka")
        if not la:
            assert not lb
        else:
            for key, value in la.items():
                np.testing.assert_array_equal(np.asarray(value), np.asarray(lb[key]))
        assert getattr(a, f"{side}_gripper_val") == getattr(b, f"{side}_gripper_val")
    for field in ("R_world_upper_body", "p_world_upper_body", "head_rotation", "R_lower_upper"):
        va, vb = getattr(a, field), getattr(b, field)
        if va is None:
            assert vb is None
        else:
            np.testing.assert_array_equal(va, vb)
    for key, value in (a.extras or {}).items():
        np.testing.assert_array_equal(value, b.extras[key])


def test_round_trip_is_exact(tmp_path):
    frames = [_frame() for _ in range(5)]
    stream = load_frames(save_frames(tmp_path / "s.npz", frames, fps=60.0, source="unit test"))
    assert len(stream) == 5 and stream.fps == 60.0 and stream.source == "unit test"
    assert stream.duration == pytest.approx(5 / 60.0)
    for original, restored in zip(frames, stream.frames()):
        _assert_frames_equal(original, restored)


def test_irregular_finger_keys_survive(tmp_path):
    """Devices name joints irregularly (thumb_mcp vs index_finger_mcp) and the
    solvers look them up by exact string - the stream must preserve them."""
    frames = [_frame()]
    stream = load_frames(save_frames(tmp_path / "s.npz", frames))
    assert set(stream[0].left_fingers["index"]) == {
        "index_finger_mcp", "index_finger_pip", "index_finger_tip"}
    assert set(stream[0].left_fingers["thumb"]) == {"thumb_mcp", "thumb_pip", "thumb_tip"}


def test_custom_finger_keys_survive(tmp_path):
    keys = tuple(tuple(f"{f}_{j}" for j in ("mcp", "pip", "tip"))
                 for f in ("thumb", "index", "middle", "ring", "pinky"))
    frames = [_frame(left_fingers=_fingers(keys), right_fingers=None)]
    stream = load_frames(save_frames(tmp_path / "s.npz", frames))
    assert set(stream[0].left_fingers["index"]) == {"index_mcp", "index_pip", "index_tip"}
    assert not stream[0].right_fingers


def test_missing_fields_stay_missing(tmp_path):
    sparse = RetargetFrame(right_sew=SEWPose.from_flat18(np.arange(18, dtype=float)))
    stream = load_frames(save_frames(tmp_path / "s.npz", [sparse]))
    restored = stream[0]
    assert restored.left_sew is None and restored.right_sew is not None
    assert restored.left_hka is None and restored.R_lower_upper is None
    assert restored.left_gripper_val is None


def test_time_sampling_and_looping(tmp_path):
    frames = [_frame() for _ in range(4)]
    stream = load_frames(save_frames(tmp_path / "s.npz", frames, fps=10.0))
    _assert_frames_equal(stream.frame_at_time(0.2), stream[2])
    _assert_frames_equal(stream.frame_at_time(0.5, loop=True), stream[1])  # wraps
    assert stream.frame_at_time(0.5, loop=False) is None


def test_rejects_empty_and_bad_schema(tmp_path):
    with pytest.raises(ValueError):
        save_frames(tmp_path / "empty.npz", [])
    np.savez(tmp_path / "bad.npz", **{"meta/schema": np.array("nope/1")})
    with pytest.raises(ValueError, match="unsupported frame-stream schema"):
        load_frames(tmp_path / "bad.npz")
