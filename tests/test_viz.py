"""Shared overlay drawing (geo_kin_core.viz), against a real MjvScene."""

import os

os.environ.setdefault("MUJOCO_GL", "disabled")

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from geo_kin_core.types import RetargetFrame, SEWPose
from geo_kin_core.viz import HumanCapsuleViz, capsules

_XML = """
<mujoco><worldbody><body name="b"><geom type="sphere" size="0.1"/></body></worldbody></mujoco>
"""


class _Viewer:
    """Minimal stand-in for a passive viewer: just owns a scene."""

    def __init__(self, maxgeom=200):
        model = mujoco.MjModel.from_xml_string(_XML)
        self.user_scn = mujoco.MjvScene(model, maxgeom=maxgeom)
        self.user_scn.ngeom = 0


def _frame(with_legs=True, with_fingers=True):
    body = np.eye(3)
    fingers = {
        "index": {"index_finger_mcp": np.array([0.1, 0.0, 0.0]),
                  "index_finger_pip": np.array([0.15, 0.0, 0.0]),
                  "index_finger_tip": np.array([0.2, 0.0, 0.0])},
        "thumb": {"thumb_mcp": np.array([0.1, 0.02, 0.0]),
                  "thumb_pip": np.array([0.13, 0.03, 0.0]),
                  "thumb_tip": np.array([0.16, 0.04, 0.0])},
    }
    leg = {"H": np.zeros(3), "K": np.array([0.0, 0.0, -0.4]),
           "A": np.array([0.0, 0.0, -0.8]), "hip_center_world": np.array([0.0, 0.0, 0.9])}
    return RetargetFrame(
        left_sew=SEWPose(S=np.array([0.0, 0.2, 0.0]), E=np.array([0.2, 0.25, -0.2]),
                         W=np.array([0.4, 0.2, -0.1])),
        right_sew=SEWPose(S=np.array([0.0, -0.2, 0.0]), E=np.array([0.2, -0.25, -0.2]),
                          W=np.array([0.4, -0.2, -0.1])),
        R_world_upper_body=body, p_world_upper_body=np.array([1.0, 0.0, 1.4]),
        R_lower_upper=body,
        left_fingers=fingers if with_fingers else None,
        right_fingers=fingers if with_fingers else None,
        left_hka=leg if with_legs else None, right_hka=leg if with_legs else None,
    )


def test_draws_full_skeleton():
    viewer = _Viewer()
    n = HumanCapsuleViz(viewer).draw(_frame())
    assert n > 0 and viewer.user_scn.ngeom == n
    # arms(4) + shoulder line(1) + head(1) + fingers + legs(4) + torso(1)
    assert n >= 11


def test_part_toggles_reduce_geoms():
    full = HumanCapsuleViz(_Viewer()).draw(_frame())
    minimal = HumanCapsuleViz(_Viewer(), show_fingers=False, show_legs=False,
                              show_head=False).draw(_frame())
    assert 0 < minimal < full


def test_missing_parts_are_skipped_not_fatal():
    viewer = _Viewer()
    assert HumanCapsuleViz(viewer).draw(_frame(with_legs=False, with_fingers=False)) > 0
    assert HumanCapsuleViz(_Viewer()).draw(None) == 0
    assert HumanCapsuleViz(_Viewer()).draw(RetargetFrame()) == 0


def test_base_offset_matches_session_convention():
    """R_offset @ (p_world - p_offset), i.e. the mocap transform the session exposes."""
    frame = _frame(with_legs=False, with_fingers=False)
    viz = HumanCapsuleViz(_Viewer())
    p_world = frame.p_world_upper_body + frame.R_world_upper_body @ frame.right_sew.S
    np.testing.assert_allclose(viz._body_to_view(frame.right_sew.S, frame), p_world)

    R_off = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    p_off = np.array([1.0, 2.0, 3.0])
    viz.set_base_offset(p_off, R_off)
    np.testing.assert_allclose(viz._body_to_view(frame.right_sew.S, frame),
                               R_off @ (p_world - p_off))


def test_scene_capacity_is_respected():
    viewer = _Viewer(maxgeom=3)
    HumanCapsuleViz(viewer).draw(_frame())
    assert viewer.user_scn.ngeom <= 3  # no overflow, no exception


def test_capsule_primitives_reject_degenerate_input():
    viewer = _Viewer()
    assert capsules.add_capsule(viewer, np.zeros(3), np.zeros(3), 0.1) is False   # zero length
    assert capsules.add_capsule(viewer, np.zeros(3), np.full(3, np.nan), 0.1) is False
    assert capsules.add_sphere(viewer, np.zeros(3), 0.1) is True
    capsules.clear(viewer)
    assert viewer.user_scn.ngeom == 0


class _WheelStyleSession:
    """Backend exposing sew_capsules() -> [(name, a, b, r)] (licensed wheel)."""

    def sew_capsules(self):
        return [("L_upper", np.zeros(3), np.array([0.0, 0.0, 0.3]), 0.05),
                ("R_upper", np.zeros(3), np.array([0.0, 0.3, 0.0]), 0.05)]


class _RefStyleSession:
    """Backend exposing last_filtered_sew + sew_filter (private reference)."""

    class _Capsule:
        def __init__(self, a, b, r):
            self.a, self.b, self.r = a, b, r

    class _Filter:
        def parse_sew(self, flat):
            return {"flat": flat}

        def sew_to_capsules(self, sew_dict):
            return {"L_upper": _RefStyleSession._Capsule(
                np.zeros(3), np.array([0.0, 0.0, 0.3]), 0.05)}

    def __init__(self):
        self.last_filtered_sew = np.zeros(24)
        self.sew_filter = self._Filter()


@pytest.mark.parametrize("session,expected", [
    (_WheelStyleSession(), 2), (_RefStyleSession(), 1), (object(), 0)])
def test_filtered_sew_overlay_supports_every_backend(session, expected):
    from geo_kin_core.viz import draw_filtered_sew

    viewer = _Viewer()
    assert draw_filtered_sew(viewer, session) == expected
    assert viewer.user_scn.ngeom == expected


def test_filtered_sew_overlay_applies_to_world():
    from geo_kin_core.viz import draw_filtered_sew

    seen = []
    draw_filtered_sew(_Viewer(), _WheelStyleSession(),
                      to_world=lambda p: seen.append(p) or np.asarray(p) + 1.0)
    assert len(seen) == 4  # two capsules x two endpoints
