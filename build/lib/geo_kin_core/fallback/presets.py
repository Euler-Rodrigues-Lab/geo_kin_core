"""Robot presets for the mink differential-IK fallback.

A preset is a plain ``sides`` config dict naming, per arm, the MJCF wrist
body tracked by the fallback's FrameTask and the ordered joint list read back
into ``q_goal_*`` — plus the torso body and waist joint list. Joint ordering
matches the geo_kin_core spec artifacts (and therefore every other backend),
so the fallback is a drop-in for the licensed / reference solvers.

Body and joint names are verified against the ``g1_29dof.xml`` MJCF shipped in
the public g1_teleop assets.
"""

from __future__ import annotations

import copy

# Unitree G1 (29-DOF MJCF, g1_29dof.xml). Arm joint order matches the
# g1_29dof.npz spec artifact ({right,left}_arm/joint_names); torso order is
# [waist_yaw, waist_roll, waist_pitch] — the same [yaw, roll, pitch] packing
# the reference session emits for q_goal_torso.
G1_SIDES = {
    "right": {
        "wrist_body": "right_wrist_yaw_link",
        "joints": [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
    },
    "left": {
        "wrist_body": "left_wrist_yaw_link",
        "joints": [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ],
    },
    "torso": {
        "body": "torso_link",
        "joints": [
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        ],
    },
}

PRESETS = {
    "g1": G1_SIDES,
}


def get_preset(name: str) -> dict:
    """Return a deep copy of the named sides preset.

    Raises:
        KeyError: If no preset exists for ``name``.
    """
    try:
        preset = PRESETS[name]
    except KeyError:
        raise KeyError(
            f"No mink-fallback preset for robot {name!r}; available: {sorted(PRESETS)}. "
            "Construct MinkFallbackSession directly with a sides config dict."
        ) from None
    return copy.deepcopy(preset)
