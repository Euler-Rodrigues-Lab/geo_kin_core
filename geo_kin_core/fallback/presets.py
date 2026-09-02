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

# Rainbow Robotics RB-Y1 Model M + XHand. The frame names are from the public
# model_v1.3_xhand_act.xml shipped by rby1_teleop. SEW points in RBY1 frame
# streams are expressed in the human upper-body frame, so the companion options
# request the frame/base mapping used by the frozen WARP MINK baseline.
RBY1_SIDES = {
    "right": {
        "shoulder_body": "link_right_arm_0",
        "elbow_body": "link_right_arm_3",
        "wrist_body": "link_right_arm_6",
        "palm_body": "right_hand_link",
        "joints": [f"right_arm_{i}" for i in range(7)],
    },
    "left": {
        "shoulder_body": "link_left_arm_0",
        "elbow_body": "link_left_arm_3",
        "wrist_body": "link_left_arm_6",
        "palm_body": "left_hand_link",
        "joints": [f"left_arm_{i}" for i in range(7)],
    },
    "torso": {
        "body": "link_torso_5",
        "joints": [f"torso_{i}" for i in range(6)],
    },
    "head": {
        "body": "link_head_2",
        "joints": ["head_0", "head_1"],
    },
    "base": {"body": "base"},
}

# Fixed human-wrist -> RB-Y1 joint-7 axis convention from the frozen MINK
# baseline. Stored as plain nested lists so this module remains numpy-free.
RBY1_OPTIONS = {
    "targets_in_upper_body_frame": True,
    "align_base_to_upper_body": True,
    "torso_target_mode": "upper_body",
    "wrist_rotation_offsets": {
        "right": [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "left": [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    },
}

PRESETS = {
    "g1": G1_SIDES,
    "rby1": RBY1_SIDES,
}

PRESET_OPTIONS = {
    "rby1": RBY1_OPTIONS,
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


def get_preset_options(name: str) -> dict:
    """Return robot-specific fallback defaults without mutating globals."""
    return copy.deepcopy(PRESET_OPTIONS.get(name, {}))
