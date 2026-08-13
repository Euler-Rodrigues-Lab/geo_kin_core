"""Model extractors: URDF/MJCF -> transform dicts consumed by the geo-kin solvers.

These are near-verbatim ports of the reference extractors so the produced
dicts are STRUCTURALLY and NUMERICALLY identical to what the solvers were
developed against:

- robot parts: ``get_frame_transforms_from_pinocchio`` (G1 full-body monolith)
- inspire hand: ``get_frame_transforms_from_xml`` (inspire_hand_teleop monolith)
- psyonic hand: ``get_psyonic_frame_transforms_from_xml`` (G1 full-body monolith)

Heavy dependencies (pinocchio for URDF, scipy for MJCF quaternion parsing) are
imported lazily inside the generate_* entry points; loading a saved spec never
needs them (see ``geo_kin_core.spec._io``).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, List, Sequence

import numpy as np

#: Robot parts the URDF extractor knows how to slice out of a G1-style model.
ROBOT_JOINT_SEQUENCES = {
    "right_arm": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"],
    "left_arm": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                 "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"],
    "torso": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],  # G1 has 3-DOF waist
    # Lower body sequences for G1 (6-DOF per leg)
    "right_leg": ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"],
    "left_leg": ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                 "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"],
}

ROBOT_PARTS: List[str] = list(ROBOT_JOINT_SEQUENCES.keys())

HAND_TYPES = ("inspire", "psyonic")


# ---------------------------------------------------------------------------
# Robot (URDF via pinocchio)
# ---------------------------------------------------------------------------

def generate_robot_spec(urdf_path: str, parts: Sequence[str]) -> Dict:
    """Extract per-part frame transforms from a URDF.

    Args:
        urdf_path: Path to the robot URDF.
        parts: Part names to extract (subset of :data:`ROBOT_PARTS`).

    Returns:
        ``{part: {'R': [3x3, ...], 'p': [(3,), ...], 'h': [(3,), ...],
        'joint_names': [str, ...]}}`` — each part dict structurally identical
        to the monolith's ``get_frame_transforms_from_pinocchio`` output.
    """
    import pinocchio as pin  # heavy dep, lazy

    parts = list(parts)
    unknown = [p for p in parts if p not in ROBOT_JOINT_SEQUENCES]
    if unknown:
        raise ValueError(f"Unknown parts: {unknown}. Available: {ROBOT_PARTS}")

    model = pin.buildModelFromUrdf(str(urdf_path))
    return {part: _extract_robot_part(model, part) for part in parts}


def _extract_robot_part(model, part_name: str) -> Dict:
    """Extract R_i_i+1, p_i_i+1, joint axes h_i for one part.

    Near-verbatim port of the monolith's ``get_frame_transforms_from_pinocchio``.
    """
    if part_name not in ROBOT_JOINT_SEQUENCES:
        raise ValueError(f"Unknown part_name: {part_name}. Available: {ROBOT_PARTS}")

    joint_names = ROBOT_JOINT_SEQUENCES[part_name]
    R_transforms = []
    p_transforms = []
    h_transforms = []

    for joint_name in joint_names:
        try:
            joint_id = model.getJointId(joint_name)
            joint = model.joints[joint_id]

            # Joint placement (local transformation)
            M_local = model.jointPlacements[joint_id]
            R_i_iplus1 = M_local.rotation.copy()     # 3x3 numpy array
            p_i_iplus1 = M_local.translation.copy()  # 3x1 numpy array

            # Joint axis in local frame
            if joint.shortname() == "JointModelRZ":
                h_i = np.array([0.0, 0.0, 1.0])
            elif joint.shortname() == "JointModelRY":
                h_i = np.array([0.0, 1.0, 0.0])
            elif joint.shortname() == "JointModelRX":
                h_i = np.array([1.0, 0.0, 0.0])
            elif hasattr(joint, 'axis'):
                h_i = joint.axis.copy()
            else:
                print(f"Warning: Unknown joint type {joint.shortname()} for joint {joint_name}, using zero axis")
                h_i = np.array([0.0, 0.0, 0.0])

            R_transforms.append(R_i_iplus1)
            p_transforms.append(p_i_iplus1)
            h_transforms.append(h_i)

        except Exception as e:
            print(f"Error processing joint {joint_name}: {e}")
            continue

    return {
        'R': R_transforms,
        'p': p_transforms,
        'h': h_transforms,
        'joint_names': joint_names
    }


# ---------------------------------------------------------------------------
# Hands (MJCF via ElementTree; no mujoco runtime needed)
# ---------------------------------------------------------------------------

def generate_hand_spec(xml_path: str, side: str, hand_type: str = "inspire") -> Dict:
    """Extract per-finger frame transforms from a hand MJCF xml.

    Args:
        xml_path: Path to the MJCF xml containing the hand.
        side: 'left' or 'right'.
        hand_type: 'inspire' or 'psyonic'.

    Returns:
        ``{finger: {'R': [...], 'p': [...], 'h': [...], 'joint_names': [...]}}``
        structurally identical to the monolith extractors' outputs.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if hand_type == "inspire":
        return _extract_inspire_transforms(str(xml_path), side)
    if hand_type == "psyonic":
        return _extract_psyonic_transforms(str(xml_path), side)
    raise ValueError(f"hand_type must be one of {HAND_TYPES}, got {hand_type!r}")


def _find_body_in_xml(parent, body_name: str):
    """Find a body element by name in XML tree."""
    return parent.find(f".//body[@name='{body_name}']")


def _find_site_in_xml(parent, site_name: str):
    """Find a site element by name in XML tree."""
    if parent is None:
        return None
    return parent.find(f".//site[@name='{site_name}']")


def _parse_transform_from_body(body_element):
    """Parse transformation (R, p) from an MJCF body element.

    Returns:
        tuple: (R, p) - 3x3 rotation matrix and 3x1 position vector
    """
    from scipy.spatial.transform import Rotation  # lazy: generation-time only

    if body_element is None:
        return np.eye(3), np.zeros(3)

    pos_str = body_element.get('pos', '0 0 0')
    p = np.array([float(x) for x in pos_str.split()])

    # MuJoCo uses (w, x, y, z) quaternion format
    quat_str = body_element.get('quat', '1 0 0 0')
    quat = np.array([float(x) for x in quat_str.split()])

    if len(quat) == 4:
        quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
        R = Rotation.from_quat(quat_scipy).as_matrix()
    else:
        R = np.eye(3)

    return R, p


def _extract_inspire_transforms(xml_path: str, hand_side: str) -> Dict:
    """Parse frame transformations R, p, h for the Inspire hand.

    Near-verbatim port of the inspire monolith's ``get_frame_transforms_from_xml``.
    Chain per finger: wrist (identity base) -> L1 -> L2 -> tip. The thumb's
    coupled L3/L4 links are collapsed into the tip transform at zero angles.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Thumb has 4 joints in XML but only 2 independent (q1, q2); q3, q4 coupled.
    # Fingers have 2 joints: q1, q2 (q2 coupled 1:1 to q1).
    finger_joints = {
        'thumb': [f'{hand_side}_thumb_1_joint', f'{hand_side}_thumb_2_joint'],
        'index': [f'{hand_side}_index_1_joint', f'{hand_side}_index_2_joint'],
        'middle': [f'{hand_side}_middle_1_joint', f'{hand_side}_middle_2_joint'],
        'ring': [f'{hand_side}_ring_1_joint', f'{hand_side}_ring_2_joint'],
        'pinky': [f'{hand_side}_little_1_joint', f'{hand_side}_little_2_joint']
    }

    # Map generic finger names to XML prefixes (pinky -> little)
    finger_xml_prefix = {
        'thumb': 'thumb',
        'index': 'index',
        'middle': 'middle',
        'ring': 'ring',
        'pinky': 'little'
    }

    transforms = {}

    # The base for all fingers is the wrist_yaw_link of the specified hand side
    wrist_link_name = f'{hand_side}_wrist_yaw_link'
    wrist_body = _find_body_in_xml(root, wrist_link_name)

    if wrist_body is None:
        print(f"Warning: Could not find {wrist_link_name} in XML")
        return transforms

    # Hand IK works in the wrist frame -> identity base transform.
    wrist_R = np.eye(3)
    wrist_p = np.zeros(3)

    for finger_name, joint_names in finger_joints.items():
        finger_transforms = {
            'R': [],
            'p': [],
            'h': [],
            'joint_names': joint_names.copy()
        }

        prefix = finger_xml_prefix.get(finger_name, finger_name)

        # 1. Base transform (wrist)
        finger_transforms['R'].append(wrist_R)
        finger_transforms['p'].append(wrist_p)

        # 2. Finger L1 body (child of wrist), e.g. right_index_1
        l1_name = f'{hand_side}_{prefix}_1'
        l1_body = _find_body_in_xml(wrist_body, l1_name)

        if l1_body is None:
            print(f"Warning: Could not find {l1_name} in XML")
            continue

        R_l1, p_l1 = _parse_transform_from_body(l1_body)
        finger_transforms['R'].append(R_l1)
        finger_transforms['p'].append(p_l1)

        # Joint 1 axis
        joint1_name = joint_names[0]
        joint1 = l1_body.find(f".//joint[@name='{joint1_name}']")
        if joint1 is not None:
            axis_str = joint1.get('axis', '0 0 1')
            h1 = np.array([float(x) for x in axis_str.split()])
            finger_transforms['h'].append(h1)
        else:
            finger_transforms['h'].append(np.array([0, 0, 1]))

        # 3. Finger L2 body (child of L1)
        l2_name = f'{hand_side}_{prefix}_2'
        l2_body = _find_body_in_xml(l1_body, l2_name)

        if l2_body is not None:
            R_l2, p_l2 = _parse_transform_from_body(l2_body)
            finger_transforms['R'].append(R_l2)
            finger_transforms['p'].append(p_l2)

            # Joint 2 axis
            joint2_name = joint_names[1]
            joint2 = l2_body.find(f".//joint[@name='{joint2_name}']")
            if joint2 is not None:
                axis_str = joint2.get('axis', '0 0 1')
                h2 = np.array([float(x) for x in axis_str.split()])
                finger_transforms['h'].append(h2)
            else:
                finger_transforms['h'].append(np.array([0, 0, 1]))

            if finger_name == 'thumb':
                # Thumb: traverse coupled L3/L4 and collapse into the tip
                # transform at zero angles (matches reference behavior).
                l3_name = f'{hand_side}_{prefix}_3'
                l3_body = _find_body_in_xml(l2_body, l3_name)

                if l3_body is not None:
                    R_l3, p_l3 = _parse_transform_from_body(l3_body)

                    l4_name = f'{hand_side}_{prefix}_4'
                    l4_body = _find_body_in_xml(l3_body, l4_name)

                    if l4_body is not None:
                        R_l4, p_l4 = _parse_transform_from_body(l4_body)

                        # Force sensor 4 marks the tip
                        sensor4 = l4_body.find(f".//geom[@mesh='{hand_side}_{prefix}_force_sensor_4']")
                        sensor_pos = np.zeros(3)
                        if sensor4 is not None:
                            s_pos_str = sensor4.get('pos', '0 0 0')
                            sensor_pos = np.array([float(x) for x in s_pos_str.split()])

                        # P_tip_wrt_L2 at zero angles
                        p_tip_local = p_l3 + R_l3 @ (p_l4 + R_l4 @ sensor_pos)

                        finger_transforms['R'].append(np.eye(3))
                        finger_transforms['p'].append(p_tip_local)
                        finger_transforms['joint_names'].append(f'{finger_name}_tip')

            else:
                # Fingers (index..pinky): force sensor 3 marks the tip
                sensor3 = l2_body.find(f".//geom[@mesh='{hand_side}_{prefix}_force_sensor_3']")
                if sensor3 is not None:
                    s_pos_str = sensor3.get('pos', '0 0 0')
                    sensor_pos = np.array([float(x) for x in s_pos_str.split()])

                    finger_transforms['R'].append(np.eye(3))
                    finger_transforms['p'].append(sensor_pos)
                    finger_transforms['joint_names'].append(f'{finger_name}_tip')
                else:
                    # Fallback: sensor 2
                    sensor2 = l2_body.find(f".//geom[@mesh='{hand_side}_{prefix}_force_sensor_2']")
                    if sensor2 is not None:
                        s_pos_str = sensor2.get('pos', '0 0 0')
                        sensor_pos = np.array([float(x) for x in s_pos_str.split()])
                        finger_transforms['R'].append(np.eye(3))
                        finger_transforms['p'].append(sensor_pos)
                        finger_transforms['joint_names'].append(f'{finger_name}_tip')

        transforms[finger_name] = finger_transforms

    return transforms


def _extract_psyonic_transforms(xml_path: str, hand_side: str) -> Dict:
    """Parse frame transformations R, p, h for the Psyonic Ability hand.

    Near-verbatim port of the G1 monolith's ``get_psyonic_frame_transforms_from_xml``.
    Naming: left hand uses 'l_' prefix (l_ability_hand, l_index_mcp, ...);
    right hand uses no prefix. Chain per finger: hand base -> prox/metacarp ->
    int_dist/mcp -> tip site.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    prefix = 'l_' if hand_side == 'left' else ''
    hand_body_name = f'{prefix}ability_hand'

    finger_joints = {
        'thumb': [f'{prefix}thumb_cmc', f'{prefix}thumb_mcp'],
        'index': [f'{prefix}index_mcp', f'{prefix}index_pip'],
        'middle': [f'{prefix}middle_mcp', f'{prefix}middle_pip'],
        'ring': [f'{prefix}ring_mcp', f'{prefix}ring_pip'],
        'pinky': [f'{prefix}pinky_mcp', f'{prefix}pinky_pip']
    }

    finger_tips = {
        'thumb': f'{prefix}thumb_tip',
        'index': f'{prefix}index_tip',
        'middle': f'{prefix}middle_tip',
        'ring': f'{prefix}ring_tip',
        'pinky': f'{prefix}pinky_tip'
    }

    transforms = {}

    hand_body = _find_body_in_xml(root, hand_body_name)
    if hand_body is None:
        print(f"Warning: Could not find {hand_body_name} in XML")
        return transforms

    hand_base_R, hand_base_p = _parse_transform_from_body(hand_body)

    for finger_name, joint_names in finger_joints.items():
        finger_transforms = {
            'R': [],
            'p': [],
            'h': [],
            'joint_names': joint_names.copy()
        }

        if finger_name == 'thumb':
            # Thumb: hand -> thumb_metacarp -> thumb_mcp -> tip
            finger_body_name = f'{prefix}thumb_metacarp'
        else:
            # Fingers: hand -> {finger}_prox -> {finger}_int_dist -> tip
            finger_body_name = f'{prefix}{finger_name}_prox'

        finger_body = _find_body_in_xml(hand_body, finger_body_name)

        if finger_body is None:
            print(f"Warning: Could not find {finger_body_name} in {hand_body_name}")
            continue

        # Transform from wrist to hand base
        finger_transforms['R'].append(hand_base_R)
        finger_transforms['p'].append(hand_base_p)

        if finger_name == 'thumb':
            metacarp_body = finger_body
            mcp_body = _find_body_in_xml(metacarp_body, f'{prefix}thumb_mcp')

            if mcp_body is None:
                print(f"Warning: Could not find {prefix}thumb_mcp in thumb kinematic chain")
                continue

            # Transform from hand to thumb_metacarp
            R_metacarp, p_metacarp = _parse_transform_from_body(metacarp_body)
            finger_transforms['R'].append(R_metacarp)
            finger_transforms['p'].append(p_metacarp)

            # CMC joint axis
            cmc_joint = metacarp_body.find(f".//joint[@name='{prefix}thumb_cmc']")
            if cmc_joint is not None:
                axis_str = cmc_joint.get('axis', '0 0 1')
                h_cmc = np.array([float(x) for x in axis_str.split()])
                finger_transforms['h'].append(h_cmc)

            # Transform from thumb_metacarp to thumb_mcp
            R_mcp, p_mcp = _parse_transform_from_body(mcp_body)
            finger_transforms['R'].append(R_mcp)
            finger_transforms['p'].append(p_mcp)

            # MCP joint axis
            mcp_joint = mcp_body.find(f".//joint[@name='{prefix}thumb_mcp']")
            if mcp_joint is not None:
                axis_str = mcp_joint.get('axis', '0 0 1')
                h_mcp = np.array([float(x) for x in axis_str.split()])
                finger_transforms['h'].append(h_mcp)

        else:
            prox_body = finger_body
            int_dist_body = _find_body_in_xml(prox_body, f'{prefix}{finger_name}_int_dist')

            if int_dist_body is None:
                print(f"Warning: Could not find {prefix}{finger_name}_int_dist in {finger_name} kinematic chain")
                continue

            # Transform from hand to finger proximal (MCP)
            R_prox, p_prox = _parse_transform_from_body(prox_body)
            finger_transforms['R'].append(R_prox)
            finger_transforms['p'].append(p_prox)

            # MCP joint axis
            mcp_joint = prox_body.find(f".//joint[@name='{prefix}{finger_name}_mcp']")
            if mcp_joint is not None:
                axis_str = mcp_joint.get('axis', '0 0 1')
                h_mcp = np.array([float(x) for x in axis_str.split()])
                finger_transforms['h'].append(h_mcp)

            # Transform from finger proximal to intermediate/distal (PIP)
            R_int_dist, p_int_dist = _parse_transform_from_body(int_dist_body)
            finger_transforms['R'].append(R_int_dist)
            finger_transforms['p'].append(p_int_dist)

            # PIP joint axis
            pip_joint = int_dist_body.find(f".//joint[@name='{prefix}{finger_name}_pip']")
            if pip_joint is not None:
                axis_str = pip_joint.get('axis', '0 0 1')
                h_pip = np.array([float(x) for x in axis_str.split()])
                finger_transforms['h'].append(h_pip)

        # Transformation to fingertip (site)
        tip_site_name = finger_tips[finger_name]
        current_body = mcp_body if finger_name == 'thumb' else int_dist_body
        tip_site = _find_site_in_xml(current_body, tip_site_name) if current_body is not None else None

        if tip_site is not None:
            pos_str = tip_site.get('pos', '0 0 0')
            tip_pos = np.array([float(x) for x in pos_str.split()])
            finger_transforms['R'].append(np.eye(3))  # No rotation for tip
            finger_transforms['p'].append(tip_pos)
            finger_transforms['joint_names'].append(tip_site_name)

        transforms[finger_name] = finger_transforms

    return transforms
