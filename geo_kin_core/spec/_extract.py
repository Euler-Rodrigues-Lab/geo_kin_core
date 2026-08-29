"""Model extractors: URDF/MJCF -> transform dicts consumed by the geo-kin solvers.

These are near-verbatim ports of the reference extractors so the produced
dicts are STRUCTURALLY and NUMERICALLY identical to what the solvers were
developed against:

- G1 robot parts: ``get_frame_transforms_from_pinocchio`` (G1 full-body monolith)
- RBY1 robot parts: ``get_frame_transforms_from_pinocchio`` (rby1_teleop
  monolith ``geometric_kinematics_rby1.py`` — a DIFFERENT function of the same
  name: it additionally extracts joint limits, falls back to pinocchio frames
  for URDF fixed joints, and appends the end-effector frame transform)
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

#: RBY1 (Rainbow Robotics RB-Y1) parts, verbatim from the monolith's
#: rby1_teleop/geometric_kinematics_rby1.py get_frame_transforms_from_pinocchio.
RBY1_JOINT_SEQUENCES = {
    "right_arm": ["right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3",
                  "right_arm_4", "right_arm_5", "right_arm_6"],
    "left_arm": ["left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3",
                 "left_arm_4", "left_arm_5", "left_arm_6"],
    # Include shoulder bases + head_base for torso IK context
    "torso": ["torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5",
              "right_arm_0", "left_arm_0", "head_base"],
    "base": ["right_wheel", "left_wheel"],
    # Keep head_base so head FK can be expressed in head_base frame
    "head": ["head_base", "head_0", "head_1"],
    # Full finger sequences for detailed hand analysis (XHand-equipped URDF)
    "right_hand_thumb": ["right_hand_thumb_bend_joint", "right_hand_thumb_rota_joint1", "right_hand_thumb_rota_joint2"],
    "left_hand_thumb": ["left_hand_thumb_bend_joint", "left_hand_thumb_rota_joint1", "left_hand_thumb_rota_joint2"],
    "right_hand_index": ["right_hand_index_bend_joint", "right_hand_index_joint1", "right_hand_index_joint2"],
    "left_hand_index": ["left_hand_index_bend_joint", "left_hand_index_joint1", "left_hand_index_joint2"],
    "right_hand_mid": ["right_hand_mid_joint1", "right_hand_mid_joint2"],
    "left_hand_mid": ["left_hand_mid_joint1", "left_hand_mid_joint2"],
    "right_hand_ring": ["right_hand_ring_joint1", "right_hand_ring_joint2"],
    "left_hand_ring": ["left_hand_ring_joint1", "left_hand_ring_joint2"],
    "right_hand_pinky": ["right_hand_pinky_joint1", "right_hand_pinky_joint2"],
    "left_hand_pinky": ["left_hand_pinky_joint1", "left_hand_pinky_joint2"],
}

#: End-effector frames appended to each RBY1 part chain (fixed transform, h=0).
RBY1_END_EFFECTOR_FRAMES = {
    "right_arm": "link_right_arm_6",
    "left_arm": "link_left_arm_6",
    "torso": "link_torso_4",
    "base": "base_link",
    "head": "link_head_2",
    "right_hand": "right_hand_ee_link",
    "left_hand": "left_hand_ee_link",
    "right_hand_thumb": "right_hand_thumb_rota_tip",
    "left_hand_thumb": "left_hand_thumb_rota_tip",
    "right_hand_index": "right_hand_index_rota_tip",
    "left_hand_index": "left_hand_index_rota_tip",
    "right_hand_mid": "right_hand_mid_tip",
    "left_hand_mid": "left_hand_mid_tip",
    "right_hand_ring": "right_hand_ring_tip",
    "left_hand_ring": "left_hand_ring_tip",
    "right_hand_pinky": "right_hand_pinky_tip",
    "left_hand_pinky": "left_hand_pinky_tip",
}

RBY1_PARTS: List[str] = list(RBY1_JOINT_SEQUENCES.keys())

#: Per-robot joint-sequence tables the URDF extractor knows about.
ROBOT_SEQUENCE_TABLES = {
    "g1": ROBOT_JOINT_SEQUENCES,
    "rby1": RBY1_JOINT_SEQUENCES,
}

HAND_TYPES = ("inspire", "psyonic", "xhand")


# ---------------------------------------------------------------------------
# Robot (URDF via pinocchio)
# ---------------------------------------------------------------------------

def generate_robot_spec(urdf_path: str, parts: Sequence[str], robot: str = "g1") -> Dict:
    """Extract per-part frame transforms from a URDF.

    Args:
        urdf_path: Path to the robot URDF.
        parts: Part names to extract (subset of the robot's sequence table).
        robot: Which robot's joint-sequence table / extractor conventions to
            use ('g1' or 'rby1'). The extractors are near-verbatim ports of the
            respective monolith functions and differ deliberately: 'rby1' also
            extracts joint limits, resolves URDF fixed joints as pinocchio
            frames, and appends the part's end-effector frame transform.

    Returns:
        ``{part: {'R': [3x3, ...], 'p': [(3,), ...], 'h': [(3,), ...],
        'joint_names': [str, ...]}}`` — each part dict structurally identical
        to the monolith's ``get_frame_transforms_from_pinocchio`` output
        ('rby1' parts additionally carry ``joint_lower``/``joint_upper``).
    """
    import pinocchio as pin  # heavy dep, lazy

    if robot not in ROBOT_SEQUENCE_TABLES:
        raise ValueError(
            f"Unknown robot {robot!r}. Available: {sorted(ROBOT_SEQUENCE_TABLES)}")
    sequences = ROBOT_SEQUENCE_TABLES[robot]

    parts = list(parts)
    unknown = [p for p in parts if p not in sequences]
    if unknown:
        raise ValueError(f"Unknown parts: {unknown}. Available: {list(sequences)}")

    model = pin.buildModelFromUrdf(str(urdf_path))
    if robot == "rby1":
        return {part: _extract_rby1_part(model, part) for part in parts}
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


def _extract_rby1_part(model, part_name: str) -> Dict:
    """Extract R_i_i+1, p_i_i+1, joint axes h_i (+ limits, + EE frame) for one RBY1 part.

    Near-verbatim port of the rby1_teleop monolith's
    ``get_frame_transforms_from_pinocchio`` (geometric_kinematics_rby1.py).
    Differences vs :func:`_extract_robot_part` are intentional monolith
    behavior: joint limits are extracted per joint (0.0 for fixed frames),
    URDF fixed joints (e.g. ``head_base``) resolve through the pinocchio frame
    fallback with a zero axis, and the part's end-effector frame transform is
    appended (its name appended to joint_names).
    """
    if part_name not in RBY1_JOINT_SEQUENCES:
        raise ValueError(
            f"Unknown part_name: {part_name}. Available: {RBY1_PARTS}")

    joint_names = RBY1_JOINT_SEQUENCES[part_name]
    R_transforms = []
    p_transforms = []
    h_transforms = []
    joint_lower = []
    joint_upper = []

    for joint_name in joint_names:
        try:
            # URDF fixed joints are often represented as frames (not model
            # joints) in Pinocchio.
            joint_id = model.getJointId(joint_name)
            joint_is_direct_match = (
                joint_id != 0
                and joint_id < len(model.names)
                and model.names[joint_id] == joint_name
            )

            if joint_is_direct_match:
                joint = model.joints[joint_id]
                M_local = model.jointPlacements[joint_id]

                R_i_iplus1 = M_local.rotation.copy()
                p_i_iplus1 = M_local.translation.copy()

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

                # Extract joint limits if joint has configuration variables
                if joint.nq > 0:
                    idx_q = joint.idx_q
                    lower = model.lowerPositionLimit[idx_q:idx_q + joint.nq].copy()
                    upper = model.upperPositionLimit[idx_q:idx_q + joint.nq].copy()
                    if joint.nq == 1:
                        lower = lower[0]
                        upper = upper[0]
                    joint_lower.append(lower)
                    joint_upper.append(upper)
                else:
                    joint_lower.append(0.0)
                    joint_upper.append(0.0)
            elif model.existFrame(joint_name):
                # Fallback path for fixed joints represented as frames.
                frame_id = model.getFrameId(joint_name)
                frame = model.frames[frame_id]
                M_local = frame.placement

                R_i_iplus1 = M_local.rotation.copy()
                p_i_iplus1 = M_local.translation.copy()
                h_i = np.array([0.0, 0.0, 0.0])
                joint_lower.append(0.0)
                joint_upper.append(0.0)
            else:
                raise ValueError(f"Joint/frame {joint_name} not found in Pinocchio model")

            R_transforms.append(R_i_iplus1)
            p_transforms.append(p_i_iplus1)
            h_transforms.append(h_i)

        except Exception as e:
            print(f"Error processing joint {joint_name}: {e}")
            continue

    # Add transformation from last joint to end effector frame if it exists
    end_effector_frame = RBY1_END_EFFECTOR_FRAMES.get(part_name)
    ee_exists = False
    if end_effector_frame:
        try:
            if model.existFrame(end_effector_frame):
                ee_exists = True
                frame_id = model.getFrameId(end_effector_frame)
                frame = model.frames[frame_id]

                M_joint_to_ee = frame.placement
                R_transforms.append(M_joint_to_ee.rotation.copy())
                p_transforms.append(M_joint_to_ee.translation.copy())
                h_transforms.append(np.array([0.0, 0.0, 0.0]))
                joint_lower.append(0.0)
                joint_upper.append(0.0)
            else:
                print(f"Warning: End effector frame {end_effector_frame} not found in model")
        except Exception as e:
            print(f"Error adding end effector frame {end_effector_frame}: {e}")

    return {
        'R': R_transforms,
        'p': p_transforms,
        'h': h_transforms,
        'joint_lower': joint_lower,
        'joint_upper': joint_upper,
        'joint_names': joint_names + ([end_effector_frame] if ee_exists else []),
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
    if hand_type == "xhand":
        return _extract_xhand_transforms(str(xml_path), side)
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


def _extract_xhand_transforms(xml_path: str, hand_side: str) -> Dict:
    """Parse frame transformations R, p, h (+ joint limits) for the XHand.

    Near-verbatim port of the xhand_teleop monolith's
    ``get_fingers_frame_transforms_from_xml`` (geometric_kinematics_xhand.py),
    minus the ``transforms['side']`` string entry (the side is carried in the
    npz metadata instead; runtime consumers re-inject it).

    Source MJCF: the standalone hand files (xhand_right.xml / xhand_left.xml).
    Chain per finger is rooted at the wrist body ``<side>`` through
    ``<side>_eef`` -> ``x_<side>_hand_root``; the FIRST body transform of each
    finger is re-expressed in the WRIST frame.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    transforms = {}

    wrist_body = None
    for elem in root.iter('body'):
        if elem.get('name') == hand_side:
            wrist_body = elem
            break

    if wrist_body is None:
        print(f"Could not find wrist body {hand_side}")
        return {}
    else:
        R_wrist, p_wrist = _parse_transform_from_body(wrist_body)
        T_wrist = np.eye(4)
        T_wrist[:3, :3] = R_wrist
        T_wrist[:3, 3] = p_wrist

    eef_name = f"{hand_side}_eef"
    eef_body = _find_body_in_xml(wrist_body, eef_name)

    if eef_body is None:
        print(f"Could not find eef body {eef_name}")
        return {}
    else:
        R_eef, p_eef = _parse_transform_from_body(eef_body)
        T_eef = np.eye(4)
        T_eef[:3, :3] = R_eef
        T_eef[:3, 3] = p_eef

    # The root body for the hand is x_{side}_hand_root
    hand_root_name = f"x_{hand_side}_hand_root"
    hand_root_body = _find_body_in_xml(wrist_body, hand_root_name)

    if hand_root_body is None:
        print(f"Could not find hand root body {hand_root_name}")
        return {}
    else:
        R_hand_root, p_hand_root = _parse_transform_from_body(hand_root_body)
        T_hand_root = np.eye(4)
        T_hand_root[:3, :3] = R_hand_root
        T_hand_root[:3, 3] = p_hand_root

    # In xhand_right.xml: right -> right_eef -> x_right_hand_root
    T_w_r = T_wrist @ T_eef @ T_hand_root

    # Define chains for each finger relative to hand root.
    # Exactly 4 transforms [Base, L1, L2, Tip] for thumb/index and
    # 3 [Base, L1, Tip] for the rest, with 'fixed' entries for jointless bodies.
    chains_config = {
        'thumb': [
            f"{hand_side}_hand_thumb_bend_link",    # L1
            f"{hand_side}_hand_thumb_rota_link1",   # L2
            f"{hand_side}_hand_thumb_rota_link2",   # L3
            f"{hand_side}_hand_thumb_rota_tip"      # L4
        ],
        'index': [
            f"{hand_side}_hand_index_bend_link",    # L1
            f"{hand_side}_hand_index_rota_link1",   # L2
            f"{hand_side}_hand_index_rota_link2",   # L3
            f"{hand_side}_hand_index_rota_tip"      # L4
        ],
        'middle': [
            f"{hand_side}_hand_mid_link1",
            f"{hand_side}_hand_mid_link2",
            f"{hand_side}_hand_mid_tip"
        ],
        'ring': [
            f"{hand_side}_hand_ring_link1",
            f"{hand_side}_hand_ring_link2",
            f"{hand_side}_hand_ring_tip"
        ],
        'pinky': [
            f"{hand_side}_hand_pinky_link1",
            f"{hand_side}_hand_pinky_link2",
            f"{hand_side}_hand_pinky_tip"
        ]
    }

    for finger, body_names in chains_config.items():
        R_list = []
        p_list = []
        h_list = []
        joint_names = []
        joint_lower_list = []
        joint_upper_list = []

        # Traverse and parse bodies from XML
        current_parent = hand_root_body
        parsed_bodies = []
        valid_chain = True

        for name in body_names:
            body = _find_body_in_xml(current_parent, name)
            if body is None:
                print(f"Warning: {name} not found for {finger}")
                valid_chain = False
                break
            R, p = _parse_transform_from_body(body)

            # Get axis
            j_elem = body.find(".//joint")
            if j_elem is not None:
                axis_str = j_elem.get('axis', '0 0 1')
                h = np.array([float(x) for x in axis_str.split()])
                j_name = j_elem.get('name', 'unknown')
                range_str = j_elem.get('range', None)
                if range_str:
                    bounds = [float(x) for x in range_str.split()]
                    j_lower, j_upper = bounds[0], bounds[1]
                else:
                    j_lower, j_upper = -np.pi, np.pi
            else:
                h = np.zeros(3)
                j_name = 'fixed'
                j_lower, j_upper = 0.0, 0.0

            parsed_bodies.append({'R': R, 'p': p, 'h': h, 'j': j_name,
                                  'lower': j_lower, 'upper': j_upper})
            current_parent = body

        if not valid_chain or not parsed_bodies:
            continue

        # Adjust the first body's transform to be in the WRIST frame!
        T_0_local = np.eye(4)
        T_0_local[:3, :3] = parsed_bodies[0]['R']
        T_0_local[:3, 3] = parsed_bodies[0]['p']

        # T_w_r is transforms from wrist to hand root
        T_0_wrist = T_w_r @ T_0_local
        parsed_bodies[0]['R'] = T_0_wrist[:3, :3]
        parsed_bodies[0]['p'] = T_0_wrist[:3, 3]

        # Construct Output Lists
        for b in parsed_bodies:
            R_list.append(b['R'])
            p_list.append(b['p'])
            h_list.append(b['h'])
            joint_names.append(b['j'])
            joint_lower_list.append(b['lower'])
            joint_upper_list.append(b['upper'])

        transforms[finger] = {
            'R': R_list,
            'p': p_list,
            'h': h_list,
            'joint_names': joint_names,
            'joint_lower': joint_lower_list,
            'joint_upper': joint_upper_list,
        }

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
