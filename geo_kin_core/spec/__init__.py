"""Robot/hand spec tooling: URDF or MJCF -> (R, p, h, limits, joint_names) npz cache.

Design follows the transform-cache pattern proven in prior work: heavy parsing
(pinocchio for URDF, mujoco for MJCF) happens once at dev time via the
`spec-tools` extra; runtime consumers — the fallback, geo_kin_ref, and the
private Rust build (which embeds the npz) — only load the cache. The source
file's hash is embedded in the npz so any consumer can detect model drift.

Hand specs additionally record: finger chains with n_links = n_joints + 1 and a
separate wrist_to_hand mount transform (supplied by the arm it mounts on).

To implement (task #2): RobotSpec / HandSpec dataclasses, generate_spec(),
load_spec(), CLI entry point `geo-kin-spec`.
"""
