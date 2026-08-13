"""Robot/hand spec tooling: URDF or MJCF -> (R, p, h, joint_names) npz cache.

Heavy parsing (pinocchio for URDF; ElementTree + scipy for MJCF hands) happens
once at dev time inside the generate_* functions; runtime consumers — the
fallback, geo_kin_ref, and the private Rust build (which embeds the npz) —
only call :func:`load_spec`, which needs numpy alone. The source file's sha256
is embedded in the npz so any consumer can detect model drift via
:func:`verify_signature`.

The dicts produced/loaded are structurally identical to the transform dicts
the reference solvers consume:

- robot: ``{part: {'R': [3x3]*n, 'p': [(3,)]*n, 'h': [(3,)]*n,
  'joint_names': [str]*n}}`` for parts in :data:`ROBOT_PARTS`.
- hand: ``{finger: {'R': [3x3]*(n+2), 'p': [(3,)]*(n+2), 'h': [(3,)]*n,
  'joint_names': [str]*(n+1)}}`` for thumb/index/middle/ring/pinky
  (base + per-joint links + tip; tip name appended to joint_names).

CLI: ``python -m geo_kin_core.spec generate --help``.
"""

from ._extract import (
    HAND_TYPES,
    ROBOT_PARTS,
    generate_hand_spec,
    generate_robot_spec,
)
from ._io import (
    GENERATOR_VERSION,
    SCHEMA,
    load_spec,
    save_spec,
    sha256_of_file,
    verify_signature,
)

__all__ = [
    "HAND_TYPES",
    "ROBOT_PARTS",
    "GENERATOR_VERSION",
    "SCHEMA",
    "generate_hand_spec",
    "generate_robot_spec",
    "load_spec",
    "save_spec",
    "sha256_of_file",
    "verify_signature",
]
