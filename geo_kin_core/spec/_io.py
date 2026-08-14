"""npz serialization for robot/hand specs.

The npz embeds the sha256 + filename of the source model file, the parts list,
and the generator version, so any consumer (Python reference, Rust build) can
detect model drift with :func:`verify_signature`. Loading needs numpy only.

npz schema (version ``geo_kin_core.spec/1``):

    meta/schema             0-d str   "geo_kin_core.spec/1"
    meta/generator_version  0-d str   e.g. "0.1.0"
    meta/kind               0-d str   "robot" | "hand"
    meta/source_filename    0-d str   basename of the source URDF/MJCF
    meta/source_sha256      0-d str   hex sha256 of the source file bytes
    meta/parts              1-d str   part names (robot) / finger names (hand)
    meta/<extra>            0-d str   extra metadata (e.g. side, hand_type)
    <part>/R                (nR,3,3)  float64 link rotations
    <part>/p                (nR,3)    float64 link translations
    <part>/h                (nH,3)    float64 joint axes
    <part>/joint_names      (nN,)     str     joint (+ tip) names
    <part>/joint_lower      (nL,)     float64 OPTIONAL joint lower limits
    <part>/joint_upper      (nL,)     float64 OPTIONAL joint upper limits

Per-part nR/nH/nN are arbitrary; conventions differ per source:

- G1-style robot parts: nR == nH == nN == n_joints.
- inspire/psyonic hand fingers: nR == n_joints + 2 (base + links + tip),
  nH == n_joints, nN == n_joints + 1 (tip name appended).
- geo_retarget legacy caches (see :func:`convert_legacy_npz` in ._legacy):
  nR == nH == nN == nL == n_joints + trailing fixed frames (h rows of zeros),
  e.g. Vega arms 7 joints + 1 tool frame, Sharpa fingers n joints + 1 tip.

joint_lower/joint_upper are optional (present when the source carried limits,
e.g. legacy geo_retarget caches); loaders must not assume them.
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, Optional, Tuple

import numpy as np

GENERATOR_VERSION = "0.1.0"
SCHEMA = "geo_kin_core.spec/1"

_META_PREFIX = "meta/"
_REQUIRED_META = ("schema", "generator_version", "kind", "source_filename",
                  "source_sha256", "parts")


def sha256_of_file(path: str) -> str:
    """Hex sha256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spec_arrays(spec: Dict, *, kind: str, source_filename: str,
                 source_sha256: str,
                 extra_meta: Optional[Dict[str, str]] = None) -> Dict:
    """Assemble the flat npz array dict for a spec (shared serialization core)."""
    if kind not in ("robot", "hand"):
        raise ValueError(f"kind must be 'robot' or 'hand', got {kind!r}")
    if not spec:
        raise ValueError("spec is empty; nothing to save")

    parts = list(spec.keys())
    for name in parts:
        if "/" in name:
            raise ValueError(f"part name may not contain '/': {name!r}")

    arrays = {
        f"{_META_PREFIX}schema": np.asarray(SCHEMA),
        f"{_META_PREFIX}generator_version": np.asarray(GENERATOR_VERSION),
        f"{_META_PREFIX}kind": np.asarray(kind),
        f"{_META_PREFIX}source_filename": np.asarray(source_filename),
        f"{_META_PREFIX}source_sha256": np.asarray(source_sha256),
        f"{_META_PREFIX}parts": np.asarray(parts),
    }
    for key, value in (extra_meta or {}).items():
        if key in _REQUIRED_META:
            raise ValueError(f"extra_meta key {key!r} collides with required metadata")
        arrays[f"{_META_PREFIX}{key}"] = np.asarray(str(value))

    for part, t in spec.items():
        arrays[f"{part}/R"] = np.stack([np.asarray(x, dtype=np.float64) for x in t["R"]])
        arrays[f"{part}/p"] = np.stack([np.asarray(x, dtype=np.float64) for x in t["p"]])
        arrays[f"{part}/h"] = np.stack([np.asarray(x, dtype=np.float64) for x in t["h"]])
        arrays[f"{part}/joint_names"] = np.asarray(list(t["joint_names"]))
        if "joint_lower" in t or "joint_upper" in t:
            if not ("joint_lower" in t and "joint_upper" in t):
                raise ValueError(f"part {part!r}: joint_lower/joint_upper must "
                                 "be given together")
            arrays[f"{part}/joint_lower"] = np.asarray(t["joint_lower"], dtype=np.float64)
            arrays[f"{part}/joint_upper"] = np.asarray(t["joint_upper"], dtype=np.float64)
    return arrays


def save_spec(spec: Dict, npz_path: str, source_path: str, *, kind: str,
              extra_meta: Optional[Dict[str, str]] = None) -> str:
    """Save a spec dict (from generate_robot_spec / generate_hand_spec) to npz.

    Args:
        spec: ``{part: {'R', 'p', 'h', 'joint_names'}}`` dict; parts may also
            carry optional ``joint_lower``/``joint_upper`` limit arrays.
        npz_path: Output .npz path.
        source_path: The URDF/MJCF the spec was generated from (hashed).
        kind: "robot" or "hand".
        extra_meta: Optional extra string metadata (e.g. side, hand_type).

    Returns:
        The npz path written.
    """
    arrays = _spec_arrays(
        spec, kind=kind,
        source_filename=os.path.basename(str(source_path)),
        source_sha256=sha256_of_file(str(source_path)),
        extra_meta=extra_meta,
    )
    np.savez(npz_path, **arrays)
    return str(npz_path)


def load_spec(npz_path: str, with_meta: bool = False):
    """Load a spec npz back into the transform-dict structure the solvers consume.

    Requires numpy only (no pinocchio/mujoco/scipy).

    Args:
        npz_path: Path to a spec .npz produced by :func:`save_spec`.
        with_meta: If True, return ``(spec, meta)`` where meta is a dict of the
            embedded string metadata (parts as a list of str).

    Returns:
        ``{part: {'R': [3x3 ...], 'p': [(3,) ...], 'h': [(3,) ...],
        'joint_names': [str ...]}}`` — structurally identical to the monolith
        extractors' outputs; optionally ``(spec, meta)``.
    """
    with np.load(npz_path) as data:
        meta = {}
        for key in data.files:
            if key.startswith(_META_PREFIX):
                value = data[key]
                meta[key[len(_META_PREFIX):]] = (
                    value.tolist() if value.ndim else value.item()
                )

        for required in _REQUIRED_META:
            if required not in meta:
                raise ValueError(f"{npz_path}: missing metadata {_META_PREFIX}{required}")
        if meta["schema"] != SCHEMA:
            raise ValueError(f"{npz_path}: unsupported schema {meta['schema']!r} "
                             f"(expected {SCHEMA!r})")

        spec = {}
        for part in meta["parts"]:
            spec[part] = {
                "R": [np.array(R) for R in data[f"{part}/R"]],
                "p": [np.array(p) for p in data[f"{part}/p"]],
                "h": [np.array(h) for h in data[f"{part}/h"]],
                "joint_names": [str(n) for n in data[f"{part}/joint_names"]],
            }
            # Optional joint limits (e.g. converted geo_retarget legacy caches).
            # Lists of floats, matching the legacy load_transforms() layout.
            if f"{part}/joint_lower" in data.files:
                spec[part]["joint_lower"] = data[f"{part}/joint_lower"].tolist()
                spec[part]["joint_upper"] = data[f"{part}/joint_upper"].tolist()

    if with_meta:
        return spec, meta
    return spec


def verify_signature(npz_path: str, source_path: str) -> bool:
    """Check that the source model file still matches the hash embedded in the npz."""
    with np.load(npz_path) as data:
        key = f"{_META_PREFIX}source_sha256"
        if key not in data.files:
            raise ValueError(f"{npz_path}: missing metadata {key}")
        stored = data[key].item()
    return stored == sha256_of_file(str(source_path))
