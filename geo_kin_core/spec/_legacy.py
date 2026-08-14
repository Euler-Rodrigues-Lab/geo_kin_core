"""Converter: geo_retarget-format legacy transform npz -> geo_kin_core.spec/1 npz.

The legacy caches (written by geo_retarget's ``utils/transform_utils.py:
save_transforms``) hold, per file, ONE part::

    R            (n,3,3) float64   local link rotations
    p            (n,3)   float64   local link translations
    h            (n,3)   float64   joint axes (rows of zeros = fixed frames,
                                   e.g. a trailing tool/fingertip frame)
    joint_lower  (n,)    float64   joint lower limits (0.0 on fixed frames)
    joint_upper  (n,)    float64   joint upper limits
    joint_names  (n,)    str
    part_name    ()      str
    urdf_source_path ()  str       path of the source URDF at generation time
    urdf_content ()      str       FULL source URDF text (the model signature)

:func:`convert_legacy_npz` merges one or more such files (which must all embed
the same source URDF) into a single spec/1 npz. Arrays are carried over
bit-exactly. The spec's ``meta/source_sha256`` is the sha256 of the embedded
URDF text (utf-8) — the legacy caches embed the text itself, not a hash — and
``meta/signature_scheme`` records that provenance so consumers don't try to
hash a URDF *file* against it.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Optional, Sequence, Union

import numpy as np

from ._io import _spec_arrays

_LEGACY_REQUIRED = ("R", "p", "h", "joint_lower", "joint_upper",
                    "joint_names", "part_name")


def _load_legacy(path: str) -> Dict:
    with np.load(path, allow_pickle=False) as data:
        missing = [k for k in _LEGACY_REQUIRED if k not in data.files]
        if missing:
            raise ValueError(f"{path}: not a geo_retarget legacy transform npz "
                             f"(missing {missing})")
        out = {k: data[k] for k in data.files}
    return out


def convert_legacy_npz(
    legacy_paths: Union[Sequence[str], Dict[str, str]],
    npz_path: str,
    *,
    kind: str,
    extra_meta: Optional[Dict[str, str]] = None,
) -> str:
    """Convert geo_retarget legacy transform cache(s) into ONE spec/1 npz.

    Args:
        legacy_paths: Paths of the legacy npz files to merge (one part each).
            A dict maps part name -> path (overriding the embedded
            ``part_name``); a sequence uses each file's ``part_name``.
        npz_path: Output spec npz path.
        kind: "robot" (e.g. Vega arms/torso/head) or "hand" (e.g. Sharpa
            fingers).
        extra_meta: Optional extra string metadata (e.g. side, hand_type).

    Returns:
        The npz path written.

    Raises:
        ValueError: On a malformed legacy file, duplicate part names, or if the
            files embed different source URDFs.
    """
    if isinstance(legacy_paths, dict):
        items = list(legacy_paths.items())
    else:
        items = [(None, p) for p in legacy_paths]
    if not items:
        raise ValueError("legacy_paths is empty; nothing to convert")

    spec: Dict[str, Dict] = {}
    shapes: Dict[str, Dict] = {}
    urdf_content = None
    urdf_source_path = None
    legacy_names = []

    for name, path in items:
        legacy = _load_legacy(path)
        part = str(name) if name is not None else str(legacy["part_name"])
        if part in spec:
            raise ValueError(f"duplicate part name {part!r} (from {path})")

        content = str(legacy["urdf_content"]) if "urdf_content" in legacy else None
        if urdf_content is None:
            urdf_content = content
            if "urdf_source_path" in legacy:
                urdf_source_path = str(legacy["urdf_source_path"])
        elif content != urdf_content:
            raise ValueError(
                f"{path}: embedded URDF differs from the other legacy files; "
                "convert caches from different models into separate spec npz files")

        spec[part] = {
            "R": legacy["R"],
            "p": legacy["p"],
            "h": legacy["h"],
            "joint_names": [str(n) for n in legacy["joint_names"]],
            "joint_lower": legacy["joint_lower"],
            "joint_upper": legacy["joint_upper"],
        }
        h = np.asarray(legacy["h"], dtype=np.float64)
        shapes[part] = {
            "n_frames": int(h.shape[0]),
            "n_joints": int(np.sum(np.linalg.norm(h, axis=1) > 1e-8)),
        }
        legacy_names.append(os.path.basename(str(path)))

    if urdf_content is None:
        raise ValueError("legacy files embed no urdf_content; cannot derive a "
                         "source signature")

    meta = {
        "signature_scheme": "sha256 of the URDF text embedded in the legacy "
                            "geo_retarget cache (urdf_content), utf-8 encoded "
                            "-- NOT a hash of a file on disk",
        "provenance": "converted from geo_retarget legacy transform caches by "
                      "geo_kin_core.spec.convert_legacy_npz",
        "legacy_files": ",".join(legacy_names),
        "part_shapes": json.dumps(shapes, sort_keys=True),
    }
    if urdf_source_path:
        # Basename ONLY: the legacy caches embed the generating machine's full
        # URDF path, which must not leak into (or ship inside) the spec npz.
        meta["legacy_urdf_source_path"] = os.path.basename(urdf_source_path)
    meta.update(extra_meta or {})

    arrays = _spec_arrays(
        spec,
        kind=kind,
        source_filename=os.path.basename(urdf_source_path) if urdf_source_path
        else "unknown.urdf",
        source_sha256=hashlib.sha256(urdf_content.encode("utf-8")).hexdigest(),
        extra_meta=meta,
    )
    np.savez(npz_path, **arrays)
    return str(npz_path)
