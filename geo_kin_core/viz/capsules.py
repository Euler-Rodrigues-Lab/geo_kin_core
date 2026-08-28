"""Minimal capsule/sphere overlay primitives for a MuJoCo passive viewer.

Shared by every robot repo's demos so overlay code is written once. Needs the
``viz`` extra (mujoco). All helpers no-op when the scene is full instead of
raising — an overlay must never take down a control loop.
"""

from __future__ import annotations

import numpy as np


def _scene(viewer):
    scn = getattr(viewer, "user_scn", None)
    return scn if scn is not None and scn.ngeom < scn.maxgeom else None


def clear(viewer) -> None:
    """Drop all user geoms (call once per rendered frame, before drawing)."""
    scn = getattr(viewer, "user_scn", None)
    if scn is not None:
        scn.ngeom = 0


def add_capsule(viewer, p1, p2, radius: float, rgba=(0.2, 0.6, 1.0, 0.5)) -> bool:
    """Capsule between two world points. Returns False if not drawn."""
    import mujoco

    scn = _scene(viewer)
    if scn is None:
        return False
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    if p1.shape != (3,) or p2.shape != (3,) or not np.isfinite(p1).all() or not np.isfinite(p2).all():
        return False
    diff = p2 - p1
    dist = float(np.linalg.norm(diff))
    if dist < 1e-6:
        return False
    z_axis = diff / dist
    ref = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.999 else np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(ref, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[radius, dist / 2, 0],
        pos=(p1 + p2) / 2,
        mat=np.column_stack([x_axis, y_axis, z_axis]).flatten(),
        rgba=np.asarray(rgba, dtype=np.float64),
    )
    scn.ngeom += 1
    return True


def add_sphere(viewer, position, radius: float, rgba=(0.2, 0.6, 1.0, 0.5)) -> bool:
    """Sphere at a world point. Returns False if not drawn."""
    import mujoco

    scn = _scene(viewer)
    if scn is None:
        return False
    position = np.asarray(position, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        return False
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0, 0],
        pos=position,
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float64),
    )
    scn.ngeom += 1
    return True


def add_chain(viewer, points, radius: float, rgba=(0.2, 0.6, 1.0, 0.5)) -> int:
    """Capsules through consecutive points. Returns how many were drawn."""
    points = [p for p in points if p is not None]
    return sum(add_capsule(viewer, a, b, radius, rgba) for a, b in zip(points, points[1:]))
