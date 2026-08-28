"""Solver-diagnostic overlays (post-XPBD filtered SEW capsules)."""

from __future__ import annotations

from . import capsules


def _capsules_from_session(session):
    """(a, b, r) triples of the last filtered SEW, across backend flavours.

    Two shapes are supported: the licensed wheel's ``sew_capsules()`` method
    (``[(name, a, b, r), ...]``) and the reference session's
    ``last_filtered_sew`` + ``sew_filter.parse_sew`` / ``sew_to_capsules``.
    """
    method = getattr(session, "sew_capsules", None)
    if callable(method):
        return [(a, b, r) for _name, a, b, r in method()]
    filtered = getattr(session, "last_filtered_sew", None)
    sew_filter = getattr(session, "sew_filter", None)
    if filtered is None or sew_filter is None:
        return []
    return [(c.a, c.b, c.r)
            for c in sew_filter.sew_to_capsules(sew_filter.parse_sew(filtered)).values()]


def draw_filtered_sew(viewer, session, to_world=None, rgba=(0.2, 0.9, 0.2, 0.35)) -> int:
    """Draw a session's filtered SEW capsules; returns the number drawn.

    Backend-agnostic (see :func:`_capsules_from_session`); silently draws
    nothing when the backend exposes no filter state (e.g. the mink fallback,
    or before the first filtered solve) — an overlay must never take down a
    control loop.

    Args:
        to_world: maps solver-frame points into world (e.g. the robot
            controller's ``get_sew_transform()``); identity when None.
    """
    to_world = (lambda p: p) if to_world is None else to_world
    try:
        return sum(capsules.add_capsule(viewer, to_world(a), to_world(b), float(r), rgba)
                   for a, b, r in _capsules_from_session(session))
    except Exception:
        return 0
