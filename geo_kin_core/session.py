"""Solver protocol + resolution.

Every solver backend implements RetargetingSolver:
  1. `geo_kin`      — licensed Rust wheel (production)
  2. `geo_kin_ref`  — private Python reference (owner's dev machine only)
  3. fallback       — public mink differential-IK (this package)

resolve_session() picks the best available backend in that order so robot repos
and their CI run out of the box, and the licensed wheel is a drop-in upgrade.
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from .types import RetargetFrame, RetargetOutput


class RetargetingSolver(Protocol):
    def solve(
        self,
        frame: RetargetFrame,
        engaged: bool = True,
        q_current_right: Optional[np.ndarray] = None,
        q_current_left: Optional[np.ndarray] = None,
    ) -> RetargetOutput: ...

    def reset(
        self,
        q_init_right: Optional[np.ndarray] = None,
        q_init_left: Optional[np.ndarray] = None,
    ) -> None: ...


def resolve_session(robot: str, hand: Optional[str] = None, **config) -> RetargetingSolver:
    """Return a solver backend for robot(+hand).

    ``backend='auto'`` uses licensed wheel -> private reference -> MINK.
    ``backend='licensed'|'reference'|'mink'`` selects one explicitly, which is
    useful for offline A/B validation even when the licensed wheel is installed.
    """
    backend = config.pop("backend", "auto")
    if backend not in ("auto", "licensed", "reference", "mink"):
        raise ValueError("backend must be one of: auto, licensed, reference, mink")
    try:
        import geo_kin  # licensed wheel

        # A wheel only contains the robots compiled into it (cargo features);
        # fall through for robots it doesn't carry rather than failing on
        # construction. Older wheels without COMPILED_ROBOTS carry g1+vega.
        compiled = getattr(geo_kin, "COMPILED_ROBOTS", ("g1", "vega"))
        if backend in ("auto", "licensed") and robot in compiled:
            return geo_kin.RetargetSession(robot=robot, hand=hand, **config)
        if backend == "licensed":
            raise ValueError(
                f"licensed geo_kin wheel does not contain robot {robot!r}; "
                f"compiled robots: {sorted(compiled)}"
            )
    except ImportError:
        if backend == "licensed":
            raise
    try:
        import geo_kin_ref  # private reference, dev machine only

        if backend in ("auto", "reference"):
            return geo_kin_ref.make_session(robot=robot, hand=hand, **config)
    except ImportError:
        if backend == "reference":
            raise

    # Public mink differential-IK fallback. Unlike the licensed/reference
    # backends it solves against the robot MJCF, so the caller must say where
    # that lives (robot descriptions ship in the public robot repos, not here).
    from .fallback import MinkFallbackSession
    from .fallback.presets import PRESETS

    model_xml = config.pop("model_xml", None)
    sides = config.pop("sides", None)
    if model_xml is None:
        raise ValueError(
            "resolve_session: neither the licensed `geo_kin` wheel nor the private "
            "`geo_kin_ref` reference is importable, so the public mink differential-IK "
            "fallback would be used — and it needs the robot MJCF. Pass "
            "model_xml=<path to the robot MJCF> (e.g. the g1_29dof_position_ctrl.xml shipped in your "
            f"robot repo's assets). Presets with body/joint names exist for "
            f"{sorted(PRESETS)}; for other robots also pass sides={{...}} "
            "(see geo_kin_core.fallback.MinkFallbackSession)."
        )
    if sides is not None:
        return MinkFallbackSession(model_xml=model_xml, sides=sides, hand=hand, **config)
    if robot in PRESETS:
        return MinkFallbackSession.from_preset(robot, model_xml=model_xml, hand=hand, **config)
    raise ValueError(
        f"resolve_session: no mink-fallback preset for robot {robot!r} (available: "
        f"{sorted(PRESETS)}). Pass sides={{...}} naming the per-arm wrist/palm bodies, "
        "torso body, and joint lists (see geo_kin_core.fallback.MinkFallbackSession)."
    )
