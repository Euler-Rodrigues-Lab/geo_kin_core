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
    """Return the best available solver backend for robot(+hand).

    Order: licensed `geo_kin` wheel -> private `geo_kin_ref` -> mink fallback.
    """
    try:
        import geo_kin  # licensed wheel

        return geo_kin.RetargetSession(robot=robot, hand=hand, **config)
    except ImportError:
        pass
    try:
        import geo_kin_ref  # private reference, dev machine only

        return geo_kin_ref.make_session(robot=robot, hand=hand, **config)
    except ImportError:
        pass
    from .fallback import MinkFallbackSession

    return MinkFallbackSession(robot=robot, hand=hand, **config)
