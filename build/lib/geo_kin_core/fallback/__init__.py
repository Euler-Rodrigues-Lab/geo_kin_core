"""Public mink differential-IK fallback solver.

Lets public robot repos and their CI run teleop end-to-end without a license;
the licensed `geo_kin` wheel is a drop-in upgrade via resolve_session().

This is a plain differential-IK wrist/torso tracker (mink FrameTasks + posture
regularizer + joint limits). It intentionally contains NO SEW-geometric arm IK
and NO hand IK. Requires the 'fallback' extra (mink, quadprog, scipy).
"""

from .session import MinkFallbackSession

__all__ = ["MinkFallbackSession"]
