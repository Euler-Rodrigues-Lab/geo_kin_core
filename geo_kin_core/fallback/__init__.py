"""Public mink differential-IK fallback solver.

Lets public robot repos and their CI run teleop end-to-end without a license;
the licensed `geo_kin` wheel is a drop-in upgrade via resolve_session().

This is a differential-IK wrist/palm, torso, elbow, and head tracker (MINK
FrameTasks + posture regularizer + joint limits). It intentionally contains no
SEW-geometric arm IK and no analytical hand IK. RBY1 and G1 presets are
included. Requires the 'fallback' extra (mink, quadprog, scipy).
"""

from .session import MinkFallbackSession

__all__ = ["MinkFallbackSession"]
