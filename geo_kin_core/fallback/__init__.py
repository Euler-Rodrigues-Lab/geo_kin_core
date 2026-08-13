"""Public mink differential-IK fallback solver.

Lets public robot repos and their CI run teleop end-to-end without a license;
the licensed `geo_kin` wheel is a drop-in upgrade via resolve_session().
Implementation ports the mink drop-in pattern already proven in the monolith
(same I/O contract as the SEW solver, SEW-specific kwargs accepted and ignored).

To implement (task #2): MinkFallbackSession(RetargetingSolver).
"""
