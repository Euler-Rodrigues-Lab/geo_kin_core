# geo-kin-core

Public core for SEW-geometric teleoperation retargeting:

- **Typed messages** — `RetargetFrame` (human keypoints in), `RetargetOutput`
  (robot joint goals + diagnostics out), replacing the historical duck-typed
  action dicts.
- **`RetargetingSolver` protocol + `resolve_session()`** — robot repos code
  against one interface; the backend is resolved at runtime:
  licensed `geo_kin` wheel → private reference (owner only) → public mink
  differential-IK fallback.
- **Spec tool** (`spec-tools` extra) — URDF/MJCF → `(R, p, h, limits)` npz
  transform cache with source-file signature; runtime consumers never parse
  robot description files.
- **Fallback solver** (`fallback` extra) — MINK-based differential IK with G1
  and RBY1 Model-M presets. The RBY1 preset carries the paper baseline's
  upper-body/base frame mapping and optional wrist/palm, torso, elbow, and head
  tasks, without embedding protected SEW or XHand analytic kernels.

Select it explicitly for baseline validation even when `geo_kin` is installed:

```python
session = resolve_session(
    "rby1", backend="mink", model_xml="path/to/model_v1.3_xhand_act.xml"
)
```

> The SEW geometric retargeting method implemented by the `geo_kin` wheel is
> patented and licensed separately. This repository grants no patent license.

MIT licensed.
