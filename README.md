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
- **Fallback solver** (`fallback` extra) — mink-based differential IK so every
  public robot repo runs end-to-end without a license.

> The SEW geometric retargeting method implemented by the `geo_kin` wheel is
> patented and licensed separately. This repository grants no patent license.

MIT licensed.
