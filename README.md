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

## Shared checkout and licensed-wheel provisioning

Robot repositories should depend on one sibling checkout of this repository
instead of embedding `geo_kin_core` as a submodule:

```text
Euler-Rodrigues-Lab/
├── geo_kin_core/
├── g1_teleop/
├── rby1_teleop/
├── WARP_retargeting/
└── rby1_dex_sim/
```

Consumer `pyproject.toml` files select the sibling checkout and declare the
licensed product they need:

```toml
[tool.uv.sources]
geo-kin-core = { path = "../geo_kin_core", editable = true }

[tool.geo-kin]
product = "rby1-xhand"
```

Licensed files remain outside every Git checkout. By default the provisioner
stores them at:

```text
~/.local/share/geo-kin/wheels/<product>/geo_kin-*.whl
~/.config/geo-kin/licenses/<name>.toml
~/.config/geo-kin/license.toml
```

Register a supplied wheel and license once per user:

```bash
cd /path/to/Euler-Rodrigues-Lab/geo_kin_core
uv run geo-kin-provision register \
  --product rby1-xhand \
  --wheel /path/to/geo_kin-0.1.0-cp310-abi3-manylinux_2_35_x86_64.whl \
  --license /path/to/geo_kin_license.toml \
  --name rl2-rby1-xhand \
  --activate
```

Then provision any consumer environment after its normal dependency sync:

```bash
cd ../rby1_dex_sim
uv sync
uv run geo-kin-provision install
```

The install command reads `[tool.geo-kin].product`, selects the registered
wheel, unpacks it once under `~/.local/share/geo-kin/installs`, and adds a
small `geo_kin_shared.pth` link to the current project's `.venv`. It then
checks the wheel's compiled robot/hand features and constructs a licensed
session. Python virtual environments remain isolated without duplicating the
private binary, and the link remains intact across normal `uv sync` and
`uv run` operations.

Useful management commands:

```bash
uv run geo-kin-provision status
uv run geo-kin-provision activate rl2-rby1-xhand
uv run geo-kin-provision install --project ../WARP_retargeting
```

Set `GEO_KIN_WHEEL_DIR` to use a different shared wheel root. An explicit
`GEO_KIN_LICENSE` continues to override the active canonical license. Never
put a real wheel or license in this public repository.

MIT licensed.
