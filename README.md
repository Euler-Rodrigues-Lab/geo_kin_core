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

## Licensed-wheel provisioning

Robot repositories pin this public package as an
`external/geo_kin_core` submodule. This keeps a fresh clone self-contained and
reproducible while the large licensed binary and private license remain shared
per user:

```text
robot_repository/
└── external/
    └── geo_kin_core/  # pinned public source only
```

Consumer `pyproject.toml` files select the submodule and declare the
licensed product they need:

```toml
[tool.uv.sources]
geo-kin-core = { path = "external/geo_kin_core", editable = true }

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

Cloning with submodules and running `uv sync` installs the public package and
the `geo-kin-provision` command into that project's environment:

```bash
git clone --recurse-submodules https://github.com/Euler-Rodrigues-Lab/rby1_teleop.git
cd rby1_teleop
uv sync
```

For an existing checkout, initialize the newly added submodule before syncing:

```bash
git pull
git submodule update --init --recursive
uv sync
```

Register a supplied wheel and license once per user. This command can be run
from any provisioned consumer repository:

```bash
uv run geo-kin-provision register \
  --product rby1-xhand \
  --wheel /path/to/geo_kin-0.1.0-cp310-abi3-manylinux_2_35_x86_64.whl \
  --license /path/to/geo_kin_license.toml \
  --name rl2-rby1-xhand \
  --activate
```

Then provision any consumer environment after its normal dependency sync:

```bash
# From the consumer repository after `uv sync`:
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
uv run geo-kin-provision install --project /path/to/WARP_retargeting
```

Each repository has its own pinned copy of the small public package and command,
but all of them use the same central wheel extraction and license. Set
`GEO_KIN_WHEEL_DIR` to use a different shared wheel root. An explicit
`GEO_KIN_LICENSE` continues to override the active canonical license. Never put
a real wheel or license in this public repository or one of its consumers.

MIT licensed.
