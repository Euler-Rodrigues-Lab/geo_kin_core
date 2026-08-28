"""Device-neutral RetargetFrame stream: save, load, and replay.

A *frame stream* is a recorded sequence of :class:`~geo_kin_core.types.RetargetFrame`
stored as a single ``.npz``. It is the neutral interchange format between input
devices and solvers:

* **Sample data** — ship a short recording with a robot repo so its offline demo
  runs with no headset, no hardware, and no device dependencies.
* **Data template** — the schema below is exactly what a new device adapter has
  to produce; transcode one recording and you can diff a new device against it.
* **Regression fixtures** — replaying a stream is deterministic.

Schema ``geo_kin_core.frames/1`` (all arrays float64 unless noted, ``n`` = frames;
every optional field has a parallel ``*_present`` uint8 mask):

===========================  ==============  ==========================================
key                          shape           meaning
===========================  ==============  ==========================================
``meta/schema``              scalar str      ``"geo_kin_core.frames/1"``
``meta/fps``                 scalar          nominal capture rate
``meta/source``              scalar str      provenance of the recording
``meta/notes``               scalar str      free text
``meta/finger_names``        (5,) str        thumb, index, middle, ring, pinky
``meta/finger_joints``       (3,) str        mcp, pip, tip
``meta/finger_keys``         (5, 3) str      exact per-joint key names captured
``{side}_sew``               (n, 18)         flat18 ``[S, E, W, R_world_wrist.flat]``
``R_world_upper_body``       (n, 3, 3)       upper-body (shoulder-centre) rotation
``p_world_upper_body``       (n, 3)          upper-body origin in world
``head_rotation``            (n, 3, 3)       head rotation in upper-body frame
``R_lower_upper``            (n, 3, 3)       lower->upper body rotation (torso solve)
``{side}_fingers``           (n, 5, 3, 3)    [finger, joint, xyz], body frame
``{side}_hka``               (n, 3, 3)       [hip, knee, ankle], lower-body frame
``{side}_ankle_rot``         (n, 3, 3)       foot rotation
``{side}_A_world``           (n, 3)          ankle in world
``{side}_hip_center_world``  (n, 3)          hip centre in world
``{side}_gripper_val``       (n,)            thumb-index distance
``skeleton/positions``       (n, N, 3)       raw device bone positions (world)
``meta/skeleton_names``      (N,) str        bone names
``meta/skeleton_parents``    (N,) int        parent index per bone (-1 = root)
``extras/<key>``             (n, 3)          pass-through vec3 extras
===========================  ==============  ==========================================

Fingers/legs are rebuilt as the same nested dicts the live device produces, so a
solver cannot tell a replayed frame from a live one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .types import RetargetFrame, SEWPose

SCHEMA = "geo_kin_core.frames/1"
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_JOINTS = ("mcp", "pip", "tip")
#: Per-finger joint key names as the devices emit them. The naming is irregular
#: (``thumb_mcp`` but ``index_finger_mcp``) and the solvers look these up by
#: exact string, so a stream stores the keys it captured (``meta/finger_keys``)
#: rather than reconstructing them from a pattern; this is only the fallback for
#: streams written before that field existed.
DEFAULT_FINGER_KEYS = (
    ("thumb_mcp", "thumb_pip", "thumb_tip"),
    ("index_finger_mcp", "index_finger_pip", "index_finger_tip"),
    ("middle_finger_mcp", "middle_finger_pip", "middle_finger_tip"),
    ("ring_finger_mcp", "ring_finger_pip", "ring_finger_tip"),
    ("pinky_mcp", "pinky_pip", "pinky_tip"),
)
SIDES = ("left", "right")
_MAT_FIELDS = ("R_world_upper_body", "head_rotation", "R_lower_upper")
_VEC_FIELDS = ("p_world_upper_body",)
_EXTRA_VEC3 = (
    "body_center", "ankle_to_body",
    "left_finger_tip_centroid", "right_finger_tip_centroid",
    "left_finger_mcp_centroid", "right_finger_mcp_centroid",
)


def _finger_keys(fingers: Optional[dict]) -> Optional[np.ndarray]:
    """Capture the (5, 3) joint key names actually used by a finger dict."""
    if not fingers:
        return None
    keys = np.array(DEFAULT_FINGER_KEYS, dtype=object)
    for fi, finger in enumerate(FINGER_NAMES):
        sub = fingers.get(finger)
        if not sub:
            continue
        for ji, joint in enumerate(FINGER_JOINTS):
            match = [k for k in sub if k.endswith(f"_{joint}")]
            if match:
                keys[fi, ji] = match[0]
    return keys


def _fingers_to_array(fingers: Optional[dict], keys) -> Optional[np.ndarray]:
    """Nested finger dict -> (5, 3, 3); None if unusable."""
    if not fingers:
        return None
    out = np.full((5, 3, 3), np.nan)
    for fi, finger in enumerate(FINGER_NAMES):
        sub = fingers.get(finger)
        if not sub:
            continue
        for ji in range(len(FINGER_JOINTS)):
            value = sub.get(str(keys[fi, ji]))
            if value is not None:
                out[fi, ji] = np.asarray(value, dtype=float)
    return out


def _array_to_fingers(arr: np.ndarray, keys) -> dict:
    """(5, 3, 3) -> the nested dict the device emits (NaN points dropped)."""
    fingers: dict = {}
    for fi, finger in enumerate(FINGER_NAMES):
        sub = {}
        for ji in range(len(FINGER_JOINTS)):
            point = arr[fi, ji]
            if not np.isnan(point).any():
                sub[str(keys[fi, ji])] = point.copy()
        if sub:
            fingers[finger] = sub
    return fingers


def save_frames(path, frames: Iterable[RetargetFrame], fps: float = 60.0,
                source: str = "", notes: str = "", compress: bool = True) -> Path:
    """Write a RetargetFrame sequence to `path` (schema above). Returns the path."""
    frames = list(frames)
    if not frames:
        raise ValueError("save_frames: no frames given")
    n = len(frames)
    finger_keys = None
    for frame in frames:
        for candidate in (frame.left_fingers, frame.right_fingers):
            if finger_keys is None:
                finger_keys = _finger_keys(candidate)
        if finger_keys is not None:
            break
    if finger_keys is None:
        finger_keys = np.array(DEFAULT_FINGER_KEYS, dtype=object)
    out: dict = {
        "meta/schema": np.array(SCHEMA),
        "meta/fps": np.array(float(fps)),
        "meta/n": np.array(n),
        "meta/source": np.array(str(source)),
        "meta/notes": np.array(str(notes)),
        "meta/finger_names": np.array(FINGER_NAMES),
        "meta/finger_joints": np.array(FINGER_JOINTS),
        "meta/finger_keys": np.array(finger_keys.astype(str)),
    }

    def stack(key: str, shape, getter):
        buf = np.zeros((n, *shape))
        mask = np.zeros(n, dtype=np.uint8)
        for i, frame in enumerate(frames):
            value = getter(frame)
            if value is None:
                continue
            buf[i] = np.asarray(value, dtype=float).reshape(shape)
            mask[i] = 1
        out[key] = buf
        out[f"{key}_present"] = mask

    for side in SIDES:
        stack(f"{side}_sew", (18,),
              lambda f, s=side: (lambda p: None if p is None else p.to_flat18())(
                  getattr(f, f"{s}_sew")))
        stack(f"{side}_fingers", (5, 3, 3),
              lambda f, s=side: _fingers_to_array(getattr(f, f"{s}_fingers"), finger_keys))
        stack(f"{side}_gripper_val", (),
              lambda f, s=side: getattr(f, f"{s}_gripper_val"))
        hka = lambda f, s=side: getattr(f, f"{s}_hka")  # noqa: E731
        stack(f"{side}_hka", (3, 3), lambda f, g=hka: None if not g(f) else
              np.stack([np.asarray(g(f)[k], float) for k in ("H", "K", "A")]))
        for name, key in (("ankle_rot", (3, 3)), ("A_world", (3,)),
                          ("hip_center_world", (3,))):
            stack(f"{side}_{name}", key,
                  lambda f, g=hka, nm=name: None if not g(f) else g(f).get(nm))
    for field in _MAT_FIELDS:
        stack(field, (3, 3), lambda f, k=field: getattr(f, k))
    for field in _VEC_FIELDS:
        stack(field, (3,), lambda f, k=field: getattr(f, k))
    for key in _EXTRA_VEC3:
        stack(f"extras/{key}", (3,), lambda f, k=key: (f.extras or {}).get(k))

    # Raw device skeleton (optional): one bone table for the stream, positions
    # per frame. Bone identity must not drift mid-stream, so the first frame
    # carrying a skeleton defines names/parents.
    skeleton = next((f.skeleton for f in frames if f.skeleton), None)
    if skeleton is not None:
        names = [str(x) for x in skeleton["names"]]
        out["meta/skeleton_names"] = np.array(names)
        out["meta/skeleton_parents"] = np.asarray(skeleton["parents"], dtype=np.int64)
        stack("skeleton/positions", (len(names), 3),
              lambda f: None if not f.skeleton else f.skeleton["positions"])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    (np.savez_compressed if compress else np.savez)(path, **out)
    return path


class FrameStream:
    """A loaded frame stream: indexable, time-samplable, replayable.

    ``stream[i]`` and :meth:`frame_at_time` return live-shaped
    :class:`RetargetFrame` objects, so a demo can swap a device for a recording
    without touching the solve path.
    """

    def __init__(self, path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            self._data = {k: data[k] for k in data.files}
        schema = str(self._data["meta/schema"])
        if schema != SCHEMA:
            raise ValueError(f"{self.path}: unsupported frame-stream schema {schema!r}")
        self.fps = float(self._data["meta/fps"])
        self.n = int(self._data["meta/n"])
        self.source = str(self._data["meta/source"])
        self.notes = str(self._data["meta/notes"])
        self.finger_keys = self._data.get("meta/finger_keys")
        if self.finger_keys is None:
            self.finger_keys = np.array(DEFAULT_FINGER_KEYS)
        self.skeleton_names = self._data.get("meta/skeleton_names")
        self.skeleton_parents = self._data.get("meta/skeleton_parents")

    def __len__(self) -> int:
        return self.n

    @property
    def duration(self) -> float:
        """Recording length in seconds."""
        return self.n / self.fps if self.fps else 0.0

    def _opt(self, key: str, i: int):
        mask = self._data.get(f"{key}_present")
        if mask is None or not mask[i]:
            return None
        return self._data[key][i]

    def __getitem__(self, i: int) -> RetargetFrame:
        if not 0 <= i < self.n:
            raise IndexError(i)
        kwargs = {}
        for side in SIDES:
            flat = self._opt(f"{side}_sew", i)
            kwargs[f"{side}_sew"] = None if flat is None else SEWPose.from_flat18(flat)
            fingers = self._opt(f"{side}_fingers", i)
            kwargs[f"{side}_fingers"] = (
                None if fingers is None else _array_to_fingers(fingers, self.finger_keys))
            grip = self._opt(f"{side}_gripper_val", i)
            kwargs[f"{side}_gripper_val"] = None if grip is None else float(grip)
            hka = self._opt(f"{side}_hka", i)
            if hka is None:
                kwargs[f"{side}_hka"] = None
            else:
                leg = {"H": hka[0].copy(), "K": hka[1].copy(), "A": hka[2].copy()}
                for name in ("ankle_rot", "A_world", "hip_center_world"):
                    value = self._opt(f"{side}_{name}", i)
                    if value is not None:
                        leg[name] = value.copy()
                kwargs[f"{side}_hka"] = leg
        for field in (*_MAT_FIELDS, *_VEC_FIELDS):
            value = self._opt(field, i)
            kwargs[field] = None if value is None else value.copy()
        positions = self._opt("skeleton/positions", i)
        kwargs["skeleton"] = None if positions is None or self.skeleton_names is None else {
            "positions": positions.copy(),
            "names": tuple(str(n) for n in self.skeleton_names),
            "parents": self.skeleton_parents.copy(),
        }
        extras = {}
        for key in _EXTRA_VEC3:
            value = self._opt(f"extras/{key}", i)
            if value is not None:
                extras[key] = value.copy()
        return RetargetFrame(extras=extras, **kwargs)

    def frames(self) -> List[RetargetFrame]:
        return [self[i] for i in range(self.n)]

    def frame_at_time(self, elapsed_time: float, loop: bool = True,
                      playback_speed: float = 1.0) -> Optional[RetargetFrame]:
        """Nearest frame at `elapsed_time` seconds (None past the end if not looping)."""
        if self.n == 0:
            return None
        idx = int(round(elapsed_time * playback_speed * self.fps))
        if idx >= self.n:
            if not loop:
                return None
            idx %= self.n
        return self[max(idx, 0)]


def load_frames(path) -> FrameStream:
    """Open a frame stream written by :func:`save_frames`."""
    return FrameStream(path)
