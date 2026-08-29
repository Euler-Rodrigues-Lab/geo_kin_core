"""CLI for the robot/hand spec tool.

Examples:

    python -m geo_kin_core.spec generate --kind robot \\
        --source g1_29dof.urdf --parts right_arm left_arm right_leg left_leg \\
        --out g1_29dof.npz

    python -m geo_kin_core.spec generate --kind hand \\
        --source g1_29dof_rev_1_0_with_inspire_hand_FTP.xml \\
        --side left --hand-type inspire --out inspire_left.npz

    python -m geo_kin_core.spec verify --npz g1_29dof.npz --source g1_29dof.urdf
    python -m geo_kin_core.spec show --npz g1_29dof.npz
"""

from __future__ import annotations

import argparse
import sys

from ._extract import (
    HAND_TYPES,
    ROBOT_PARTS,
    ROBOT_SEQUENCE_TABLES,
    generate_hand_spec,
    generate_robot_spec,
)
from ._io import load_spec, save_spec, verify_signature


def _cmd_generate(args) -> int:
    if args.kind == "robot":
        parts = args.parts or list(ROBOT_SEQUENCE_TABLES[args.robot])
        spec = generate_robot_spec(args.source, parts, robot=args.robot)
        # g1 npz files predate the robot tag; keep them meta-identical.
        extra_meta = None if args.robot == "g1" else {"robot": args.robot}
    else:
        if not args.side or not args.hand_type:
            print("error: --side and --hand-type are required for --kind hand",
                  file=sys.stderr)
            return 2
        spec = generate_hand_spec(args.source, args.side, args.hand_type)
        if not spec:
            print(f"error: no hand chains extracted from {args.source}", file=sys.stderr)
            return 1
        extra_meta = {"side": args.side, "hand_type": args.hand_type}

    save_spec(spec, args.out, args.source, kind=args.kind, extra_meta=extra_meta)
    print(f"wrote {args.out} ({args.kind}: {', '.join(spec.keys())})")
    return 0


def _cmd_verify(args) -> int:
    try:
        ok = verify_signature(args.npz, args.source)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"{args.npz}: signature {'OK' if ok else 'MISMATCH'} against {args.source}")
    return 0 if ok else 1


def _cmd_show(args) -> int:
    spec, meta = load_spec(args.npz, with_meta=True)
    for key in sorted(meta):
        print(f"{key}: {meta[key]}")
    for part, t in spec.items():
        print(f"{part}: R x{len(t['R'])}, p x{len(t['p'])}, h x{len(t['h'])}, "
              f"joint_names={t['joint_names']}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m geo_kin_core.spec",
        description="Generate, inspect, and verify robot/hand spec npz files.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser(
        "generate",
        help="Extract transforms from a URDF (robot) or MJCF xml (hand) into an npz.")
    gen.add_argument("--kind", choices=["robot", "hand"], required=True,
                     help="What to extract: robot parts (URDF) or hand fingers (MJCF).")
    gen.add_argument("--source", required=True,
                     help="Path to the source URDF (robot) or MJCF xml (hand).")
    gen.add_argument("--out", required=True, help="Output .npz path.")
    gen.add_argument("--robot", choices=sorted(ROBOT_SEQUENCE_TABLES),
                     default="g1",
                     help="Robot whose joint-sequence table to use (default: g1).")
    gen.add_argument("--parts", nargs="+", metavar="PART",
                     help="Robot parts to extract (default: the robot's full "
                          f"table, e.g. {ROBOT_PARTS} for g1).")
    gen.add_argument("--side", choices=["left", "right"],
                     help="Hand side (required for --kind hand).")
    gen.add_argument("--hand-type", choices=list(HAND_TYPES),
                     help="Hand type (required for --kind hand).")
    gen.set_defaults(func=_cmd_generate)

    ver = sub.add_parser("verify",
                         help="Check an npz's embedded sha256 against a source file.")
    ver.add_argument("--npz", required=True)
    ver.add_argument("--source", required=True)
    ver.set_defaults(func=_cmd_verify)

    show = sub.add_parser("show", help="Print an npz's metadata and part shapes.")
    show.add_argument("--npz", required=True)
    show.set_defaults(func=_cmd_show)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
