"""Independent stages: validate, hands, depth, align, contacts, report."""

from __future__ import annotations

import argparse
import json

from .manifest import load
from .pipeline import DEPENDENCIES, compare_reports, execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    for name in DEPENDENCIES:
        sub = commands.add_parser(name)
        sub.add_argument("--manifest", required=True)
        sub.add_argument(
            "--output",
            required=True,
            help="Fresh run directory; stages share this directory",
        )
        sub.add_argument(
            "--limit",
            type=int,
            help="First N frames; use the same value at every stage",
        )
    compare = commands.add_parser("compare")
    compare.add_argument("first")
    compare.add_argument("second")
    args = parser.parse_args()
    try:
        if args.stage == "compare":
            print(json.dumps(compare_reports(args.first, args.second), indent=2))
        else:
            print(execute(load(args.manifest, args.limit), args.output, args.stage))
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(2, f"issue6: {exc}\n")


if __name__ == "__main__":
    main()
