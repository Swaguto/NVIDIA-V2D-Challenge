import argparse
from pathlib import Path
import numpy as np
from .core import SIGNALS, fit_normalizer, score
from .experiments import make_split, matrix, pilot_decision
from .io import read_json, write_json
from .signals import extract
from .workflow import doctor, evaluate, run_recorded


def main(argv=None):
    parser = argparse.ArgumentParser(description="V2D contact reliability research tools")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("track3-split"); p.add_argument("--public-root", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("audit-public"); p.add_argument("--public-root", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("track3-matrix"); p.add_argument("--split", required=True); p.add_argument("--steps", type=int, required=True)
    p.add_argument("--phase", choices=["pilot", "holdout"], default="pilot"); p.add_argument("--floor", type=float, default=.25); p.add_argument("--output", required=True)
    p = sub.add_parser("validate-trajectory"); p.add_argument("--reference", required=True); p.add_argument("--candidate"); p.add_argument("--time-mapping"); p.add_argument("--output", required=True)
    p = sub.add_parser("ledger"); p.add_argument("--runs", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("compare-track3"); p.add_argument("--results", required=True); p.add_argument("--split", required=True)
    p.add_argument("--phase", choices=["pilot", "holdout"], default="pilot"); p.add_argument("--metric", default="add_auc"); p.add_argument("--output", required=True)
    p = sub.add_parser("doctor"); p.add_argument("--output", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--diagnostics", required=True); p.add_argument("--metadata", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--split", required=True); p.add_argument("--observations", nargs="+", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("score")
    p.add_argument("--observations", required=True); p.add_argument("--normalizer", required=True); p.add_argument("--floor", type=float, default=.25)
    p.add_argument("--variant", choices=["baseline", "reliability", "constant", "shuffled"], default="reliability")
    p.add_argument("--seed", type=int, default=17); p.add_argument("--output", required=True)
    p = sub.add_parser("split"); p.add_argument("--catalog", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("matrix"); p.add_argument("--split", required=True); p.add_argument("--steps", type=int, required=True)
    p.add_argument("--phase", choices=["pilot", "holdout"], default="pilot"); p.add_argument("--output", required=True)
    p = sub.add_parser("decide"); p.add_argument("--results", required=True); p.add_argument("--split", required=True)
    p.add_argument("--lower-is-better", action="store_true"); p.add_argument("--output", required=True)
    p = sub.add_parser("evaluate"); p.add_argument("--config", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("run"); p.add_argument("--config", required=True); p.add_argument("--output", required=True)
    p.add_argument("--cwd"); p.add_argument("--input", action="append", default=[]); p.add_argument("--artifact", action="append", default=[])
    p.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "compare-track3":
        from .track3 import summarize_matched, validate_split
        split = read_json(args.split); validate_split(split)
        result = summarize_matched(read_json(args.results), split[args.phase], args.metric)
    elif args.command == "audit-public":
        from .track3 import audit_public
        result = audit_public(args.public_root)
    elif args.command == "track3-split":
        from .track3 import discover_split
        result = discover_split(args.public_root)
    elif args.command == "track3-matrix":
        from .track3 import experiment_matrix
        result = experiment_matrix(read_json(args.split), args.steps, args.phase, args.floor)
    elif args.command == "validate-trajectory":
        from .track3 import validate_trajectory, validate_pair, validate_time_mapping
        reference = read_json(args.reference)
        _, mask = validate_trajectory(reference)
        if args.candidate:
            validate_pair(reference, read_json(args.candidate))
        if args.time_mapping:
            validate_time_mapping(read_json(args.time_mapping))
        result = {"status": "valid", "visible_poses": int(mask.sum()), "official_score": False,
                  "note": "Boundary validation only; not motion-loader or simulator validation."}
    elif args.command == "ledger":
        from .ledger import collect_runs
        result = collect_runs(args.runs)
    elif args.command == "doctor":
        result = doctor()
    elif args.command == "extract":
        meta = read_json(args.metadata)
        with np.load(args.diagnostics, allow_pickle=False) as arrays:
            values, valid = extract(**{name: arrays[name] for name in (
                "visibility_mask", "observed_masks", "rendered_masks", "observed_keypoints",
                "projected_keypoints", "relative_positions", "timestamps", "image_size")})
            if not np.array_equal(arrays["timestamps"], meta["timestamps"]):
                raise ValueError("diagnostic/metadata timestamps differ")
        result = {**meta, "schema_version": "observations_v1", "signal_names": list(SIGNALS),
                  "signals": values.tolist(), "valid": valid.tolist()}
        from .core import validate_observations
        validate_observations(result)
    elif args.command == "fit":
        split = read_json(args.split)
        development = split["development"]
        if split.get("schema_version") == "track3_split_v1":
            from .track3 import validate_split
            validate_split(split)
            development = [f"episode_{ep:06d}" for ep in development]
        result = fit_normalizer([read_json(p) for p in args.observations], development)
    elif args.command == "score":
        result = score(read_json(args.observations), read_json(args.normalizer), args.floor, args.variant, args.seed)
    elif args.command == "split":
        result = make_split(read_json(args.catalog))
    elif args.command == "matrix":
        result = matrix(read_json(args.split), args.steps, phase=args.phase)
    elif args.command == "decide":
        result = pilot_decision(read_json(args.results), read_json(args.split)["pilot"], not args.lower_is_better)
    elif args.command == "evaluate":
        result = evaluate(args.config, args.output)
        return 0 if result["status"] == "completed" else 1
    else:
        argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
        result = run_recorded(argv, args.output, read_json(args.config), inputs=[args.config, *args.input], artifacts=args.artifact, cwd=args.cwd)
        return 0 if result["status"] == "completed" else 1
    write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
