"""Command-line interface for fitting calibrators and scoring held-out labels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .fusion_lr import DEFAULT_RHO_GRID, evaluate, fit_calibrator, predict_calibrator
from .paired_bootstrap import paired_group_bootstrap

METHODS = ("fusion_lr", "guard_lr", "qc_lr")


def _require_distinct_paths(inputs: list[Path], outputs: list[Path]) -> None:
    input_paths = {path.resolve() for path in inputs}
    output_paths = [path.resolve() for path in outputs]
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("output paths must be distinct")
    if input_paths.intersection(output_paths):
        raise ValueError("an output path would overwrite an input file")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL value must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no records found")
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def _parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(part.strip() for part in value.split(",") if part.strip())
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise argparse.ArgumentTypeError("choose unique methods from: " + ", ".join(METHODS))
    return methods


def _parse_pairs(value: str) -> tuple[tuple[str, str], ...]:
    pairs = []
    for item in value.split(","):
        parts = item.strip().split(":")
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise argparse.ArgumentTypeError("pairs must look like fusion_lr:guard_lr,fusion_lr:qc_lr")
        pairs.append((parts[0].strip(), parts[1].strip()))
    if not pairs or len(set(pairs)) != len(pairs):
        raise argparse.ArgumentTypeError("provide at least one unique method pair")
    return tuple(pairs)


def _predict(args: argparse.Namespace) -> None:
    _require_distinct_paths(
        [args.calibration, args.test_scores], [args.predictions, args.calibrators_out]
    )
    calibration_rows = _read_jsonl(args.calibration)
    test_score_rows = _read_jsonl(args.test_scores)
    all_predictions: list[dict[str, Any]] = []
    calibrators: dict[str, Any] = {
        "schema": "calibrationguard_fitted_calibrators_v1",
        "n_splits": args.n_splits,
        "random_state": args.random_state,
        "rho_grid": list(args.rho_grid),
        "calibrators": {},
    }
    for method in args.methods:
        calibrator = fit_calibrator(
            calibration_rows,
            method=method,
            rho_grid=args.rho_grid,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        calibrators["calibrators"][method] = calibrator
        all_predictions.extend(predict_calibrator(calibrator, test_score_rows))

    _write_jsonl(args.predictions, all_predictions)
    _write_json(args.calibrators_out, calibrators)
    print(f"Wrote {len(all_predictions)} predictions for {len(args.methods)} method(s) to {args.predictions}")
    print(f"Wrote fitted calibration parameters to {args.calibrators_out}")


def _evaluate(args: argparse.Namespace) -> None:
    _require_distinct_paths([args.predictions, args.test_labels], [args.output])
    prediction_rows = _read_jsonl(args.predictions)
    label_rows = _read_jsonl(args.test_labels)

    labels: dict[str, int] = {}
    for row in label_rows:
        sample_id = row.get("sample_id")
        label = row.get("label")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("test-label records require a non-empty string sample_id")
        if sample_id in labels:
            raise ValueError(f"duplicate test label for sample_id={sample_id!r}")
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError(f"test label for sample_id={sample_id!r} must be 0 or 1")
        labels[sample_id] = int(label)

    by_method: dict[str, dict[str, float]] = {}
    for row in prediction_rows:
        method = row.get("method")
        if method not in METHODS:
            raise ValueError(f"unknown method in predictions: {method!r}")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in labels:
            raise ValueError(f"prediction has no matching test label: {sample_id!r}")
        method_predictions = by_method.setdefault(method, {})
        if sample_id in method_predictions:
            raise ValueError(f"duplicate prediction for method={method!r}, sample_id={sample_id!r}")
        method_predictions[sample_id] = row.get("probability")

    if not by_method:
        raise ValueError("prediction file contains no methods")
    if args.expected_methods is not None and set(by_method) != set(args.expected_methods):
        raise ValueError(
            f"prediction methods {sorted(by_method)} do not match expected methods "
            f"{sorted(args.expected_methods)}"
        )

    results: dict[str, Any] = {
        "schema": "calibrationguard_evaluation_v1",
        "ece_bins": args.bins,
        "methods": {},
    }
    for method, method_predictions in by_method.items():
        if set(method_predictions) != set(labels):
            missing = len(set(labels) - set(method_predictions))
            extra = len(set(method_predictions) - set(labels))
            raise ValueError(f"{method}: test ID mismatch (missing={missing}, extra={extra})")
        ordered_ids = sorted(labels)
        metrics = evaluate(
            [method_predictions[sample_id] for sample_id in ordered_ids],
            [labels[sample_id] for sample_id in ordered_ids],
            bins=args.bins,
        )
        results["methods"][method] = metrics

    _write_json(args.output, results)
    print(f"Wrote held-out metrics to {args.output}")


def _bootstrap(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    input_paths = {args.manifest.resolve()}
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"predictions_path", "test_labels_path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("manifest must include predictions_path and test_labels_path columns")
        for row in reader:
            for column in required:
                raw_value = row.get(column)
                if not isinstance(raw_value, str) or not raw_value.strip():
                    raise ValueError(f"manifest row has an empty {column}")
                raw_path = raw_value.strip()
                source_path = Path(raw_path)
                if not source_path.is_absolute():
                    source_path = args.manifest.parent / source_path
                input_paths.add(source_path.resolve())
    if output in input_paths:
        raise ValueError("bootstrap output would overwrite an input file")
    result = paired_group_bootstrap(
        args.manifest,
        list(args.pairs),
        replicates=args.replicates,
        seed=args.seed,
        bins=args.bins,
    )
    _write_json(args.output, result)
    print(f"Wrote paired group-bootstrap intervals to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="calibrationguard",
        description="Fit current score calibrators, then evaluate saved predictions against held-out labels.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser(
        "predict", help="fit calibrators on labeled calibration rows and write test predictions"
    )
    predict_parser.add_argument("--calibration", type=Path, required=True,
                                help="JSONL with sample_id, group_id, guard_probability, q_c, label")
    predict_parser.add_argument("--test-scores", type=Path, required=True,
                                help="JSONL with sample_id, group_id, guard_probability, q_c; no labels")
    predict_parser.add_argument("--predictions", type=Path, required=True,
                                help="output JSONL of held-out probabilities")
    predict_parser.add_argument("--calibrators-out", type=Path, required=True,
                                help="output JSON with fitted parameters and CV selection details")
    predict_parser.add_argument("--methods", type=_parse_methods, default=METHODS,
                                help="comma-separated subset of: " + ", ".join(METHODS))
    predict_parser.add_argument("--n-splits", type=int, default=5)
    predict_parser.add_argument("--random-state", type=int, default=20260930)
    predict_parser.add_argument("--rho-grid", type=float, nargs="+", default=list(DEFAULT_RHO_GRID))
    predict_parser.set_defaults(handler=_predict)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="join saved predictions to a separate test-label JSONL and compute metrics"
    )
    evaluate_parser.add_argument("--predictions", type=Path, required=True,
                                 help="JSONL produced by the predict command")
    evaluate_parser.add_argument("--test-labels", type=Path, required=True,
                                 help="JSONL with sample_id and binary label only")
    evaluate_parser.add_argument("--output", type=Path, required=True,
                                 help="output JSON with ECE, Brier, and NLL by method")
    evaluate_parser.add_argument("--bins", type=int, default=15)
    evaluate_parser.add_argument("--expected-methods", type=_parse_methods, default=None,
                                 help="optional expected method list, e.g. fusion_lr,guard_lr,qc_lr")
    evaluate_parser.set_defaults(handler=_evaluate)

    bootstrap_parser = subparsers.add_parser(
        "bootstrap", help="compute paired group-bootstrap intervals from saved test probabilities"
    )
    bootstrap_parser.add_argument("--manifest", type=Path, required=True,
                                  help="CSV mapping each source/guard to predictions and test labels")
    bootstrap_parser.add_argument("--pairs", type=_parse_pairs, required=True,
                                  help="contrasts as methodA:methodB, where delta is A minus B")
    bootstrap_parser.add_argument("--output", type=Path, required=True,
                                  help="output JSON with source-level and macro 95% intervals")
    bootstrap_parser.add_argument("--replicates", type=int, default=10_000)
    bootstrap_parser.add_argument("--seed", type=int, required=True)
    bootstrap_parser.add_argument("--bins", type=int, default=15)
    bootstrap_parser.set_defaults(handler=_bootstrap)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
