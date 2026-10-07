"""Paired group bootstrap for saved held-out probabilities.

Intervals are conditional on fitted calibrators. Groups are resampled within each
source, with shared resampling weights across guards and methods. Source-level
contrasts receive equal weight in the macro estimate.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def _metric_sums(probabilities: np.ndarray, labels: np.ndarray, bins: int) -> np.ndarray:
    """Per-example bin residuals followed by squared and log-loss terms."""
    if probabilities.ndim != 1 or probabilities.shape != labels.shape:
        raise ValueError("probability and label arrays must be aligned one-dimensional vectors")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary")
    bin_ids = np.minimum((probabilities * bins).astype(int), bins - 1)
    terms = np.zeros((len(labels), bins + 2), dtype=float)
    terms[np.arange(len(labels)), bin_ids] = probabilities - labels
    terms[:, bins] = (probabilities - labels) ** 2
    clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
    terms[:, bins + 1] = -labels * np.log(clipped) - (1 - labels) * np.log1p(-clipped)
    return terms


def _metrics_from_sums(sums: np.ndarray, denominator: np.ndarray | float, bins: int) -> np.ndarray:
    denom = np.asarray(denominator, dtype=float)
    ece = np.abs(sums[..., :bins]).sum(axis=-1) / denom
    brier = sums[..., bins] / denom
    nll = sums[..., bins + 1] / denom
    return np.stack((ece, brier, nll), axis=-1)


def _read_manifest(manifest_path: Path) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    by_source: dict[str, list[dict[str, str]]] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"source", "guard", "predictions_path", "test_labels_path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("manifest must have columns: source, guard, predictions_path, test_labels_path")
        for row in reader:
            if not all(isinstance(row.get(key), str) and row[key].strip() for key in required):
                raise ValueError("every manifest row must fill all required columns")
            by_source.setdefault(row["source"].strip(), []).append(
                {key: row[key].strip() for key in required}
            )
    if not by_source:
        raise ValueError("manifest has no data rows")
    for source, rows in by_source.items():
        guards = [row["guard"] for row in rows]
        if len(guards) != len(set(guards)):
            raise ValueError(f"manifest has duplicate guard rows for source={source!r}")
    return list(by_source), by_source


def _path_from_manifest(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def _read_labels(path: Path) -> tuple[list[str], np.ndarray]:
    rows = _jsonl(path)
    labels: dict[str, int] = {}
    for row in rows:
        sample_id, label = row.get("sample_id"), row.get("label")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{path}: label rows need a non-empty sample_id")
        if sample_id in labels:
            raise ValueError(f"{path}: duplicate label for sample_id={sample_id!r}")
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError(f"{path}: label for {sample_id!r} must be 0 or 1")
        labels[sample_id] = int(label)
    ids = sorted(labels)
    return ids, np.asarray([labels[sample_id] for sample_id in ids], dtype=int)


def _read_predictions(path: Path, ids: list[str]) -> tuple[list[str], dict[str, np.ndarray], list[str]]:
    rows = _jsonl(path)
    by_method: dict[str, dict[str, tuple[str, float]]] = {}
    expected_ids = set(ids)
    for row in rows:
        sample_id, group_id, method, probability = (
            row.get("sample_id"), row.get("group_id"), row.get("method"), row.get("probability")
        )
        if not isinstance(sample_id, str) or sample_id not in expected_ids:
            raise ValueError(f"{path}: unexpected sample_id={sample_id!r}")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"{path}: rows need a non-empty group_id")
        if not isinstance(method, str) or not method:
            raise ValueError(f"{path}: rows need a non-empty method")
        try:
            probability = float(probability)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid probability for sample_id={sample_id!r}") from exc
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"{path}: probability must be finite and in [0, 1]")
        method_rows = by_method.setdefault(method, {})
        if sample_id in method_rows:
            raise ValueError(f"{path}: duplicate prediction for method={method!r}, sample_id={sample_id!r}")
        method_rows[sample_id] = (group_id, probability)

    if not by_method:
        raise ValueError(f"{path}: no method predictions found")
    groups_by_method: dict[str, list[str]] = {}
    probabilities: dict[str, np.ndarray] = {}
    for method, method_rows in by_method.items():
        if set(method_rows) != expected_ids:
            raise ValueError(f"{path}: method {method!r} does not cover the label IDs exactly")
        groups_by_method[method] = [method_rows[sample_id][0] for sample_id in ids]
        probabilities[method] = np.asarray([method_rows[sample_id][1] for sample_id in ids], dtype=float)
    reference_groups = groups_by_method[next(iter(groups_by_method))]
    if any(groups != reference_groups for groups in groups_by_method.values()):
        raise ValueError(f"{path}: group assignments differ across methods")
    return reference_groups, probabilities, list(by_method)


def paired_group_bootstrap(
    manifest_path: Path,
    pairs: list[tuple[str, str]],
    *,
    replicates: int = 10_000,
    seed: int,
    bins: int = 15,
) -> dict[str, Any]:
    """Compute paired group-bootstrap intervals from saved predictions.

    The returned contrast is method A minus method B. The macro interval gives
    equal weight to sources and averages contrasts across guards within each source.
    """
    if replicates < 1 or bins < 1 or not pairs:
        raise ValueError("replicates, bins, and at least one method pair are required")
    if any(a == b or not a or not b for a, b in pairs):
        raise ValueError("each pair must name two different non-empty methods")

    sources, manifest = _read_manifest(manifest_path)
    rng = np.random.default_rng(seed)
    metrics = ("ece15", "brier", "nll")
    macro_draws = np.zeros((replicates, len(pairs), len(metrics)), dtype=float)
    source_results: dict[str, Any] = {}
    source_order: list[str] = []

    for source in sources:
        source_order.append(source)
        rows = manifest[source]
        label_paths = {_path_from_manifest(manifest_path, row["test_labels_path"]).resolve() for row in rows}
        if len(label_paths) != 1:
            raise ValueError(f"all guards for source={source!r} must use the same test-label file")
        label_path = next(iter(label_paths))
        ids, labels = _read_labels(label_path)
        guard_names = [row["guard"] for row in rows]
        group_reference: list[str] | None = None
        per_guard: list[dict[str, np.ndarray]] = []
        methods: set[str] | None = None
        for row in rows:
            prediction_path = _path_from_manifest(manifest_path, row["predictions_path"])
            groups, probabilities, found_methods = _read_predictions(prediction_path, ids)
            if group_reference is None:
                group_reference = groups
            elif groups != group_reference:
                raise ValueError(f"group assignments differ across guards for source={source!r}")
            if methods is None:
                methods = set(found_methods)
            elif set(found_methods) != methods:
                raise ValueError(f"method sets differ across guards for source={source!r}")
            missing = {method for pair in pairs for method in pair} - set(found_methods)
            if missing:
                raise ValueError(f"{source}/{row['guard']}: missing methods {sorted(missing)}")
            per_guard.append(probabilities)

        assert group_reference is not None and methods is not None
        unique_groups, group_index = np.unique(np.asarray(group_reference), return_inverse=True)
        n_groups = len(unique_groups)
        n_examples = len(ids)
        method_order = sorted(methods)
        guard_sums = np.zeros((len(guard_names), len(method_order), bins + 2), dtype=float)
        group_sums = np.zeros((n_groups, len(guard_names), len(method_order), bins + 2), dtype=float)
        group_sizes = np.bincount(group_index, minlength=n_groups)
        for guard_i, probabilities in enumerate(per_guard):
            for method_i, method in enumerate(method_order):
                terms = _metric_sums(probabilities[method], labels, bins)
                guard_sums[guard_i, method_i] = terms.sum(axis=0)
                np.add.at(group_sums[:, guard_i, method_i], group_index, terms)

        observed_scores = _metrics_from_sums(guard_sums, n_examples, bins)
        observed_contrasts = np.zeros((len(pairs), len(metrics)), dtype=float)
        method_indices = {name: i for i, name in enumerate(method_order)}
        for pair_i, (method_a, method_b) in enumerate(pairs):
            observed_contrasts[pair_i] = (
                observed_scores[:, method_indices[method_a], :] -
                observed_scores[:, method_indices[method_b], :]
            ).mean(axis=0)

        draws = np.zeros((replicates, len(pairs), len(metrics)), dtype=float)
        flat_group_sums = group_sums.reshape(n_groups, -1)
        for start in range(0, replicates, 250):
            stop = min(replicates, start + 250)
            weights = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=stop - start)
            denominators = weights @ group_sizes
            sampled_sums = (weights @ flat_group_sums).reshape(
                stop - start, len(guard_names), len(method_order), bins + 2
            )
            sampled_scores = _metrics_from_sums(
                sampled_sums, denominators[:, None, None], bins
            )
            for pair_i, (method_a, method_b) in enumerate(pairs):
                draws[start:stop, pair_i] = (
                    sampled_scores[:, :, method_indices[method_a], :] -
                    sampled_scores[:, :, method_indices[method_b], :]
                ).mean(axis=1)

        macro_draws += draws / len(sources)
        source_results[source] = {
            "n": n_examples,
            "groups": n_groups,
            "guards": guard_names,
            "contrasts": _summarize(draws, observed_contrasts, pairs, metrics),
        }

    macro_observed = np.mean(
        [
            np.asarray([
                [
                    source_results[source]["contrasts"][f"{a}_minus_{b}"][metric]["delta"]
                    for metric in metrics
                ]
                for a, b in pairs
            ])
            for source in sources
        ],
        axis=0,
    )
    return {
        "schema": "calibrationguard_paired_group_bootstrap_v1",
        "replicates": replicates,
        "seed": seed,
        "ece_bins": bins,
        "sources_in_resampling_order": source_order,
        "pairs_are": "method A minus method B",
        "scope": "conditional on fitted calibrators; no refitting; group resampling within source; equal-weight source macro average",
        "macro": _summarize(macro_draws, macro_observed, pairs, metrics),
        "by_source": source_results,
    }


def _summarize(
    draws: np.ndarray,
    observed: np.ndarray,
    pairs: list[tuple[str, str]],
    metrics: tuple[str, ...],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for pair_i, (method_a, method_b) in enumerate(pairs):
        contrast = f"{method_a}_minus_{method_b}"
        summary[contrast] = {}
        for metric_i, metric in enumerate(metrics):
            summary[contrast][metric] = {
                "delta": float(observed[pair_i, metric_i]),
                "ci95": np.quantile(draws[:, pair_i, metric_i], [0.025, 0.975]).tolist(),
            }
    return summary
