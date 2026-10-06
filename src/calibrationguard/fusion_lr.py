"""Source-specific regularized logistic calibration using guard and auxiliary scores.

This module implements only the fixed FusionLR method and its current single-signal
controls. It deliberately excludes exploratory gates, beta-additive variants, legacy
TS-plus-auxiliary methods, and CoT methods.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedGroupKFold

EPS = 1e-7
DEFAULT_RHO_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def _vectors(rows, *, labels_required: bool):
    if not rows:
        raise ValueError("input rows must be non-empty")
    ids = [r["sample_id"] for r in rows]
    groups = [r["group_id"] for r in rows]
    if any(not isinstance(v, str) or not v for v in ids + groups) or len(ids) != len(set(ids)):
        raise ValueError("sample_id and group_id must be non-empty strings; sample_id must be unique")
    pg = np.asarray([r["guard_probability"] for r in rows], dtype=float)
    qc = np.asarray([r["q_c"] for r in rows], dtype=float)
    if pg.ndim != 1 or qc.ndim != 1 or not np.isfinite(pg).all() or not np.isfinite(qc).all():
        raise ValueError("scores must be finite one-dimensional values")
    if np.any((pg < 0) | (pg > 1)) or np.any((qc < 0) | (qc > 1)):
        raise ValueError("guard_probability and q_c must be in [0, 1]")
    y = None
    if labels_required:
        y = np.asarray([r["label"] for r in rows])
        if y.shape != pg.shape or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
            raise ValueError("calibration labels must be binary and contain both classes")
        y = y.astype(int)
    elif any("label" in r for r in rows):
        raise ValueError("test score inputs must not contain labels")
    return np.asarray(ids), np.asarray(groups), pg, qc, y


def _features(pg: np.ndarray, qc: np.ndarray, method: str) -> np.ndarray:
    z = logit(np.clip(pg, EPS, 1 - EPS))
    v = 2 * qc - 1
    if method == "fusion_lr":
        return np.column_stack((z, v))
    if method == "guard_lr":
        return z[:, None]
    if method == "qc_lr":
        return v[:, None]
    raise ValueError("method must be fusion_lr, guard_lr, or qc_lr")


def _fit_logistic(x: np.ndarray, y: np.ndarray, rho: float) -> dict:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    constant = scale < 1e-8
    scale = np.where(constant, 1.0, scale)
    xs = (x - mean) / scale
    xs[:, constant] = 0.0
    design = np.column_stack((np.ones(len(y)), xs))

    def objective(theta):
        z = design @ theta
        loss = np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * rho * np.sum(theta[1:] ** 2)
        grad = design.T @ (expit(z) - y) / len(y)
        grad[1:] += rho * theta[1:]
        return float(loss), grad

    result = minimize(objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
    if not result.success or not np.isfinite(result.fun) or np.max(np.abs(objective(result.x)[1])) > 1e-5:
        raise RuntimeError(f"logistic fit failed: {result.message}")
    return {"rho": float(rho), "mean": mean.tolist(), "scale": scale.tolist(),
            "constant": constant.tolist(), "theta": result.x.tolist()}


def _predict(model: dict, x: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["mean"])
    scale = np.asarray(model["scale"])
    constant = np.asarray(model["constant"], dtype=bool)
    xs = (x - mean) / scale
    xs[:, constant] = 0.0
    return expit(model["theta"][0] + xs @ np.asarray(model["theta"][1:]))


def nll(probability: Sequence[float], labels: Sequence[int]) -> float:
    p = np.clip(np.asarray(probability, dtype=float), 1e-15, 1 - 1e-15)
    y = np.asarray(labels, dtype=int)
    if p.shape != y.shape or not np.isfinite(p).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("invalid probability/label vectors")
    return float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p)))


def fit_calibrator(calibration_rows: list[dict], method: str = "fusion_lr",
                   rho_grid: Iterable[float] = DEFAULT_RHO_GRID,
                   n_splits: int = 5, random_state: int = 20260930) -> dict:
    """Select ridge strength by pooled grouped OOF NLL, then refit on calibration rows.

    Calibration rows require sample_id, group_id, guard_probability, q_c, and binary label.
    All records from one group stay in the same validation fold.
    """
    ids, groups, pg, qc, y = _vectors(calibration_rows, labels_required=True)
    grid = tuple(sorted({float(r) for r in rho_grid}))
    if not grid or any(r <= 0 for r in grid):
        raise ValueError("rho_grid must contain positive values")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = {rho: np.full(len(y), np.nan) for rho in grid}
    for train, valid in splitter.split(pg, y, groups):
        if set(groups[train]) & set(groups[valid]):
            raise RuntimeError("group leakage across CV folds")
        if len(np.unique(y[train])) != 2 or len(np.unique(y[valid])) != 2:
            raise ValueError("each grouped CV fold must contain both labels")
        x_train = _features(pg[train], qc[train], method)
        x_valid = _features(pg[valid], qc[valid], method)
        for rho in grid:
            oof[rho][valid] = _predict(_fit_logistic(x_train, y[train], rho), x_valid)
    losses = {rho: nll(oof[rho], y) for rho in grid}
    best = min(losses.values())
    chosen = max(rho for rho, loss in losses.items() if loss <= best + 1e-6)
    model = _fit_logistic(_features(pg, qc, method), y, chosen)
    return {"schema": "calibrationguard_fusionlr_v1", "method": method,
            "rho_grid": list(grid), "oof_nll": {str(k): v for k, v in losses.items()},
            "selected_rho": chosen, "fit_ids": ids.tolist(),
            "fit_groups": sorted(set(groups.tolist())), "model": model}


def predict_calibrator(calibrator: dict, test_rows: list[dict]) -> list[dict]:
    ids, groups, pg, qc, _ = _vectors(test_rows, labels_required=False)
    if set(ids) & set(calibrator["fit_ids"]) or set(groups) & set(calibrator["fit_groups"]):
        raise ValueError("calibration and test IDs/groups overlap")
    p = _predict(calibrator["model"], _features(pg, qc, calibrator["method"]))
    return [{"sample_id": str(ids[i]), "group_id": str(groups[i]),
             "method": calibrator["method"], "probability": float(p[i])}
            for i in range(len(ids))]


def evaluate(probability: Sequence[float], labels: Sequence[int], bins: int = 15) -> dict:
    p = np.asarray(probability, dtype=float)
    y = np.asarray(labels, dtype=int)
    if p.ndim != 1 or p.shape != y.shape or not len(p) or not np.isfinite(p).all():
        raise ValueError("invalid probability/label vectors")
    if np.any((p < 0) | (p > 1)) or not np.isin(y, [0, 1]).all() or bins < 1:
        raise ValueError("probabilities must be in [0,1], labels binary, bins positive")
    ix = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = ix == b
        if mask.any():
            ece += mask.mean() * abs(float(p[mask].mean()) - float(y[mask].mean()))
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    return {"n": len(p), "ece15": float(ece), "brier": float(np.mean((p-y)**2)),
            "nll": nll(p, y)}
