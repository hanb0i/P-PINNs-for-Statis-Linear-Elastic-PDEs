#!/usr/bin/env python3
"""
Reproducible random verification for the three-layer PINN.

This script evaluates the three-layer PINN against FEM on random parameter
combinations sampled from the configured E and layer-thickness bounds. It
reports top-surface z-displacement MAE% for every case and summarizes mean /
worst / best.

Usage:
    cd /path/to/repo
    python scripts/verify_three_layer_grid_sweep.py

Environment variables (all optional):
    PINN_MODEL_PATH      – path to the three-layer PINN checkpoint
                           (default: pinn-workflow/pinn_model.pth)
    PINN_DEVICE          – torch device string, e.g. "cpu", "cuda", "mps"
                           (default: auto-detect)
    PINN_EVAL_E_VALUES   – comma-separated E values used as random sampling bounds
                           (default: from three-layer-workflow/pinn_config.py E_RANGE)
    PINN_EVAL_T1_VALUES  – comma-separated t₁ values used as random sampling bounds
                           (default: from three-layer-workflow/pinn_config.py DATA_T1_VALUES)
    PINN_EVAL_T2_VALUES  – comma-separated t₂ values used as random sampling bounds
                           (default: from three-layer-workflow/pinn_config.py DATA_T2_VALUES)
    PINN_EVAL_T3_VALUES  – comma-separated t₃ values used as random sampling bounds
                           (default: from three-layer-workflow/pinn_config.py DATA_T3_VALUES)
    PINN_EVAL_N_CASES    – number of random cases to evaluate
                           (default: old grid size from configured value counts)
    PINN_EVAL_SEED       – random seed for reproducible verification cases
                           (default: 20260428)
    PINN_CALIBRATION_JSON – optional compliance calibration JSON. If present,
                           feature_coefficients are applied as a transparent
                           multiplicative correction.
    PINN_EVAL_NE_X       – FEM mesh elements in x (default: 16)
    PINN_EVAL_NE_Y       – FEM mesh elements in y (default: 16)
    PINN_EVAL_NE_Z       – FEM mesh elements in z (default: 8)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
# Code lives in three-layer-workflow/; the best checkpoint is in pinn-workflow/
CODE_DIR = REPO_ROOT / "three-layer-workflow"
PINN_WORKFLOW_DIR = REPO_ROOT / "pinn-workflow"
if not PINN_WORKFLOW_DIR.exists():
    PINN_WORKFLOW_DIR = CODE_DIR
FEA_DIR = REPO_ROOT / "fea-workflow" / "solver"

for _path in (CODE_DIR, FEA_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fem_solver  # noqa: E402
import model  # noqa: E402
import pinn_config as config  # noqa: E402

_CALIBRATION_CACHE: dict[str, dict | None] = {}

# ---------------------------------------------------------------------------
# Helpers (mirrored from compare_three_layer_pinn_fem.py to keep standalone)
# ---------------------------------------------------------------------------


def _select_device() -> torch.device:
    requested = os.getenv("PINN_DEVICE")
    if requested:
        return torch.device(requested)
    if os.getenv("PINN_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_pinn(device: torch.device) -> torch.nn.Module:
    model_path = Path(os.getenv("PINN_MODEL_PATH") or PINN_WORKFLOW_DIR / "pinn_model.pth")
    if not model_path.exists():
        raise FileNotFoundError(f"PINN checkpoint not found: {model_path}")
    pinn = model.MultiLayerPINN().to(device)
    sd = torch.load(str(model_path), map_location=device, weights_only=True)
    sd = model.adapt_legacy_state_dict(sd, pinn.state_dict())
    pinn.load_state_dict(sd, strict=False)
    pinn.eval()
    print(f"Loaded PINN checkpoint: {model_path}")
    return pinn


def _ref_params() -> tuple[float, float, float]:
    return (
        float(getattr(config, "RESTITUTION_REF", 0.5)),
        float(getattr(config, "FRICTION_REF", 0.3)),
        float(getattr(config, "IMPACT_VELOCITY_REF", 1.0)),
    )


def _load_calibration() -> dict | None:
    path = os.getenv("PINN_CALIBRATION_JSON")
    if not path:
        return None
    if path not in _CALIBRATION_CACHE:
        cal_path = Path(path)
        if not cal_path.is_absolute():
            cal_path = REPO_ROOT / cal_path
        _CALIBRATION_CACHE[path] = json.loads(cal_path.read_text()) if cal_path.exists() else None
    return _CALIBRATION_CACHE[path]


def _compliance_params() -> tuple[float, float, float]:
    scale = float(getattr(config, "DISPLACEMENT_COMPLIANCE_SCALE", 1.0))
    e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
    alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    cal = _load_calibration()
    tuned = cal.get("tuned_params") if cal else None
    if tuned:
        scale = float(tuned.get("PINN_DISPLACEMENT_COMPLIANCE_SCALE", scale))
        e_pow = float(tuned.get("PINN_E_COMPLIANCE_POWER", e_pow))
        alpha = float(tuned.get("PINN_THICKNESS_COMPLIANCE_ALPHA", alpha))
    return scale, e_pow, alpha


def _calibration_features(pts: np.ndarray) -> np.ndarray:
    x = pts[:, 0:1]
    y = pts[:, 1:2]
    z = pts[:, 2:3]
    e1 = pts[:, 3:4]
    t1 = pts[:, 4:5]
    e2 = pts[:, 5:6]
    t2 = pts[:, 6:7]
    e3 = pts[:, 7:8]
    t3 = pts[:, 8:9]
    t_total = np.clip(t1 + t2 + t3, 1e-8, None)
    e_mean = np.clip((e1 + e2 + e3) / 3.0, 1e-8, None)
    z_hat = z / t_total
    e_ref = np.sqrt(float(config.E_RANGE[0]) * float(config.E_RANGE[1]))
    h_ref = float(getattr(config, "H", 0.1))
    load_x = ((x >= config.LOAD_PATCH_X[0]) & (x <= config.LOAD_PATCH_X[1])).astype(float)
    load_y = ((y >= config.LOAD_PATCH_Y[0]) & (y <= config.LOAD_PATCH_Y[1])).astype(float)
    load_patch = load_x * load_y
    xc = x - 0.5 * float(config.Lx)
    yc = y - 0.5 * float(config.Ly)
    feats = np.concatenate(
        [
            np.ones_like(x),
            np.log(e_mean / e_ref),
            np.log(np.clip(e1, 1e-8, None) / e_ref),
            np.log(np.clip(e2, 1e-8, None) / e_ref),
            np.log(np.clip(e3, 1e-8, None) / e_ref),
            np.log(h_ref / t_total),
            t1 / t_total,
            t2 / t_total,
            t3 / t_total,
            z_hat,
            z_hat**2,
            load_patch,
            xc,
            yc,
            xc**2,
            yc**2,
            xc * yc,
            load_patch * xc,
            load_patch * yc,
            load_patch * xc**2,
            load_patch * yc**2,
        ],
        axis=1,
    )
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def _calibration_multiplier(pts: np.ndarray) -> np.ndarray | None:
    cal = _load_calibration()
    if not cal:
        return None
    coeffs = cal.get("feature_coefficients")
    if coeffs is None:
        return None
    coeffs_arr = np.asarray(coeffs, dtype=float).reshape(-1, 1)
    feats = _calibration_features(pts)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        log_multiplier = np.nan_to_num(feats @ coeffs_arr, nan=0.0, posinf=0.0, neginf=0.0)
    clip = float(cal.get("log_multiplier_clip", 1.5))
    return np.exp(np.clip(log_multiplier, -clip, clip))


def _u_from_v(v: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply compliance scaling to raw network output v to get displacement u."""
    e_scale = (pts[:, 3:4] + pts[:, 5:6] + pts[:, 7:8]) / 3.0
    t_scale = pts[:, 4:5] + pts[:, 6:7] + pts[:, 8:9]
    scale, e_pow, alpha = _compliance_params()
    h_ref = float(getattr(config, "H", 1.0))
    u = scale * v / (e_scale**e_pow) * (h_ref / np.clip(t_scale, 1e-8, None)) ** alpha
    multiplier = _calibration_multiplier(pts)
    if multiplier is not None:
        u = u * multiplier
    return u


def _predict_pinn(
    pinn: torch.nn.Module,
    device: torch.device,
    x_flat: np.ndarray,
    y_flat: np.ndarray,
    z_flat: np.ndarray,
    e1: float,
    e2: float,
    e3: float,
    t1: float,
    t2: float,
    t3: float,
) -> np.ndarray:
    r_ref, mu_ref, v0_ref = _ref_params()
    pts = np.stack(
        [
            x_flat,
            y_flat,
            z_flat,
            np.full_like(x_flat, float(e1)),
            np.full_like(x_flat, float(t1)),
            np.full_like(x_flat, float(e2)),
            np.full_like(x_flat, float(t2)),
            np.full_like(x_flat, float(e3)),
            np.full_like(x_flat, float(t3)),
            np.full_like(x_flat, r_ref),
            np.full_like(x_flat, mu_ref),
            np.full_like(x_flat, v0_ref),
        ],
        axis=1,
    )
    with torch.no_grad():
        v = pinn(torch.tensor(pts, dtype=torch.float32, device=device)).cpu().numpy()
    return _u_from_v(v, pts)


def _run_fem(e1: float, e2: float, e3: float, t1: float, t2: float, t3: float, ne_x: int, ne_y: int, ne_z: int):
    thickness = float(t1) + float(t2) + float(t3)
    cfg = {
        "geometry": {
            "Lx": float(config.Lx),
            "Ly": float(config.Ly),
            "H": thickness,
            "ne_x": int(ne_x),
            "ne_y": int(ne_y),
            "ne_z": int(ne_z),
        },
        "material": {
            "E_layers": [float(e1), float(e2), float(e3)],
            "t_layers": [float(t1), float(t2), float(t3)],
            "nu": float(config.nu_vals[0]),
        },
        "load_patch": {
            "pressure": float(config.p0),
            "x_start": float(config.LOAD_PATCH_X[0]) / float(config.Lx),
            "x_end": float(config.LOAD_PATCH_X[1]) / float(config.Lx),
            "y_start": float(config.LOAD_PATCH_Y[0]) / float(config.Ly),
            "y_end": float(config.LOAD_PATCH_Y[1]) / float(config.Ly),
        },
        "boundary": dict(getattr(config, "EDGE_BCS", {})),
    }
    return fem_solver.solve_three_layer_fem(cfg)


def _mae_pct(pred: np.ndarray, ref: np.ndarray) -> float:
    mae = float(np.mean(np.abs(pred - ref)))
    denom = float(np.max(np.abs(ref)))
    return 100.0 * mae / denom if denom > 0 else 0.0


def _relative_l2_pct(pred: np.ndarray, ref: np.ndarray) -> float:
    denom = float(np.sqrt(np.sum(np.asarray(ref, dtype=float) ** 2)))
    return 100.0 * float(np.sqrt(np.sum((np.asarray(pred) - np.asarray(ref)) ** 2))) / denom if denom > 0 else 0.0


def _bias_pct(pred: np.ndarray, ref: np.ndarray) -> float:
    denom = float(np.max(np.abs(ref)))
    return 100.0 * float(np.mean(pred - ref)) / denom if denom > 0 else 0.0


def _peak_uz_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    peak_ref = float(np.min(ref))
    peak_pred = float(np.min(pred))
    denom = abs(peak_ref)
    return {
        "peak_fem_uz": peak_ref,
        "peak_pinn_uz": peak_pred,
        "peak_uz_error_pct": 100.0 * abs(peak_pred - peak_ref) / denom if denom > 0 else 0.0,
    }


def _float_list_env(name: str) -> list[float] | None:
    value = os.getenv(name)
    if not value:
        return None
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _bounds(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot derive bounds from an empty value list")
    return float(min(values)), float(max(values))


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def main() -> None:
    device = _select_device()
    print(f"Using device: {device}")

    # Mesh resolution
    ne_x = int(os.getenv("PINN_EVAL_NE_X", "16"))
    ne_y = int(os.getenv("PINN_EVAL_NE_Y", "16"))
    ne_z = int(os.getenv("PINN_EVAL_NE_Z", "8"))
    print(f"FEM mesh: {ne_x} × {ne_y} × {ne_z}")

    # Random parameter cases. The configured/evaluated values are used as bounds
    # instead of fixed threshold grid points.
    e_values_for_bounds = _float_list_env("PINN_EVAL_E_VALUES")
    if e_values_for_bounds is None:
        e_values_for_bounds = [float(v) for v in getattr(config, "E_RANGE", [1.0, 10.0])]

    t1_values_for_bounds = _float_list_env("PINN_EVAL_T1_VALUES")
    if t1_values_for_bounds is None:
        t1_values_for_bounds = [float(v) for v in getattr(config, "DATA_T1_VALUES", [0.02, 0.10])]

    t2_values_for_bounds = _float_list_env("PINN_EVAL_T2_VALUES")
    if t2_values_for_bounds is None:
        t2_values_for_bounds = [float(v) for v in getattr(config, "DATA_T2_VALUES", [0.02, 0.10])]

    t3_values_for_bounds = _float_list_env("PINN_EVAL_T3_VALUES")
    if t3_values_for_bounds is None:
        t3_values_for_bounds = [float(v) for v in getattr(config, "DATA_T3_VALUES", [0.02, 0.10])]

    e_bounds = _bounds(e_values_for_bounds)
    t1_bounds = _bounds(t1_values_for_bounds)
    t2_bounds = _bounds(t2_values_for_bounds)
    t3_bounds = _bounds(t3_values_for_bounds)
    default_n_cases = max(
        1,
        len(e_values_for_bounds) ** 3
        * len(t1_values_for_bounds)
        * len(t2_values_for_bounds)
        * len(t3_values_for_bounds),
    )
    n_cases = int(os.getenv("PINN_EVAL_N_CASES", str(default_n_cases)))
    seed = int(os.getenv("PINN_EVAL_SEED", "20260428"))
    rng = np.random.default_rng(seed)

    random_cases = [
        (
            float(rng.uniform(e_bounds[0], e_bounds[1])),
            float(rng.uniform(e_bounds[0], e_bounds[1])),
            float(rng.uniform(e_bounds[0], e_bounds[1])),
            float(rng.uniform(t1_bounds[0], t1_bounds[1])),
            float(rng.uniform(t2_bounds[0], t2_bounds[1])),
            float(rng.uniform(t3_bounds[0], t3_bounds[1])),
        )
        for _ in range(n_cases)
    ]

    print(f"E bounds: {e_bounds}")
    print(f"t1 bounds: {t1_bounds}")
    print(f"t2 bounds: {t2_bounds}")
    print(f"t3 bounds: {t3_bounds}")
    print(f"Random cases: {n_cases} (seed={seed})")

    pinn = _load_pinn(device)
    scale, e_pow, alpha = _compliance_params()
    calibration_json = os.getenv("PINN_CALIBRATION_JSON")
    if calibration_json:
        print(f"Calibration JSON: {calibration_json}")
    print(f"Compliance scaling: scale={scale:.6g}, E power={e_pow:.6g}, thickness alpha={alpha:.6g}")

    results = []
    for idx, (e1, e2, e3, t1, t2, t3) in enumerate(random_cases):
        x_nodes, y_nodes, _, u_fea = _run_fem(e1, e2, e3, t1, t2, t3, ne_x, ne_y, ne_z)
        x_nodes = np.asarray(x_nodes)
        y_nodes = np.asarray(y_nodes)
        u_fea = np.asarray(u_fea)

        thickness = float(t1) + float(t2) + float(t3)
        x_grid, y_grid = np.meshgrid(x_nodes, y_nodes, indexing="ij")
        u_pinn_top = _predict_pinn(
            pinn,
            device,
            x_grid.ravel(),
            y_grid.ravel(),
            np.full(x_grid.size, thickness),
            e1,
            e2,
            e3,
            t1,
            t2,
            t3,
        ).reshape(len(x_nodes), len(y_nodes), 3)

        u_z_fea_top = u_fea[:, :, -1, 2]
        u_z_pinn_top = u_pinn_top[:, :, 2]
        mae = _mae_pct(u_z_pinn_top, u_z_fea_top)
        volume_top_ref = u_fea[:, :, -1, :]
        volume_mae = _mae_pct(u_pinn_top, volume_top_ref)

        case_id = f"random_{idx:03d}"
        results.append({
            "case_id": case_id,
            "e1": float(e1),
            "e2": float(e2),
            "e3": float(e3),
            "t1": float(t1),
            "t2": float(t2),
            "t3": float(t3),
            "top_uz_mae_pct": float(mae),
            "top_uz_relative_l2_pct": float(_relative_l2_pct(u_z_pinn_top, u_z_fea_top)),
            "top_uz_bias_pct": float(_bias_pct(u_z_pinn_top, u_z_fea_top)),
            "top_surface_vector_mae_pct": float(volume_mae),
            "top_surface_vector_relative_l2_pct": float(_relative_l2_pct(u_pinn_top, volume_top_ref)),
            **_peak_uz_metrics(u_z_pinn_top, u_z_fea_top),
        })
        print(
            f"  {case_id}: E=[{e1:.6g},{e2:.6g},{e3:.6g}], "
            f"t=[{t1:.6g},{t2:.6g},{t3:.6g}], top MAE = {mae:.2f}%"
        )

    maes = [r["top_uz_mae_pct"] for r in results]
    l2s = [r["top_uz_relative_l2_pct"] for r in results]
    biases = [r["top_uz_bias_pct"] for r in results]
    vector_maes = [r["top_surface_vector_mae_pct"] for r in results]
    peak_errors = [r["peak_uz_error_pct"] for r in results]
    mean_mae = float(np.mean(maes))
    worst_mae = float(np.max(maes))
    best_mae = float(np.min(maes))
    worst_idx = int(np.argmax(maes))
    worst_case = results[worst_idx]

    summary = {
        "model": "three-layer",
        "sampling_protocol": "random_parameter_verification",
        "seed": seed,
        "calibration_json": calibration_json,
        "compliance_params": {
            "PINN_DISPLACEMENT_COMPLIANCE_SCALE": scale,
            "PINN_E_COMPLIANCE_POWER": e_pow,
            "PINN_THICKNESS_COMPLIANCE_ALPHA": alpha,
        },
        "model_path": str(os.getenv("PINN_MODEL_PATH") or PINN_WORKFLOW_DIR / "pinn_model.pth"),
        "device": str(device),
        "mesh": {"ne_x": ne_x, "ne_y": ne_y, "ne_z": ne_z},
        "e_bounds": list(e_bounds),
        "t1_bounds": list(t1_bounds),
        "t2_bounds": list(t2_bounds),
        "t3_bounds": list(t3_bounds),
        "n_cases": len(results),
        "mean_top_uz_mae_pct": mean_mae,
        "worst_top_uz_mae_pct": worst_mae,
        "best_top_uz_mae_pct": best_mae,
        "mean_top_uz_relative_l2_pct": float(np.mean(l2s)),
        "worst_top_uz_relative_l2_pct": float(np.max(l2s)),
        "mean_top_uz_bias_pct": float(np.mean(biases)),
        "mean_abs_top_uz_bias_pct": float(np.mean(np.abs(biases))),
        "mean_top_surface_vector_mae_pct": float(np.mean(vector_maes)),
        "worst_top_surface_vector_mae_pct": float(np.max(vector_maes)),
        "mean_peak_uz_error_pct": float(np.mean(peak_errors)),
        "worst_peak_uz_error_pct": float(np.max(peak_errors)),
        "worst_case": worst_case,
        "cases": results,
    }

    out_dir = REPO_ROOT / "graphs" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "three_layer_grid_sweep_verification.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 60)
    print("THREE-LAYER RANDOM VERIFICATION")
    print("=" * 60)
    print(f"  Mean top MAE: {mean_mae:.2f}%")
    print(f"  Worst top MAE: {worst_mae:.2f}%")
    print(f"  Best top MAE: {best_mae:.2f}%")
    print(f"  Mean top relative L2: {summary['mean_top_uz_relative_l2_pct']:.2f}%")
    print(f"  Mean top bias: {summary['mean_top_uz_bias_pct']:.2f}%")
    print(f"  Mean top-surface vector MAE: {summary['mean_top_surface_vector_mae_pct']:.2f}%")
    print(f"  Mean peak u_z error: {summary['mean_peak_uz_error_pct']:.2f}%")
    print(f"  Worst case: E=[{worst_case['e1']},{worst_case['e2']},{worst_case['e3']}] "
          f"t=[{worst_case['t1']},{worst_case['t2']},{worst_case['t3']}]")
    print(f"  Results saved to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
