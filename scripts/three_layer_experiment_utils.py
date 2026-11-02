"""Shared helpers for three-layer PINN/FEM experiment scripts."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PINN_DIR = REPO_ROOT / "pinn-workflow"
if not PINN_DIR.exists() or not (PINN_DIR / "model.py").exists():
    PINN_DIR = REPO_ROOT / "three-layer-workflow"
FEA_DIR = REPO_ROOT / "fea-workflow" / "solver"
GRAPHS_DATA_DIR = REPO_ROOT / "graphs" / "data"
GRAPHS_FIG_DIR = REPO_ROOT / "graphs" / "figures"

for _path in (PINN_DIR, FEA_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fem_solver  # noqa: E402
import model  # noqa: E402
import pinn_config as config  # noqa: E402

_CALIBRATION_CACHE: dict[str, dict | None] = {}


@dataclass(frozen=True)
class ThreeLayerCase:
    case_id: str
    e1: float
    e2: float
    e3: float
    t1: float
    t2: float
    t3: float

    @property
    def e(self) -> tuple[float, float, float]:
        return (self.e1, self.e2, self.e3)

    @property
    def t(self) -> tuple[float, float, float]:
        return (self.t1, self.t2, self.t3)

    @property
    def thickness(self) -> float:
        return float(self.t1 + self.t2 + self.t3)


def ensure_output_dirs() -> None:
    GRAPHS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_FIG_DIR.mkdir(parents=True, exist_ok=True)


def select_device() -> torch.device:
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


def load_pinn(device: torch.device, model_path: str | os.PathLike | None = None):
    model_path = Path(model_path or os.getenv("PINN_MODEL_PATH") or PINN_DIR / "pinn_model.pth")
    pinn = model.MultiLayerPINN().to(device)
    state = torch.load(str(model_path), map_location=device, weights_only=True)
    state = model.adapt_legacy_state_dict(state, pinn.state_dict())
    pinn.load_state_dict(state, strict=False)
    pinn.eval()
    return pinn, model_path


def u_from_v(v: np.ndarray, pts: np.ndarray) -> np.ndarray:
    e_scale = (pts[:, 3:4] + pts[:, 5:6] + pts[:, 7:8]) / 3.0
    t_scale = pts[:, 4:5] + pts[:, 6:7] + pts[:, 8:9]
    e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
    alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    scale = float(getattr(config, "DISPLACEMENT_COMPLIANCE_SCALE", 1.0))
    h_ref = float(getattr(config, "H", 1.0))
    u = scale * v / (e_scale**e_pow) * (h_ref / np.clip(t_scale, 1e-8, None)) ** alpha
    multiplier = calibration_multiplier(pts)
    if multiplier is not None:
        u = u * multiplier
    return u


def calibration_features(pts: np.ndarray) -> np.ndarray:
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


def _load_calibration() -> dict | None:
    path = os.getenv("PINN_CALIBRATION_JSON")
    if not path:
        return None
    if path not in _CALIBRATION_CACHE:
        cal_path = Path(path)
        _CALIBRATION_CACHE[path] = json.loads(cal_path.read_text()) if cal_path.exists() else None
    return _CALIBRATION_CACHE[path]


def calibration_multiplier(pts: np.ndarray) -> np.ndarray | None:
    cal = _load_calibration()
    if not cal:
        return None
    coeffs = cal.get("feature_coefficients")
    if coeffs is None:
        return None
    coeffs_arr = np.asarray(coeffs, dtype=float).reshape(-1, 1)
    feats = calibration_features(pts)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        log_multiplier = np.nan_to_num(feats @ coeffs_arr, nan=0.0, posinf=0.0, neginf=0.0)
    clip = float(cal.get("log_multiplier_clip", 1.5))
    multiplier = np.exp(np.clip(log_multiplier, -clip, clip))
    return multiplier


def make_points(x: np.ndarray, y: np.ndarray, z: np.ndarray, case: ThreeLayerCase) -> np.ndarray:
    r_ref = float(getattr(config, "RESTITUTION_REF", 0.5))
    mu_ref = float(getattr(config, "FRICTION_REF", 0.3))
    v0_ref = float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))
    return np.stack(
        [
            x,
            y,
            z,
            np.full_like(x, case.e1, dtype=float),
            np.full_like(x, case.t1, dtype=float),
            np.full_like(x, case.e2, dtype=float),
            np.full_like(x, case.t2, dtype=float),
            np.full_like(x, case.e3, dtype=float),
            np.full_like(x, case.t3, dtype=float),
            np.full_like(x, r_ref, dtype=float),
            np.full_like(x, mu_ref, dtype=float),
            np.full_like(x, v0_ref, dtype=float),
        ],
        axis=1,
    )


def predict_displacement(pinn, device: torch.device, pts: np.ndarray, batch_size: int = 32768) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, len(pts), batch_size):
            batch = torch.tensor(pts[start : start + batch_size], dtype=torch.float32, device=device)
            v = pinn(batch).detach().cpu().numpy()
            out.append(u_from_v(v, pts[start : start + batch_size]))
    return np.concatenate(out, axis=0)


def fem_cfg(case: ThreeLayerCase, ne_x: int, ne_y: int, ne_z: int) -> dict:
    return {
        "geometry": {"Lx": config.Lx, "Ly": config.Ly, "H": case.thickness, "ne_x": ne_x, "ne_y": ne_y, "ne_z": ne_z},
        "material": {"E_layers": [case.e1, case.e2, case.e3], "t_layers": [case.t1, case.t2, case.t3], "nu": config.nu_vals[0]},
        "load_patch": {
            "pressure": config.p0,
            "x_start": config.LOAD_PATCH_X[0] / config.Lx,
            "x_end": config.LOAD_PATCH_X[1] / config.Lx,
            "y_start": config.LOAD_PATCH_Y[0] / config.Ly,
            "y_end": config.LOAD_PATCH_Y[1] / config.Ly,
        },
        "boundary": dict(getattr(config, "EDGE_BCS", {})),
        **(
            {"domain": {"type": "L", "cut_x": config.L_CUT_FRACTION, "cut_y": config.L_CUT_FRACTION}}
            if getattr(config, "DOMAIN", "rect") == "L"
            else {}
        ),
    }


def solve_fem_case(case: ThreeLayerCase, ne_x: int, ne_y: int, ne_z: int):
    start = time.perf_counter()
    x_nodes, y_nodes, z_nodes, u_grid = fem_solver.solve_three_layer_fem(fem_cfg(case, ne_x, ne_y, ne_z))
    elapsed = time.perf_counter() - start
    return np.asarray(x_nodes), np.asarray(y_nodes), np.asarray(z_nodes), np.asarray(u_grid), elapsed


def mae_pct(pred: np.ndarray, ref: np.ndarray) -> float:
    denom = float(np.max(np.abs(ref)))
    return 100.0 * float(np.mean(np.abs(pred - ref))) / denom if denom > 0 else 0.0


def max_pct(pred: np.ndarray, ref: np.ndarray) -> float:
    denom = float(np.max(np.abs(ref)))
    return 100.0 * float(np.max(np.abs(pred - ref))) / denom if denom > 0 else 0.0


def relative_l2_pct(pred: np.ndarray, ref: np.ndarray) -> float:
    """Discrete integral/relative L2 error on the evaluation grid."""
    denom = float(np.sqrt(np.sum(np.asarray(ref, dtype=float) ** 2)))
    if denom <= 0:
        return 0.0
    return 100.0 * float(np.sqrt(np.sum((np.asarray(pred) - np.asarray(ref)) ** 2))) / denom


def mean_abs_error(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(ref))))


def average_displacement_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    pred_avg = np.mean(np.asarray(pred), axis=tuple(range(np.asarray(pred).ndim - 1)))
    ref_avg = np.mean(np.asarray(ref), axis=tuple(range(np.asarray(ref).ndim - 1)))
    diff = pred_avg - ref_avg
    ref_norm = float(np.linalg.norm(ref_avg))
    return {
        "avg_ux_fem": float(ref_avg[0]),
        "avg_uy_fem": float(ref_avg[1]),
        "avg_uz_fem": float(ref_avg[2]),
        "avg_ux_pinn": float(pred_avg[0]),
        "avg_uy_pinn": float(pred_avg[1]),
        "avg_uz_pinn": float(pred_avg[2]),
        "avg_displacement_abs_error": float(np.linalg.norm(diff)),
        "avg_displacement_relative_error_pct": 100.0 * float(np.linalg.norm(diff)) / ref_norm if ref_norm > 0 else 0.0,
    }


def evaluate_case_grid(pinn, device: torch.device, case: ThreeLayerCase, ne_x: int, ne_y: int, ne_z: int) -> dict:
    x_nodes, y_nodes, z_nodes, u_fem, fem_seconds = solve_fem_case(case, ne_x, ne_y, ne_z)
    xg, yg, zg = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing="ij")
    pts = make_points(xg.ravel(), yg.ravel(), zg.ravel(), case)
    pred_start = time.perf_counter()
    u_pred = predict_displacement(pinn, device, pts).reshape(u_fem.shape)
    pinn_eval_seconds = time.perf_counter() - pred_start
    # Score only nodes that belong to the domain (all nodes for a rectangle); the
    # removed corner of an L-shape holds zeros that must not dilute the metrics.
    active = fem_solver.active_node_mask(fem_cfg(case, ne_x, ne_y, ne_z))
    vol = np.broadcast_to(active[:, :, None], u_fem.shape[:3])
    u_pred_all, u_fem_all = u_pred, u_fem
    u_pred, u_fem = u_pred_all[vol], u_fem_all[vol]
    top_pred = u_pred_all[:, :, -1, 2][active]
    top_ref = u_fem_all[:, :, -1, 2][active]
    return {
        "case": case,
        "x_nodes": x_nodes,
        "y_nodes": y_nodes,
        "z_nodes": z_nodes,
        "u_fem": u_fem_all,
        "u_pinn": u_pred_all,
        "fem_seconds": fem_seconds,
        "pinn_eval_seconds": pinn_eval_seconds,
        "volume_mae_pct": mae_pct(u_pred, u_fem),
        "volume_relative_l2_pct": relative_l2_pct(u_pred, u_fem),
        "volume_mean_abs_error": mean_abs_error(u_pred, u_fem),
        "volume_max_pct": max_pct(u_pred, u_fem),
        "top_uz_mae_pct": mae_pct(top_pred, top_ref),
        "top_uz_relative_l2_pct": relative_l2_pct(top_pred, top_ref),
        "top_uz_mean_abs_error": mean_abs_error(top_pred, top_ref),
        "top_uz_max_pct": max_pct(top_pred, top_ref),
        "peak_fem_uz": float(np.min(top_ref)),
        "peak_pinn_uz": float(np.min(top_pred)),
        **average_displacement_metrics(u_pred, u_fem),
        "n_eval_points": int(vol.sum()),
    }


def top_surface_grid(case: ThreeLayerCase, ne_x: int, ne_y: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_nodes = np.linspace(0.0, float(config.Lx), int(ne_x) + 1)
    y_nodes = np.linspace(0.0, float(config.Ly), int(ne_y) + 1)
    xg, yg = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    zg = np.full_like(xg, case.thickness, dtype=float)
    return xg, yg, zg


def load_patch_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (
        (x >= float(config.LOAD_PATCH_X[0]))
        & (x <= float(config.LOAD_PATCH_X[1]))
        & (y >= float(config.LOAD_PATCH_Y[0]))
        & (y <= float(config.LOAD_PATCH_Y[1]))
    )


def top_surface_metrics(u_top: np.ndarray, xg: np.ndarray, yg: np.ndarray) -> dict:
    uz = np.asarray(u_top)[..., 2]
    peak_downward_uz = float(np.min(uz))
    peak_downward_abs = float(abs(peak_downward_uz))
    patch = load_patch_mask(xg, yg)
    patch_uz = uz[patch] if np.any(patch) else uz.reshape(-1)
    mean_patch_uz = float(np.mean(patch_uz))
    mean_patch_abs = float(np.mean(np.abs(patch_uz)))
    return {
        "peak_downward_uz": peak_downward_uz,
        "peak_downward_abs": peak_downward_abs,
        "mean_patch_uz": mean_patch_uz,
        "mean_patch_abs": mean_patch_abs,
        "n_patch_points": int(patch_uz.size),
    }


def case_grid_top_surface_metrics(result: dict) -> dict:
    x_nodes = np.asarray(result["x_nodes"])
    y_nodes = np.asarray(result["y_nodes"])
    xg, yg = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    fem_top = np.asarray(result["u_fem"])[:, :, -1, :]
    pinn_top = np.asarray(result["u_pinn"])[:, :, -1, :]
    fem_metrics = top_surface_metrics(fem_top, xg, yg)
    pinn_metrics = top_surface_metrics(pinn_top, xg, yg)
    return {
        "x_grid": xg,
        "y_grid": yg,
        "fem_top_metrics": fem_metrics,
        "pinn_top_metrics": pinn_metrics,
    }


def evaluate_case_top_surface(pinn, device: torch.device, case: ThreeLayerCase, ne_x: int, ne_y: int) -> dict:
    xg, yg, zg = top_surface_grid(case, ne_x, ne_y)
    pts = make_points(xg.ravel(), yg.ravel(), zg.ravel(), case)
    start = time.perf_counter()
    u_top = predict_displacement(pinn, device, pts).reshape(xg.shape + (3,))
    elapsed = time.perf_counter() - start
    metrics = top_surface_metrics(u_top, xg, yg)
    metrics.update(
        {
            "case": case,
            "x_grid": xg,
            "y_grid": yg,
            "z_grid": zg,
            "u_top": u_top,
            "pinn_eval_seconds": elapsed,
            "n_eval_points": int(xg.size),
        }
    )
    return metrics


def fem_top_surface_metrics(case: ThreeLayerCase, ne_x: int, ne_y: int, ne_z: int) -> dict:
    x_nodes, y_nodes, _z_nodes, u_fem, fem_seconds = solve_fem_case(case, ne_x, ne_y, ne_z)
    xg, yg = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    u_top = u_fem[:, :, -1, :]
    metrics = top_surface_metrics(u_top, xg, yg)
    metrics.update(
        {
            "case": case,
            "x_grid": xg,
            "y_grid": yg,
            "u_top": u_top,
            "fem_seconds": fem_seconds,
            "n_eval_points": int(xg.size),
        }
    )
    return metrics


def random_interior_cases(n_cases: int, seed: int) -> list[ThreeLayerCase]:
    rng = np.random.default_rng(seed)
    e_min, e_max = [float(v) for v in getattr(config, "E_RANGE", [1.0, 10.0])]
    t1_min, t1_max = [float(v) for v in getattr(config, "T1_RANGE", [0.02, 0.10])]
    t2_min, t2_max = [float(v) for v in getattr(config, "T2_RANGE", [0.02, 0.10])]
    t3_min, t3_max = [float(v) for v in getattr(config, "T3_RANGE", [0.02, 0.10])]
    cases: list[ThreeLayerCase] = []
    for idx in range(n_cases):
        vals = rng.uniform([e_min, e_min, e_min, t1_min, t2_min, t3_min], [e_max, e_max, e_max, t1_max, t2_max, t3_max])
        cases.append(ThreeLayerCase(f"random_interior_{idx:03d}", *[float(v) for v in vals]))
    return cases


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def supervised_parameter_grid() -> set[tuple[float, float, float, float, float, float]]:
    out = set()
    for t1 in getattr(config, "DATA_T1_VALUES", []):
        for t2 in getattr(config, "DATA_T2_VALUES", []):
            for t3 in getattr(config, "DATA_T3_VALUES", []):
                for e1 in getattr(config, "DATA_E_VALUES", []):
                    for e2 in getattr(config, "DATA_E_VALUES", []):
                        for e3 in getattr(config, "DATA_E_VALUES", []):
                            out.add(tuple(round(float(v), 10) for v in (e1, e2, e3, t1, t2, t3)))
    return out


def is_supervised_parameter_case(case: ThreeLayerCase) -> bool:
    key = tuple(round(float(v), 10) for v in (case.e1, case.e2, case.e3, case.t1, case.t2, case.t3))
    return key in supervised_parameter_grid()


def rows_to_csv(path: Path, fieldnames: Iterable[str], rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
