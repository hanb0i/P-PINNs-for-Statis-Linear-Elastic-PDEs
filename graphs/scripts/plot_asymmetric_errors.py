"""Top-surface error maps for the asymmetric cases (B_cccc, B_ccss, C, C with 16x weights).

For each model: FEM u_z, PINN u_z and |error| on the top surface for one fixed
parameter case. Each model is evaluated in a subprocess with the environment it
was trained under (geometry, edge BCs, domain), then the arrays are plotted together.

Run from the repo root:  python3 graphs/scripts/plot_asymmetric_errors.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from _common import DATA_DIR, REPO_ROOT, apply_ieee_style, save_figure

import numpy as np

SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

CASE = dict(e=(10.0, 1.0, 5.0), t=(0.05, 0.03, 0.02))  # Figure 1b configuration
OUT_NPZ = REPO_ROOT / "graphs" / "data" / "asymmetric_error_maps"

_B = {
    "PINN_LX": "1.5", "PINN_LY": "1.0",
    "PINN_PATCH_X_FRAC": "0.55,0.75", "PINN_PATCH_Y_FRAC": "0.30,0.50",
}
_C = {"PINN_DOMAIN": "L", "PINN_PATCH_X_FRAC": "0.62,0.82", "PINN_PATCH_Y_FRAC": "0.12,0.32"}

# name -> (title, checkpoint, env, mesh)
MODELS = {
    "B_cccc": ("B, CCCC (mask-trained)", "hardmask_stage2/B_cccc_hardmask/seed_0", _B, (24, 16, 8)),
    "B_ccss": ("B, CCSS (soft-trained)", "asymmetric_stage2/B_ccss/seed_0",
               {**_B, "PINN_EDGE_BCS": "x1=simply_supported,y1=simply_supported"}, (24, 16, 8)),
    "C": ("C, L-shape (mask-trained)", "c_study_stage2/C_cccc_hardmask/seed_0", _C, (16, 16, 8)),
    "C_w16": ("C, L-shape, 16x weights", "c_variants/stage2/C_cccc_wd16/seed_0", _C, (16, 16, 8)),
}


def dump(name: str) -> None:
    import torch  # noqa: F401
    import three_layer_experiment_utils as U

    _, ckpt, _, (nx, ny, nz) = MODELS[name]
    device = U.select_device()
    pinn, _ = U.load_pinn(device, DATA_DIR / ckpt / "pinn_model.pth")
    case = U.ThreeLayerCase("map", *CASE["e"], *CASE["t"])
    r = U.evaluate_case_grid(pinn, device, case, nx, ny, nz)
    from fem_solver import active_node_mask
    active = active_node_mask(U.fem_cfg(case, nx, ny, nz))
    fem = r["u_fem"][:, :, -1, 2].copy()
    pred = r["u_pinn"][:, :, -1, 2].copy()
    fem[~active] = np.nan
    pred[~active] = np.nan
    OUT_NPZ.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ / f"{name}.npz", x=r["x_nodes"], y=r["y_nodes"], fem=fem, pred=pred,
             vol_mae=r["volume_mae_pct"], rel_l2=r["volume_relative_l2_pct"],
             peak_fem=r["peak_fem_uz"], peak_pinn=r["peak_pinn_uz"])


GROUPS = {"fig_asym_errors_B": ["B_cccc", "B_ccss"], "fig_asym_errors_C": ["C", "C_w16"]}


def plot() -> None:
    for stem, names in GROUPS.items():
        _plot_group(stem, names)


def _plot_group(stem: str, names: list) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    apply_ieee_style()
    fig, axes = plt.subplots(len(names), 3, figsize=(7.2, 2.1 * len(names)), squeeze=False)
    patch = {"B": ((0.55 * 1.5, 0.30), 0.2 * 1.5, 0.2), "C": ((0.62, 0.12), 0.2, 0.2)}
    for row, name in enumerate(names):
        d = np.load(OUT_NPZ / f"{name}.npz")
        x, y = d["x"], d["y"]
        fem, pred = d["fem"], d["pred"]
        err = np.abs(pred - fem)
        vmax = np.nanmax(np.abs(fem))
        ext = (x.min(), x.max(), y.min(), y.max())
        panels = [(fem, "viridis", r"FEM $u_z$", (-vmax, 0.0)),
                  (pred, "viridis", r"P2INN $u_z$", (-vmax, 0.0)),
                  (err, "magma", r"$|u_z^{P2INN}-u_z^{FEM}|$", (0.0, vmax))]
        for col, (arr, cmap, title, (lo, hi)) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(arr.T, origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi, aspect="equal")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            (px, w, h) = patch["C" if name.startswith("C") else "B"]
            ax.add_patch(Rectangle(px, w, h, fill=False, edgecolor="white", linestyle="--", linewidth=0.8))
            if row == 0:
                ax.set_title(title, fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{MODELS[name][0]}\ny", fontsize=7)
            if row == len(names) - 1:
                ax.set_xlabel("x", fontsize=7)
            ax.tick_params(labelsize=6)
        axes[row, 2].text(0.02, 0.97, f"VolMAE {float(d['vol_mae']):.1f}%\nrel. L2 {float(d['rel_l2']):.0f}%",
                          transform=axes[row, 2].transAxes, va="top", fontsize=6, color="white")
    fig.tight_layout()
    for p in save_figure(fig, stem):
        print("wrote", p)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--plot-only":
        plot()
        return
    if len(sys.argv) > 2 and sys.argv[1] == "--dump":
        dump(sys.argv[2])
        return
    from run_asymmetric_case_study import CASES  # noqa: F401  (documents mesh/env origin)
    from run_three_layer_volume_ablation import build_base_env, build_variants

    recipe = dict(build_variants())["Full framework"]
    for name, (_, _, env, (nx, ny, nz)) in MODELS.items():
        full_env = {**os.environ, **build_base_env(400, 0, None), **recipe, **env,
                    "PINN_FEM_NE_X": str(nx), "PINN_FEM_NE_Y": str(ny), "PINN_FEM_NE_Z": str(nz)}
        subprocess.run([sys.executable, __file__, "--dump", name], env=full_env, check=True)
        print("done", name, flush=True)
    plot()


if __name__ == "__main__":
    main()
