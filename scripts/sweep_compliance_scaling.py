"""Post-hoc compliance-scaling sweep for a trained P2INN checkpoint.

The network output v is mapped to displacement as
    u = s * v / E_avg**p * (H / t_total)**alpha.
The checkpoint is trained with one (alpha, p); this re-evaluates the SAME
checkpoint under other scalings against FEM, with FEM solved once per case.
It is a diagnostic: does the clamped-fitted scaling still suit this BC set?

Run with the case's PINN_* env (see run_asymmetric_case_study.CASES), e.g. via
`--case B_ccss`, which sets the env then re-executes this script.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

if "--_child" not in sys.argv:
    # Re-exec with the case env so pinn_config picks up geometry/BCs at import.
    import subprocess
    from run_asymmetric_case_study import CASES

    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=list(CASES))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-cases", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    env = {**os.environ, **CASES[a.case]["env"], "PINN_DEVICE": os.getenv("PINN_DEVICE", "cpu")}
    nx, ny, nz = CASES[a.case]["mesh"]
    sys.exit(subprocess.call(
        [sys.executable, __file__, "--_child", "--ckpt", a.ckpt, "--n-cases", str(a.n_cases),
         "--mesh", f"{nx},{ny},{nz}", "--out", a.out], env=env))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from three_layer_experiment_utils import (  # noqa: E402
    config, evaluate_case_grid, fem_cfg, load_pinn, make_points, random_interior_cases,
    select_device, solve_fem_case,
)
import fem_solver as fem_solver_mod  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--_child", action="store_true")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--n-cases", type=int, default=30)
ap.add_argument("--mesh", default="16,16,8")
ap.add_argument("--out", required=True)
args = ap.parse_args()
nx, ny, nz = (int(v) for v in args.mesh.split(","))

device = select_device()
pinn, _ = load_pinn(device, args.ckpt)
H = float(config.H)

refs, raws, Es, Ts = [], [], [], []
for case in random_interior_cases(args.n_cases, 20260430):
    xn, yn, zn, u_fem, _ = solve_fem_case(case, nx, ny, nz)
    xg, yg, zg = np.meshgrid(xn, yn, zn, indexing="ij")
    pts = make_points(xg.ravel(), yg.ravel(), zg.ravel(), case)
    with torch.no_grad():
        v = pinn(torch.tensor(pts, dtype=torch.float32, device=device)).cpu().numpy()
    # Score only nodes inside the domain (matters for the L-shape; all nodes for a rectangle).
    act = fem_solver_mod.active_node_mask(fem_cfg(case, nx, ny, nz))
    sel = np.broadcast_to(act[:, :, None], u_fem.shape[:3]).ravel()
    refs.append(u_fem.reshape(-1, 3)[sel]); raws.append(v[sel])
    Es.append(np.mean(case.e)); Ts.append(case.thickness)

def metrics(alpha, p, s):
    vm, l2 = [], []
    for ref, v, E, T in zip(refs, raws, Es, Ts):
        u = s * v / E ** p * (H / T) ** alpha
        vm.append(100 * np.mean(np.abs(u - ref)) / np.max(np.abs(ref)))
        l2.append(100 * np.linalg.norm(u - ref) / np.linalg.norm(ref))
    return float(np.mean(vm)), float(np.mean(l2))

def best_global_s(alpha, p):
    # Least squares for one s shared by all cases (on max-normalised fields).
    num = den = 0.0
    for ref, v, E, T in zip(refs, raws, Es, Ts):
        base = v / E ** p * (H / T) ** alpha
        w = 1.0 / np.max(np.abs(ref))
        num += np.sum((base * w) * (ref * w)); den += np.sum((base * w) ** 2)
    return num / den

trained = (float(config.THICKNESS_COMPLIANCE_ALPHA), float(config.E_COMPLIANCE_POWER))
alphas = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
ps = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10]
rows = []
for a_ in alphas:
    for p_ in ps:
        s_star = best_global_s(a_, p_)
        m1, l1 = metrics(a_, p_, 1.0)
        m2, l2_ = metrics(a_, p_, s_star)
        rows.append(dict(alpha=a_, p=p_, s_star=s_star, vol_mae_s1=m1, rel_l2_s1=l1, vol_mae_sstar=m2, rel_l2_sstar=l2_))

base = next(r for r in rows if (r["alpha"], r["p"]) == trained)
best1 = min(rows, key=lambda r: r["vol_mae_s1"])
bests = min(rows, key=lambda r: r["vol_mae_sstar"])
out = dict(ckpt=args.ckpt, n_cases=args.n_cases, mesh=[nx, ny, nz], edge_bcs=dict(getattr(config, "EDGE_BCS", {})),
           trained_alpha_p=trained, baseline=base, best_with_s1=best1, best_with_global_s=bests, grid=rows)
json.dump(out, open(args.out, "w"), indent=2)
f = lambda r: f"alpha={r['alpha']} p={r['p']:.2f} s*={r['s_star']:.2f} | s=1: VolMAE {r['vol_mae_s1']:.2f} relL2 {r['rel_l2_s1']:.1f} | s*: VolMAE {r['vol_mae_sstar']:.2f} relL2 {r['rel_l2_sstar']:.1f}"
print("baseline ", f(base)); print("best s=1 ", f(best1)); print("best s*  ", f(bests))
