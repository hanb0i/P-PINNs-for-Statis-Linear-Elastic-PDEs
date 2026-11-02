"""Train and evaluate the three-layer P2INN on reduced-symmetry cases.

Responds to the reviewer request to break the x-y symmetry of the benchmark.
Each case is a bundle of PINN_* env overrides (plate size, patch, per-edge BCs,
FEM mesh) layered on the canonical "Full framework" recipe, so everything else
matches the symmetric baseline. Results are written per (case, seed) and the
run is resumable.

Cases
  A        : original symmetric benchmark (all clamped, centered patch).
  B_cccc   : rectangle 1.5x1.0, off-center patch, all clamped (isolates geometry+load).
  B_ccss   : rectangle 1.5x1.0, x=0,y=0 clamped, x=Lx,y=Ly simply supported
             (classical analogue: uz=0 on the bottom edge line, rest of face free).
  C_cccc   : L-shaped plate (unit square minus the top-right quarter), all faces
             clamped, 0.2x0.2 patch in the lower leg off the diagonal.
  B_ccrf   : same plate, but S edges are uz=0 on the whole side face (earlier
             provisional definition, kept for comparison).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from run_three_layer_volume_ablation import (
    DATA_DIR,
    REPO_ROOT,
    WORKFLOW_DIR,
    build_base_env,
    build_variants,
    run,
)

# Cells stay ~1/16 wide so the rectangle is not under-resolved in x.
_B_GEOM = {
    "PINN_LX": "1.5",
    "PINN_LY": "1.0",
    "PINN_PATCH_X_FRAC": "0.55,0.75",  # patch 0.2a x 0.2b centered at (0.65a, 0.40b)
    "PINN_PATCH_Y_FRAC": "0.30,0.50",
    "PINN_FEM_NE_X": "24",
    "PINN_FEM_NE_Y": "16",
    "PINN_FEM_NE_Z": "8",
}

CASES: dict[str, dict] = {
    "A": {"env": {}, "mesh": (16, 16, 8)},
    "B_cccc": {"env": dict(_B_GEOM), "mesh": (24, 16, 8)},
    "B_ccss": {
        "env": {**_B_GEOM, "PINN_EDGE_BCS": "x1=simply_supported,y1=simply_supported"},
        "mesh": (24, 16, 8),
    },
    # L-shaped plate: unit square minus [0.5,1]x[0.5,1], all outer faces clamped.
    # The L is diagonally symmetric, so the 0.2x0.2 patch sits in the lower leg
    # at (0.72, 0.22), off the diagonal. Use with --env PINN_FORCE_SOFT_SIDE_BC=0
    # (mask-trained recipe, which helps clamped cases).
    "C_cccc": {
        "env": {
            "PINN_DOMAIN": "L",
            "PINN_PATCH_X_FRAC": "0.62,0.82",
            "PINN_PATCH_Y_FRAC": "0.12,0.32",
            "PINN_FEM_NE_X": "16",
            "PINN_FEM_NE_Y": "16",
            "PINN_FEM_NE_Z": "8",
        },
        "mesh": (16, 16, 8),
    },
    "B_ccrf": {
        "env": {**_B_GEOM, "PINN_EDGE_BCS": "x1=roller_face,y1=roller_face"},
        "mesh": (24, 16, 8),
    },
}


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["B_ccss"], choices=list(CASES))
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--eval-seed", type=int, default=20260430)
    parser.add_argument("--n-cases", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=str(DATA_DIR / "asymmetric_case_study"))
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VAL",
                        help="Extra PINN_* env override (repeatable), e.g. PINN_S_LINE_WEIGHT=10.")
    parser.add_argument("--tag", default="", help="Suffix for the run directory (use with --env).")
    args = parser.parse_args()

    recipe = dict(build_variants())["Full framework"]
    extra_env = dict(kv.split("=", 1) for kv in args.env)
    for name in args.cases:
        spec = CASES[name]
        for seed in args.train_seeds:
            run_dir = Path(args.out_dir).resolve() / (f"{name}_{args.tag}" if args.tag else name) / f"seed_{seed}"
            result_path = run_dir / "result.json"
            if result_path.exists():
                print(f"[skip] {name} seed={seed}", flush=True)
                continue
            env = {
                **build_base_env(args.epochs, seed, args.device),
                **recipe,
                **spec["env"],
                **extra_env,
                "PINN_OUT_DIR": str(run_dir),
            }
            checkpoint = run_dir / "pinn_model.pth"
            summary_path = run_dir / "random_summary.json"
            nx, ny, nz = spec["mesh"]

            t0 = time.time()
            if checkpoint.exists() and (run_dir / "training_timing.json").exists():
                print(f"[train] {name} seed={seed} (reusing checkpoint)", flush=True)
            else:
                print(f"[train] {name} seed={seed}", flush=True)
                run([sys.executable, str(WORKFLOW_DIR / "train.py")], env, run_dir / "train.log")
            wall = time.time() - t0

            print(f"[eval]  {name} seed={seed}", flush=True)
            if not summary_path.exists():
                run(
                    [
                        sys.executable,
                        str(REPO_ROOT / "scripts" / "run_random_interior_generalization.py"),
                        "--model-path", str(checkpoint),
                        "--n-cases", str(args.n_cases),
                        "--seed", str(args.eval_seed),
                        "--ne-x", str(nx), "--ne-y", str(ny), "--ne-z", str(nz),
                        "--out-csv", str(run_dir / "random.csv"),
                        "--out-summary", str(summary_path),
                    ],
                    env,
                    run_dir / "eval.log",
                )
            summary = json.loads(summary_path.read_text())
            timing_path = run_dir / "training_timing.json"
            timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
            result = {
                "case": name,
                "train_seed": seed,
                "eval_seed": args.eval_seed,
                "n_cases": args.n_cases,
                "epochs": args.epochs,
                "git_hash": git_hash(),
                "eval_mesh": list(spec["mesh"]),
                **{k: summary[k] for k in (
                    "volume_mae_pct_mean", "volume_mae_pct_worst",
                    "top_uz_mae_pct_mean", "top_uz_mae_pct_worst",
                    "volume_relative_l2_pct_mean", "volume_relative_l2_pct_worst",
                )},
                "train_seconds": timing.get("total_training_seconds"),
                "wall_seconds": wall,
                "device": timing.get("device"),
                "case_env": spec["env"],
                "extra_env": extra_env,
                "tag": args.tag,
            }
            result_path.write_text(json.dumps(result, indent=2))
            print(
                f"        VolMAE mean={result['volume_mae_pct_mean']:.3f}% "
                f"worst={result['volume_mae_pct_worst']:.3f}% ({result['train_seconds'] or wall:.0f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
