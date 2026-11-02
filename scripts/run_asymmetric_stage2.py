"""Stage-2 (longer training) for the reduced-symmetry cases.

Warm starts from the stage-1 checkpoints of run_asymmetric_case_study.py and
runs the Table 5 budget (1200 SOAP epochs + 100 L-BFGS steps) with the same
corner-only FEM supervision. This is the "control" arm of
run_interior_finetune_study.py, which the reproducibility doc finds is what
produces the Table 5 accuracy. Resumable; relative --out-dir is resolved.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from run_asymmetric_case_study import CASES, git_hash
from run_three_layer_volume_ablation import DATA_DIR, REPO_ROOT, WORKFLOW_DIR, build_base_env, build_variants, run
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", default=["B_cccc", "B_ccss"], choices=list(CASES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--lbfgs", type=int, default=100)
    ap.add_argument("--n-cases", type=int, default=100)
    ap.add_argument("--eval-seed", type=int, default=20260430)
    ap.add_argument("--device", default=None)
    ap.add_argument("--stage1-dir", default=str(DATA_DIR / "asymmetric_case_study_100"))
    ap.add_argument("--out-dir", default=str(DATA_DIR / "asymmetric_stage2"))
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VAL", help="Extra PINN_* env override (repeatable).")
    ap.add_argument("--tag", default="", help="Suffix for the output run directory.")
    ap.add_argument("--stage1-tag", default="", help="Suffix of the stage-1 run directory to warm start from.")
    args = ap.parse_args()

    recipe = dict(build_variants())["Full framework"]
    extra_env = dict(kv.split("=", 1) for kv in args.env)
    for name in args.cases:
        spec = CASES[name]
        nx, ny, nz = spec["mesh"]
        for seed in args.seeds:
            start = Path(args.stage1_dir).resolve() / (f"{name}_{args.stage1_tag}" if args.stage1_tag else name) / f"seed_{seed}" / "pinn_model.pth"
            if not start.exists():
                raise SystemExit(f"Missing stage-1 checkpoint: {start}")
            run_dir = Path(args.out_dir).resolve() / (f"{name}_{args.tag}" if args.tag else name) / f"seed_{seed}"
            result_path = run_dir / "result.json"
            if result_path.exists():
                print(f"[skip] {name} seed={seed}", flush=True)
                continue
            env = {
                **build_base_env(args.epochs, seed, args.device),
                **recipe,
                **spec["env"],
                "PINN_WARM_START": "1",
                "PINN_WARM_START_PATH": str(start),
                "PINN_EPOCHS_LBFGS": str(args.lbfgs),
                **extra_env,
                "PINN_OUT_DIR": str(run_dir),
            }
            checkpoint = run_dir / "pinn_model.pth"
            summary_path = run_dir / "random_summary.json"
            t0 = time.time()
            if not (checkpoint.exists() and (run_dir / "training_timing.json").exists()):
                print(f"[train] {name} seed={seed}", flush=True)
                run([sys.executable, str(WORKFLOW_DIR / "train.py")], env, run_dir / "train.log")
            wall = time.time() - t0
            if not summary_path.exists():
                print(f"[eval]  {name} seed={seed}", flush=True)
                run(
                    [sys.executable, str(REPO_ROOT / "scripts" / "run_random_interior_generalization.py"),
                     "--model-path", str(checkpoint), "--n-cases", str(args.n_cases), "--seed", str(args.eval_seed),
                     "--ne-x", str(nx), "--ne-y", str(ny), "--ne-z", str(nz),
                     "--out-csv", str(run_dir / "random.csv"), "--out-summary", str(summary_path)],
                    env, run_dir / "eval.log",
                )
            s = json.loads(summary_path.read_text())
            timing = json.loads((run_dir / "training_timing.json").read_text()) if (run_dir / "training_timing.json").exists() else {}
            result = {
                "case": name, "stage": 2, "train_seed": seed, "eval_seed": args.eval_seed, "n_cases": args.n_cases,
                "epochs": args.epochs, "lbfgs_steps": args.lbfgs, "git_hash": git_hash(),
                "start_checkpoint": str(start), "eval_mesh": [nx, ny, nz],
                **{k: s[k] for k in ("volume_mae_pct_mean", "volume_mae_pct_worst", "top_uz_mae_pct_mean",
                                     "top_uz_mae_pct_worst", "volume_relative_l2_pct_mean", "volume_relative_l2_pct_worst")},
                "train_seconds": timing.get("total_training_seconds"), "wall_seconds": wall,
                "device": timing.get("device"), "case_env": spec["env"], "extra_env": extra_env, "tag": args.tag,
            }
            result_path.write_text(json.dumps(result, indent=2))
            print(f"        VolMAE mean={result['volume_mae_pct_mean']:.3f}% worst={result['volume_mae_pct_worst']:.3f}% "
                  f"relL2={result['volume_relative_l2_pct_mean']:.1f}% ({result['train_seconds'] or wall:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
