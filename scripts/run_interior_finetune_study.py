"""Test whether interior-parameter FEM supervision explains the Table 5 checkpoint.

Starting from the corner-supervised stage-1 checkpoints of the seed study, run
the second-stage budget of the released checkpoint (about 1,200 SOAP epochs
then 100 L-BFGS steps, warm started) under two arms:

  control   continue training with the original corner-only supervision
  interior  continue training with 64 random interior FEM cases
            (supervision seed 20260429, as in supervised_retrain_log.txt)

`stage1` is the unmodified starting checkpoint. Every model is scored on two
sets of 100 random interior cases (evaluation seeds 20260430 and 20260501).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from run_three_layer_volume_ablation import (
    DATA_DIR,
    REPO_ROOT,
    WORKFLOW_DIR,
    build_base_env,
    build_variants,
    run,
)

EVAL_SEEDS = (20260430, 20260501)
STAGE1_DIR = DATA_DIR / "three_layer_seed_study_gpu" / "full_framework"
OUT_DIR = DATA_DIR / "interior_finetune_study"

INTERIOR_ENV = {
    "PINN_RANDOM_SUPERVISION_CASES": "64",
    "PINN_RANDOM_SUPERVISION_SEED": "20260429",
    "PINN_SUPERVISION_TOP_FRACTION": "0.6",
    "PINN_SUPERVISION_INTERFACE_FRACTION": "0.3",
}


def evaluate(checkpoint: Path, env: dict[str, str], run_dir: Path, n_cases: int) -> dict:
    out = {}
    for eval_seed in EVAL_SEEDS:
        summary_path = run_dir / f"eval_{eval_seed}_summary.json"
        run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_random_interior_generalization.py"),
                "--model-path", str(checkpoint),
                "--n-cases", str(n_cases),
                "--seed", str(eval_seed),
                "--out-csv", str(run_dir / f"eval_{eval_seed}.csv"),
                "--out-summary", str(summary_path),
            ],
            env,
            run_dir / f"eval_{eval_seed}.log",
        )
        s = json.loads(summary_path.read_text())
        out[str(eval_seed)] = {
            "volume_mae_pct_mean": s["volume_mae_pct_mean"],
            "volume_mae_pct_worst": s["volume_mae_pct_worst"],
        }
    return out


def summarize(out_dir: Path) -> None:
    table: dict[str, list[dict]] = {}
    for path in sorted(out_dir.glob("*/seed_*/result.json")):
        r = json.loads(path.read_text())
        table.setdefault(r["arm"], []).append(r)
    for arm in ("stage1", "control", "interior"):
        rs = table.get(arm, [])
        if not rs:
            continue
        for eval_seed in EVAL_SEEDS:
            mean = [r["eval"][str(eval_seed)]["volume_mae_pct_mean"] for r in rs]
            worst = [r["eval"][str(eval_seed)]["volume_mae_pct_worst"] for r in rs]
            sd = lambda v: statistics.stdev(v) if len(v) > 1 else float("nan")
            print(
                f"{arm:<9} eval_seed={eval_seed} n={len(rs)} "
                f"mean {statistics.mean(mean):.2f}±{sd(mean):.2f} [{min(mean):.2f}, {max(mean):.2f}]  "
                f"worst {statistics.mean(worst):.2f}±{sd(worst):.2f} [{min(worst):.2f}, {max(worst):.2f}]"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["stage1", "control", "interior"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lbfgs", type=int, default=100)
    parser.add_argument("--n-cases", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    if args.summarize:
        summarize(out_root)
        return

    full = dict(build_variants())["Full framework"]
    for arm in args.arms:
        for seed in args.seeds:
            run_dir = out_root / arm / f"seed_{seed}"
            result_path = run_dir / "result.json"
            if result_path.exists():
                print(f"[skip] {arm} seed={seed}", flush=True)
                continue
            start = STAGE1_DIR / f"seed_{seed}" / "pinn_model.pth"
            if not start.exists():
                raise SystemExit(f"Missing stage-1 checkpoint: {start}")

            env = {**build_base_env(args.epochs, seed, args.device), **full, "PINN_OUT_DIR": str(run_dir)}
            if arm == "stage1":
                checkpoint = start
            else:
                env.update(
                    {
                        "PINN_WARM_START": "1",
                        "PINN_WARM_START_PATH": str(start),
                        "PINN_EPOCHS_LBFGS": str(args.lbfgs),
                    }
                )
                if arm == "interior":
                    env.update(INTERIOR_ENV)
                print(f"[train] {arm} seed={seed}", flush=True)
                run([sys.executable, str(WORKFLOW_DIR / "train.py")], env, run_dir / "train.log")
                checkpoint = run_dir / "pinn_model.pth"

            print(f"[eval]  {arm} seed={seed}", flush=True)
            result = {
                "arm": arm,
                "train_seed": seed,
                "epochs": args.epochs if arm != "stage1" else 0,
                "lbfgs_steps": args.lbfgs if arm != "stage1" else 0,
                "start_checkpoint": str(start.relative_to(REPO_ROOT)),
                "eval": evaluate(checkpoint, env, run_dir, args.n_cases),
            }
            result_path.write_text(json.dumps(result, indent=2))
            e = result["eval"]
            print("        " + "  ".join(f"[{k}] {v['volume_mae_pct_mean']:.2f}/{v['volume_mae_pct_worst']:.2f}" for k, v in e.items()), flush=True)


if __name__ == "__main__":
    main()
