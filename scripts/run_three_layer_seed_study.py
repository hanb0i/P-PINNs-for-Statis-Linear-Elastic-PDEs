"""Multi-seed repeatability study for the three-layer P2INN.

Each (variant, training seed) pair is trained from scratch with the canonical
recipe in `run_three_layer_volume_ablation.py` and scored on one fixed set of
random interior cases. The training seed varies; the evaluation seed does not,
so every model sees identical test cases. Results are written per run so the
study can be resumed and extended with more seeds.
"""

from __future__ import annotations

import argparse
import json
import os
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
    slugify,
)

DEFAULT_EVAL_SEED = 20260430


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="*", default=["Full framework"],
                        help="Variant names (see build_variants); 'all' runs every variant.")
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--eval-seed", type=int, default=DEFAULT_EVAL_SEED)
    parser.add_argument("--n-cases", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=str(DATA_DIR / "three_layer_seed_study"))
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    available = dict(build_variants())
    names = list(available) if args.variants == ["all"] else args.variants
    unknown = [n for n in names if n not in available]
    if unknown:
        raise SystemExit(f"Unknown variants {unknown}; choose from {list(available)}")

    for name in names:
        for seed in args.train_seeds:
            slug = slugify(name)
            run_dir = out_root / slug / f"seed_{seed}"
            result_path = run_dir / "result.json"
            if result_path.exists():
                print(f"[skip] {name} seed={seed} (result exists)", flush=True)
                continue

            env = {
                **build_base_env(args.epochs, seed, args.device),
                **available[name],
                "PINN_OUT_DIR": str(run_dir),
            }
            checkpoint = run_dir / "pinn_model.pth"
            summary_path = run_dir / "random100_summary.json"

            print(f"[train] {name} seed={seed}", flush=True)
            t0 = time.time()
            run([sys.executable, str(WORKFLOW_DIR / "train.py")], env, run_dir / "train.log")
            wall = time.time() - t0

            print(f"[eval]  {name} seed={seed}", flush=True)
            run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "run_random_interior_generalization.py"),
                    "--model-path", str(checkpoint),
                    "--n-cases", str(args.n_cases),
                    "--seed", str(args.eval_seed),
                    "--out-csv", str(run_dir / "random100.csv"),
                    "--out-summary", str(summary_path),
                ],
                env,
                run_dir / "eval.log",
            )
            summary = json.loads(summary_path.read_text())
            timing_path = run_dir / "training_timing.json"
            timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
            result = {
                "variant": name,
                "train_seed": seed,
                "eval_seed": args.eval_seed,
                "n_cases": args.n_cases,
                "epochs": args.epochs,
                "git_hash": git_hash(),
                "volume_mae_pct_mean": summary["volume_mae_pct_mean"],
                "volume_mae_pct_worst": summary["volume_mae_pct_worst"],
                "top_uz_mae_pct_mean": summary["top_uz_mae_pct_mean"],
                "top_uz_mae_pct_worst": summary["top_uz_mae_pct_worst"],
                "train_seconds": timing.get("total_training_seconds"),
                "wall_seconds": wall,
                "device": timing.get("device"),
                "recipe_env": {k: v for k, v in env.items() if k.startswith("PINN_")},
                "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
            }
            result_path.write_text(json.dumps(result, indent=2))
            print(
                f"        VolMAE mean={result['volume_mae_pct_mean']:.3f}% "
                f"worst={result['volume_mae_pct_worst']:.3f}% "
                f"({result['train_seconds'] or wall:.0f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
