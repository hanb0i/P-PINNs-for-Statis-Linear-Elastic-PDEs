"""Run the Table 5 second stage for every ablation variant.

Each variant's 400-epoch seed-study checkpoint is warm started and trained for
the second-stage budget (about 1,200 SOAP epochs then 100 L-BFGS steps) with
that variant's own recipe, so Tables 5 and 6 can use the same training budget.
Every model is scored on two sets of 100 random interior cases.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from run_interior_finetune_study import EVAL_SEEDS, evaluate
from run_three_layer_volume_ablation import (
    DATA_DIR,
    REPO_ROOT,
    WORKFLOW_DIR,
    build_base_env,
    build_variants,
    run,
    slugify,
)

STAGE1_DIR = DATA_DIR / "three_layer_seed_study_gpu"
OUT_DIR = DATA_DIR / "three_layer_second_stage"


def summarize(out_dir: Path) -> None:
    runs: dict[str, list[dict]] = {}
    for path in sorted(out_dir.glob("*/seed_*/result.json")):
        r = json.loads(path.read_text())
        runs.setdefault(r["variant"], []).append(r)
    sd = lambda v: statistics.stdev(v) if len(v) > 1 else float("nan")
    for variant, rs in runs.items():
        for eval_seed in EVAL_SEEDS:
            mean = [r["eval"][str(eval_seed)]["volume_mae_pct_mean"] for r in rs]
            worst = [r["eval"][str(eval_seed)]["volume_mae_pct_worst"] for r in rs]
            print(
                f"{variant:<32} eval_seed={eval_seed} n={len(rs)} "
                f"mean {statistics.mean(mean):.2f}±{sd(mean):.2f} [{min(mean):.2f}, {max(mean):.2f}]  "
                f"worst {statistics.mean(worst):.2f}±{sd(worst):.2f} [{min(worst):.2f}, {max(worst):.2f}]"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="*", default=["all"])
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

    available = dict(build_variants())
    names = list(available) if args.variants == ["all"] else args.variants
    unknown = [n for n in names if n not in available]
    if unknown:
        raise SystemExit(f"Unknown variants {unknown}; choose from {list(available)}")

    for name in names:
        for seed in args.seeds:
            slug = slugify(name)
            run_dir = out_root / slug / f"seed_{seed}"
            result_path = run_dir / "result.json"
            if result_path.exists():
                print(f"[skip] {name} seed={seed}", flush=True)
                continue
            start = STAGE1_DIR / slug / f"seed_{seed}" / "pinn_model.pth"
            if not start.exists():
                raise SystemExit(f"Missing stage-1 checkpoint: {start}")

            env = {
                **build_base_env(args.epochs, seed, args.device),
                **available[name],
                "PINN_OUT_DIR": str(run_dir),
                "PINN_WARM_START": "1",
                "PINN_WARM_START_PATH": str(start),
                "PINN_EPOCHS_LBFGS": str(args.lbfgs),
            }
            print(f"[train] {name} seed={seed}", flush=True)
            run([sys.executable, str(WORKFLOW_DIR / "train.py")], env, run_dir / "train.log")

            print(f"[eval]  {name} seed={seed}", flush=True)
            result = {
                "variant": name,
                "train_seed": seed,
                "epochs": args.epochs,
                "lbfgs_steps": args.lbfgs,
                "start_checkpoint": str(start.relative_to(REPO_ROOT)),
                "eval": evaluate(run_dir / "pinn_model.pth", env, run_dir, args.n_cases),
            }
            result_path.write_text(json.dumps(result, indent=2))
            print("        " + "  ".join(
                f"[{k}] {v['volume_mae_pct_mean']:.2f}/{v['volume_mae_pct_worst']:.2f}"
                for k, v in result["eval"].items()), flush=True)


if __name__ == "__main__":
    main()
