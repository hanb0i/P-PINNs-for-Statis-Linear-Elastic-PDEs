"""Train and evaluate the paper's three-layer reverse ablation with volume MAE."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "graphs" / "data"
WORKFLOW_DIR = REPO_ROOT / "three-layer-workflow"


def run(cmd: list[str], env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env={**os.environ, **env},
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}); see {log_path}")


def slugify(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def build_base_env(epochs: int, seed: int, device: str | None) -> dict[str, str]:
    base = {
        "MPLCONFIGDIR": str(REPO_ROOT / ".mplconfig"),
        "PYTHONPYCACHEPREFIX": str(REPO_ROOT / ".pycache"),
        "PINN_WARM_START": "0",
        "PINN_EPOCHS_ADAM": str(epochs),
        "PINN_EPOCHS_LBFGS": "0",
        "PINN_SEED": str(seed),
        "PINN_SUPERVISION_CACHE": "1",
        "PINN_REGEN_SUPERVISION": "0",
        "PINN_ADAPTIVE_RESAMPLE_EVERY": "0",
    }
    if device:
        base["PINN_DEVICE"] = device
    return base


def build_variants() -> list[tuple[str, dict[str, str]]]:
    """Canonical three-layer recipe plus the reverse-ablation overrides."""
    full = {
        "PINN_E_COMPLIANCE_POWER": "0.95",
        "PINN_THICKNESS_COMPLIANCE_ALPHA": "3",
        "PINN_DISPLACEMENT_COMPLIANCE_SCALE": "1",
        "PINN_PDE_DECOMPOSE_BY_LAYER": "1",
        "PINN_W_INTERFACE_U": "300",
        "PINN_W_PDE": "10",
        "PINN_W_DATA": "400",
        "PINN_N_INTERFACE": "16000",
        "PINN_INTERFACE_SAMPLE_FRACTION": "0.75",
        "PINN_USE_SUPERVISION_DATA": "1",
        "PINN_N_DATA_POINTS": "36000",
        "PINN_SUPERVISION_THICKNESS_POWER": "3.0",
        "PINN_FEM_NE_X": "16",
        "PINN_FEM_NE_Y": "16",
        "PINN_FEM_NE_Z": "8",
    }

    def without(**changes: str) -> dict[str, str]:
        result = dict(full)
        result.update(changes)
        return result

    variants = [
        ("Full framework", full),
        (
            "- Compliance-aware scaling",
            without(
                PINN_E_COMPLIANCE_POWER="0",
                PINN_THICKNESS_COMPLIANCE_ALPHA="0",
            ),
        ),
        (
            "- Layerwise PDE decomposition",
            without(PINN_PDE_DECOMPOSE_BY_LAYER="0"),
        ),
        (
            "- Interface continuity",
            without(PINN_W_INTERFACE_U="0", PINN_N_INTERFACE="2000"),
        ),
        (
            "- Sparse FEM supervision",
            without(
                PINN_USE_SUPERVISION_DATA="0",
                PINN_W_DATA="0",
                PINN_N_DATA_POINTS="0",
            ),
        ),
        (
            "Vanilla PINN baseline",
            without(
                PINN_E_COMPLIANCE_POWER="0",
                PINN_THICKNESS_COMPLIANCE_ALPHA="0",
                PINN_PDE_DECOMPOSE_BY_LAYER="0",
                PINN_W_INTERFACE_U="0",
                PINN_N_INTERFACE="2000",
                PINN_INTERFACE_SAMPLE_FRACTION="0.25",
                PINN_USE_SUPERVISION_DATA="0",
                PINN_W_DATA="0",
                PINN_N_DATA_POINTS="0",
            ),
        ),
    ]
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--n-cases", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    runs_dir = DATA_DIR / "three_layer_volume_ablation_runs"
    logs_dir = DATA_DIR / "three_layer_volume_ablation_logs"
    out_csv = DATA_DIR / "three_layer_volume_ablation.csv"

    base = build_base_env(args.epochs, args.seed, args.device)
    variants = build_variants()

    rows = []
    for name, overrides in variants:
        slug = slugify(name)
        run_dir = runs_dir / slug
        checkpoint = run_dir / "pinn_model.pth"
        summary_path = run_dir / "random100_summary.json"
        cases_path = run_dir / "random100.csv"
        env = {**base, **overrides, "PINN_OUT_DIR": str(run_dir)}

        print(f"Training {name}...", flush=True)
        if not (args.skip_train and checkpoint.exists()):
            run(
                [sys.executable, str(WORKFLOW_DIR / "train.py")],
                env,
                logs_dir / f"{slug}_train.log",
            )

        print(f"Evaluating {name}...", flush=True)
        run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_random_interior_generalization.py"),
                "--model-path",
                str(checkpoint),
                "--n-cases",
                str(args.n_cases),
                "--seed",
                str(args.seed),
                "--out-csv",
                str(cases_path),
                "--out-summary",
                str(summary_path),
            ],
            env,
            logs_dir / f"{slug}_eval.log",
        )
        summary = json.loads(summary_path.read_text())
        row = {
            "variant": name,
            "mean_volume_mae_pct": f"{summary['volume_mae_pct_mean']:.4f}",
            "worst_volume_mae_pct": f"{summary['volume_mae_pct_worst']:.4f}",
            "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
        }
        rows.append(row)
        print(
            f"  volume MAE: mean={row['mean_volume_mae_pct']}%, "
            f"worst={row['worst_volume_mae_pct']}%",
            flush=True,
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
