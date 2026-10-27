"""Check paper-facing result artifacts against the release tables.

This script verifies the checked-in artifacts used by the manuscript. It is a
fast release check; it does not rerun the expensive FEM/PINN sweeps. Regenerate
the one-layer ablation with `graphs/scripts/eval_one_layer_ablation.py` when the
one-layer checkpoints change.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "graphs" / "data"
FIGURES = REPO_ROOT / "graphs" / "figures"


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _assert_close(label: str, actual: float, expected: float, tol: float = 0.015) -> None:
    if not math.isclose(float(actual), float(expected), abs_tol=tol):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")
    print(f"OK  {label}: {float(actual):.4g}")


def _row_by_variant(rows: list[dict[str, str]], variant: str) -> dict[str, str]:
    for row in rows:
        if row.get("variant") == variant:
            return row
    raise AssertionError(f"Missing row variant={variant!r}")


def check_one_layer_ablation() -> None:
    rows = _csv_rows(DATA / "one_layer_paper_ablation.csv")
    expected = {
        "Full framework": (1.62, 3.15),
        "- Compliance-aware scaling": (10.76, 17.73),
        "Vanilla PINN": (22.17, 24.01),
    }
    for variant, (mean, worst) in expected.items():
        row = _row_by_variant(rows, variant)
        _assert_close(f"one-layer ablation {variant} mean", float(row["mean_mae"]), mean)
        _assert_close(f"one-layer ablation {variant} worst", float(row["worst_mae"]), worst)
        checkpoint = REPO_ROOT / row["checkpoint"]
        if not checkpoint.exists():
            raise AssertionError(f"Missing checkpoint for {variant}: {checkpoint}")
        print(f"OK  checkpoint exists: {row['checkpoint']}")


def check_three_layer_ablation() -> None:
    """Recompute the multi-seed ablation table from the per-seed results."""
    rows = _csv_rows(DATA / "three_layer_paper_ablation.csv")
    runs: dict[str, list[dict]] = {}
    for path in sorted((DATA / "three_layer_seed_study_gpu").glob("*/seed_*/result.json")):
        result = _json(path)
        runs.setdefault(result["variant"], []).append(result)
    for row in rows:
        variant = row["variant"]
        rs = runs.get(variant)
        if not rs or len(rs) != int(row["n_seeds"]):
            raise AssertionError(f"{variant}: CSV lists {row['n_seeds']} seeds, found {len(rs or [])} results")
        mean = [r["volume_mae_pct_mean"] for r in rs]
        worst = [r["volume_mae_pct_worst"] for r in rs]
        _assert_close(f"three-layer ablation {variant} mean", sum(mean) / len(mean), float(row["mean_volmae"]))
        _assert_close(f"three-layer ablation {variant} worst", sum(worst) / len(worst), float(row["worst_volmae"]))
        for r in rs:
            if not (REPO_ROOT / r["checkpoint"]).exists():
                raise AssertionError(f"Missing checkpoint: {r['checkpoint']}")


def check_three_layer_final_supervised() -> None:
    """Verify the executable artifact behind the paper's three-layer row."""
    summary_path = DATA / "three_layer_final_supervised_random100_summary.json"
    csv_path = DATA / "three_layer_final_supervised_random100.csv"
    checkpoint = DATA / "random_supervision_runs" / "three_layer_final_supervised" / "pinn_model.pth"
    summary = _json(summary_path)

    if not checkpoint.exists():
        raise AssertionError(f"Missing three-layer final supervised checkpoint: {checkpoint}")
    print("OK  checkpoint exists: graphs/data/random_supervision_runs/three_layer_final_supervised/pinn_model.pth")

    if int(summary["seed"]) != 20260430 or int(summary["n_cases"]) != 100:
        raise AssertionError(f"Unexpected three-layer final supervised protocol: {summary_path}")
    _assert_close("three-layer final supervised random100 mean", summary["top_uz_mae_pct_mean"], 5.74)
    _assert_close("three-layer final supervised random100 worst", summary["top_uz_mae_pct_worst"], 10.81)

    rows = _csv_rows(csv_path)
    worst = max(rows, key=lambda row: float(row["top_uz_mae_pct"]))
    if worst["case_id"] != "random_interior_046":
        raise AssertionError(f"Unexpected worst three-layer random100 case: {worst['case_id']}")
    _assert_close("three-layer final supervised worst case 046", worst["top_uz_mae_pct"], 10.81)


def check_verification_table() -> None:
    rows = _csv_rows(DATA / "paper_verification_results.csv")
    expected = {
        "One-layer": (1.56, 2.87),
        "Three-layer": (2.53, 4.66),
    }
    for config, (mean, worst) in expected.items():
        row = next((r for r in rows if r.get("configuration") == config), None)
        if row is None:
            raise AssertionError(f"Missing verification row: {config}")
        _assert_close(f"verification {config} mean", float(row["mean_mae"]), mean)
        _assert_close(f"verification {config} worst", float(row["worst_mae"]), worst)


def check_timing_tables() -> None:
    one = _json(DATA / "one_layer_efficiency_timing_summary.json")
    three = _json(DATA / "efficiency_timing_summary.json")
    _assert_close("one-layer FEM seconds", one["fem_seconds_mean"], 0.47, tol=0.01)
    _assert_close("one-layer P2INN seconds", one["pinn_eval_seconds_mean"], 0.0025, tol=0.0002)
    _assert_close("three-layer FEM seconds", three["fem_seconds_mean"], 0.44, tol=0.01)
    _assert_close("three-layer P2INN seconds", three["pinn_eval_seconds_mean"], 0.0030, tol=0.0002)
    train = three.get("one_time_training_cost", {}).get("total_training_seconds")
    if train is not None:
        print(f"WARN three-layer timing artifact records training as {train / 60.0:.2f} min; manuscript says 4.5 min.")


def check_figures() -> None:
    required = [
        "fig_geometry_bc.pdf",
        "fig_1l_top.pdf",
        "fig_1l_cross.pdf",
        "fig_3l_top.pdf",
        "fig_3l_cross.pdf",
        "fig_1l_errors.pdf",
        "fig_3l_errors.pdf",
        "fig_one_layer_ablation.pdf",
        "fig_efficiency_timing.pdf",
    ]
    for name in required:
        path = FIGURES / name
        if not path.exists():
            raise AssertionError(f"Missing figure: {path}")
        print(f"OK  figure exists: graphs/figures/{name}")


def main() -> None:
    check_verification_table()
    check_one_layer_ablation()
    check_three_layer_ablation()
    check_three_layer_final_supervised()
    check_timing_tables()
    check_figures()
    print("\nPaper reproducibility artifact check completed.")


if __name__ == "__main__":
    main()
