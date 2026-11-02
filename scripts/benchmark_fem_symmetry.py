"""Fair FEM cost baseline for the symmetric benchmark (reviewer point 2).

The original benchmark (Case A) is symmetric about x=Lx/2, y=Ly/2 and x=y, so a
competent FEM user would solve a reduced model. This validates that quarter and
1/8-size solves reproduce the full solution, then times full vs reduced at the
same element size. Hardware is recorded with the result.

  full      : whole plate
  quarter   : [0,Lx/2]x[0,Ly/2], symmetry planes ux=0 at x=Lx/2, uy=0 at y=Ly/2
  eighth    : quarter model whose linear system is further reduced by the x<->y
              diagonal symmetry (about 1/8 of the unknowns). Assembly is still the
              quarter mesh; the SOLVE is the 1/8-size system.
"""
from __future__ import annotations

import argparse, contextlib, io, json, platform, statistics, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "fea-workflow" / "solver"))
import fem_solver  # noqa: E402

NU = 0.3


def random_cases(n_cases, seed=20260430):
    """Same random interior (E1..E3, t1..t3) cases as the accuracy evaluation."""
    rng = np.random.default_rng(seed)
    lo, hi = [1.0] * 3 + [0.02] * 3, [10.0] * 3 + [0.10] * 3
    return [tuple(float(v) for v in rng.uniform(lo, hi)) for _ in range(n_cases)]


def cfg_full(n, nz, case):
    e, t = list(case[:3]), list(case[3:])
    return {"geometry": {"Lx": 1.0, "Ly": 1.0, "H": sum(t), "ne_x": n, "ne_y": n, "ne_z": nz},
            "material": {"nu": NU, "E_layers": e, "t_layers": t},
            "load_patch": {"pressure": 1.0, "x_start": 1 / 3, "x_end": 2 / 3, "y_start": 1 / 3, "y_end": 2 / 3}}


def cfg_quarter(n, nz, diagonal, case):
    c = cfg_full(n // 2, nz, case)
    c["geometry"].update(Lx=0.5, Ly=0.5)
    # patch is [1/3,2/3] of the full plate = [2/3, 4/3] of the quarter extent
    c["load_patch"].update(x_start=2 / 3, x_end=4 / 3, y_start=2 / 3, y_end=4 / 3)
    c["boundary"] = {"x0": "clamped", "y0": "clamped", "x1": "sym_ux", "y1": "sym_uy"}
    if diagonal:
        c["diagonal_symmetry"] = True
    return c


def solve(cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        t0 = time.perf_counter()
        out = fem_solver.solve_three_layer_fem(cfg)
        return out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meshes", type=int, nargs="+", default=[16, 24, 32], help="in-plane elements per side (full plate)")
    ap.add_argument("--nz", type=int, default=8)
    ap.add_argument("--n-cases", type=int, default=20, help="random layer configurations to average over")
    ap.add_argument("--out", default=str(REPO / "graphs" / "data" / "fem_symmetry_benchmark.json"))
    args = ap.parse_args()

    cases = random_cases(args.n_cases)
    rows = []
    for n in args.meshes:
        nz = args.nz
        h = n // 2 + 1
        errs_q, errs_e, tf, tq, te = [], [], [], [], []
        for case in cases:
            (_, _, _, u_full), t_f = solve(cfg_full(n, nz, case))
            (_, _, _, u_q), t_q = solve(cfg_quarter(n, nz, False, case))
            (_, _, _, u_e), t_e = solve(cfg_quarter(n, nz, True, case))
            scale = np.max(np.abs(u_full))
            errs_q.append(float(np.max(np.abs(u_full[:h, :h] - u_q)) / scale))
            errs_e.append(float(np.max(np.abs(u_full[:h, :h] - u_e)) / scale))
            tf.append(t_f); tq.append(t_q); te.append(t_e)
        dofs = lambda a, b, c: 3 * (a + 1) * (b + 1) * (c + 1)
        mf, mq, me = (statistics.mean(v) for v in (tf, tq, te))
        rows.append({
            "mesh_full": [n, n, nz], "dofs_full": dofs(n, n, nz), "dofs_quarter": dofs(n // 2, n // 2, nz),
            "n_cases": len(cases),
            "max_rel_err_quarter_vs_full": max(errs_q), "max_rel_err_eighth_vs_full": max(errs_e),
            "t_full_s": mf, "t_quarter_s": mq, "t_eighth_s": me,
            "speedup_quarter": mf / mq, "speedup_eighth": mf / me,
        })
        r = rows[-1]
        print(f"{n}x{n}x{nz} ({len(cases)} cases): max err quarter {max(errs_q):.1e} eighth {max(errs_e):.1e} | "
              f"full {mf:.3f}s quarter {mq:.3f}s ({r['speedup_quarter']:.1f}x) eighth {me:.3f}s ({r['speedup_eighth']:.1f}x)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"hardware": {"machine": platform.machine(), "platform": platform.platform(),
                                                         "processor": platform.processor(), "python": platform.python_version()},
                                          "n_cases": args.n_cases, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
