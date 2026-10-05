#!/usr/bin/env python3
"""Validation against closed-form burn area for three grain families, each swept over its geometry.

    tube   hollow cylinder, ends inhibited          A = 2 pi (r + w) L
    bates  hollow cylinder, both ends burning       A = 2 pi (r + w)(L - 2w) + 2 pi (R^2 - (r + w)^2)
    star   classical star, ends inhibited           phase I / II perimeter formulas (alight.geometry.StarGrain)

Every case runs the full pipeline (CAD, mesh, alight on two element sizes, extrapolation) and is
compared with the formula from ignition to 95 % of the web. The last 5 % is left out because the
burn area is discontinuous at burnout, which any mesh smears over about one element; the burnout
web itself is compared instead. Writes results/kpis.csv and one figure per family (PNG here, PDF
in report/fig).

Usage: python run_validation.py [--jobs 6]
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from alight import style
from alight.geometry import StarGrain, TubeGrain
from alight.postprocess import extrapolate_tables
from alight.solve import run_case, work_dir

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE.parents[1] / "report" / "fig"
WORK = work_dir(HERE / "_work")

STAR = dict(length=0.5)
CASES = {
    "tube": [(f"port / case = {d / 10:g}", TubeGrain(port_diameter=d, length=0.5)) for d in (2.0, 4.0, 6.0, 8.0)],
    "bates": [(f"L / D = {length / 10:g}", TubeGrain(port_diameter=4.0, length=length, ends_inhibited=False))
              for length in (5.0, 10.0, 15.0)],
    "star": [
        ("N = 5", StarGrain(n_points=5, **STAR)),
        ("N = 6 (baseline)", StarGrain(**STAR)),
        ("N = 7", StarGrain(n_points=7, **STAR)),
        ("N = 8", StarGrain(n_points=8, **STAR)),
        ("angular fraction 0.5", StarGrain(angular_fraction=0.5, **STAR)),
        ("angular fraction 0.9", StarGrain(angular_fraction=0.9, **STAR)),
        ("point angle 60°", StarGrain(point_angle=60.0, **STAR)),
        ("point angle 100°", StarGrain(point_angle=100.0, **STAR)),
    ],
}
# coarse and fine element size, in. The 2-D families are thin slices, so they afford finer elements.
SIZES = {"tube": (0.05, 0.025), "bates": (0.1, 0.05), "star": (0.05, 0.025)}
COMPARED = 0.95   # fraction of the web over which the burn area is compared
TITLES = {"tube": "Tube, ends inhibited", "bates": "BATES, both ends burning", "star": "Star, ends inhibited"}


def solve_case(job):
    family, index, label, grain, sizes = job
    root = WORK / f"{family}_{index:02d}"
    raw = [run_case(grain, root / f"h{size:g}", size, caddir=root / "cad", threads=2) for size in sizes]
    tables = [solution.table() for solution in raw]
    final = extrapolate_tables(tables[0], tables[1], sizes[0] / sizes[1])
    web = np.linspace(0.0, COMPARED * grain.web, 191)
    exact = grain.burn_area(web)
    error = lambda table: 100 * (table.area_at(web) / exact - 1)
    return {
        "family": family, "label": label, "web": web, "exact": exact,
        "area": final.area_at(web), "error": error(final), "error_fine_mesh": error(tables[1]),
        "kpi": {
            "family": family, "case": label, "tetrahedra_fine": raw[1].meta["tetrahedra"],
            "element_sizes_in": f"{sizes[0]:g} / {sizes[1]:g}",
            # the star's web is where its arcs reach the case; slivers burn on after that
            "burnout_web_error_pct": 100 * (final.web_burnout / grain.web - 1) if family != "star" else float("nan"),
            "web_in": grain.web, "max_error_pct": float(np.abs(error(final)).max()),
            "rms_error_pct": float(np.sqrt(np.mean(error(final)**2))),
            "rms_error_fine_mesh_pct": float(np.sqrt(np.mean(error(tables[1])**2))),
            "initial_area_error_pct": 100 * (final.burn_area[0] / grain.initial_burn_area - 1),
            "volume_error_pct": 100 * (final.burned_volume[-1] / grain.propellant_volume - 1),
        },
    }


def plot_family(family, cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style.apply(9.5)
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.5), dpi=150, gridspec_kw=dict(wspace=0.28))
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(cases)))
    for case, color in zip(cases, colors):
        fraction = case["web"] / case["web"][-1] * COMPARED
        left.plot(fraction, case["exact"], color=color, lw=1.3)
        left.plot(fraction[::12], case["area"][::12], "o", color=color, ms=3.6, mfc="white", mew=1.0, label=case["label"])
        right.plot(fraction, case["error"], color=color, lw=1.3)
    left.plot([], [], color="0.3", lw=1.3, label="closed form (line)")
    left.set_xlabel("web burned / web")
    left.set_ylabel("burn area (in²)")
    left.set_title(f"{TITLES[family]}: alight (markers) and closed form")
    left.legend(fontsize=7.5, frameon=False, ncol=2)
    left.grid(True, alpha=0.25)
    right.axhline(0, color="k", lw=0.6)
    right.set_xlabel("web burned / web")
    right.set_ylabel("burn area error (%)")
    right.set_title("Difference from the closed form")
    limit = max(1.0, 1.15 * max(np.abs(case["error"]).max() for case in cases))
    right.set_ylim(-limit, limit)
    right.grid(True, alpha=0.25)
    for folder, suffix in ((RESULTS, ".png"), (FIGURES, ".pdf")):
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"validation_{family}{suffix}", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    jobs = [(family, i, label, grain, SIZES[family])
            for family, cases in CASES.items() for i, (label, grain) in enumerate(cases)]
    with ProcessPoolExecutor(args.jobs) as pool:
        done = list(pool.map(solve_case, jobs))
    for family in CASES:
        plot_family(family, [case for case in done if case["family"] == family])
    RESULTS.mkdir(exist_ok=True)
    kpis = [case["kpi"] for case in done]
    with open(RESULTS / "kpis.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(kpis[0]))
        writer.writeheader()
        for row in kpis:
            writer.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"{'family':7s}{'case':24s}{'tets':>10s}{'max %':>9s}{'rms %':>9s}{'rms fine mesh %':>17s}{'burnout %':>11s}{'volume %':>10s}")
    for k in kpis:
        print(f"{k['family']:7s}{k['case']:24s}{k['tetrahedra_fine']:10d}{k['max_error_pct']:9.3f}{k['rms_error_pct']:9.3f}"
              f"{k['rms_error_fine_mesh_pct']:17.3f}{k['burnout_web_error_pct']:11.3f}{k['volume_error_pct']:10.3f}")


if __name__ == "__main__":
    main()
