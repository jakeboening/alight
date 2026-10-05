#!/usr/bin/env python3
"""Mesh sensitivity study for the baseline finocyl grain.

Runs alight on a series of element sizes, Richardson-extrapolates
neighbouring pairs, computes an independent fast-marching reference, and writes
    results/sensitivity.csv       one row per mesh level / pair / reference grid
    results/solver_settings.csv   effect of CFL, tolerance and diffusive weight
    results/tables/*.csv          burn-area tables
    results/design.json           final table choice and nozzle

Usage: python run_sensitivity.py [--levels 0.4 0.3 0.2 0.15 0.1 0.075 0.05] [--jobs 3]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from alight import sensitivity as sens
from alight.ballistics import AP_HTPB_AL, IN, PSI, size_throat
from alight.fmm_check import cached_reference
from alight.geometry import FinocylGrain
from alight.postprocess import extrapolate_tables
from alight.solve import solve, work_dir

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results"
WORK = work_dir(HERE / "_work")
GRAIN = FinocylGrain()
PEAK_PRESSURE = 1000 * PSI
REFERENCE_GRIDS = (0.1, 0.07, 0.05, 0.035)
SETTINGS_LEVEL = 0.2
WEIGHT_PAIR = (0.2, 0.15)       # mesh pair for the diffusive-weight check
WEIGHTS = (0.75, 1.0, 2.0)
SETTINGS = (  # cfl, diffusive weight, tolerance
    (1.0, 1.0, 1e-4), (0.5, 1.0, 1e-4), (2.0, 1.0, 1e-4), (1.0, 1.0, 1e-6),
    (0.5, 0.75, 1e-4), (1.0, 2.0, 1e-4), (0.5, 0.5, 1e-4),
)


def _level(size):
    solution = sens.run_level(GRAIN, WORK, size, threads=4)
    sens.cached_table(solution, "table")
    return size, solution.meta


def _reference(args):
    dx, method = args
    cached_reference(GRAIN, dx, method, WORK / "ref")
    return args


def _weight_pair(weight):
    """Extrapolated result from a mesh pair solved with a non-default diffusive weight."""
    tables = [solve(GRAIN, sens.level_dir(WORK, h), cfl=0.5, weight=weight, tag=f"cfl0.5_w{weight:g}_tol0.0001").table()
              for h in WEIGHT_PAIR]
    return weight, extrapolate_tables(tables[0], tables[1], WEIGHT_PAIR[0] / WEIGHT_PAIR[1])


def _setting(args):
    cfl, weight, tol = args
    try:
        solution = solve(GRAIN, sens.level_dir(WORK, SETTINGS_LEVEL), cfl=cfl, weight=weight, tol=tol,
                         tag=f"cfl{cfl:g}_w{weight:g}_tol{tol:g}")
    except RuntimeError:
        return args, None, None
    return args, solution.meta, solution.table()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=float, nargs="+", default=list(sens.LEVELS))
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()
    levels = sorted(args.levels, reverse=True)
    (OUTPUT / "tables").mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(args.jobs) as pool:
        metas = dict(pool.map(_level, levels))
        list(pool.map(_reference, [(dx, "fmm") for dx in REFERENCE_GRIDS] + [(REFERENCE_GRIDS[-1], "exact")]))
        settings = list(pool.map(_setting, SETTINGS))
        weight_pairs = list(pool.map(_weight_pair, WEIGHTS))

    raw = {h: sens.cached_table(sens.run_level(GRAIN, WORK, h), "table") for h in levels}
    pairs = list(zip(levels[:-1], levels[1:]))
    extrapolated = {(c, f): extrapolate_tables(raw[c], raw[f], c / f) for c, f in pairs}
    references = {(dx, "fmm"): cached_reference(GRAIN, dx, "fmm", WORK / "ref") for dx in REFERENCE_GRIDS}
    references[(REFERENCE_GRIDS[-1], "exact")] = cached_reference(GRAIN, REFERENCE_GRIDS[-1], "exact", WORK / "ref")
    reference = references[(REFERENCE_GRIDS[-1], "fmm")]

    final_pair = pairs[-1]
    final = extrapolated[final_pair]
    nozzle = size_throat(final, AP_HTPB_AL, PEAK_PRESSURE)

    rows = []
    for h in levels:
        raw[h].save(OUTPUT / "tables" / f"raw_h{h:g}.csv")
        rows.append({"kind": "raw", "size_in": h, "label": f"h = {h:g}", "nodes": metas[h]["nodes"],
                     "tetrahedra": metas[h]["tetrahedra"], "iterations": metas[h]["iterations"],
                     "solve_seconds": metas[h]["seconds"],
                     **sens.metrics(raw[h], GRAIN, AP_HTPB_AL, nozzle, reference)})
    for (c, f), table in extrapolated.items():
        table.save(OUTPUT / "tables" / f"extrapolated_h{c:g}_h{f:g}.csv")
        rows.append({"kind": "extrapolated", "size_in": f, "size_coarse_in": c, "label": f"{c:g} / {f:g}",
                     **sens.metrics(table, GRAIN, AP_HTPB_AL, nozzle, reference)})
    for (dx, method), table in references.items():
        table.save(OUTPUT / "tables" / f"reference_{method}_dx{dx:g}.csv")
        rows.append({"kind": f"reference-{method}", "size_in": dx, "label": f"{method} dx = {dx:g}",
                     **sens.metrics(table, GRAIN, AP_HTPB_AL, nozzle, reference)})
    sens.write_csv(rows, OUTPUT / "sensitivity.csv")

    setting_rows = []
    for (cfl, weight, tol), meta, table in settings:
        row = {"size_in": SETTINGS_LEVEL, "cfl": cfl, "diffusive_weight": weight, "tolerance": tol,
               "status": "diverged" if meta is None else "converged"}
        if meta is not None:
            row.update(iterations=meta["iterations"], **sens.metrics(table, GRAIN, AP_HTPB_AL, nozzle, reference))
        setting_rows.append(row)
    for weight, table in weight_pairs:
        setting_rows.append({"size_in": WEIGHT_PAIR[1], "size_coarse_in": WEIGHT_PAIR[0], "cfl": 0.5,
                             "diffusive_weight": weight, "tolerance": 1e-4, "status": "extrapolated pair",
                             **sens.metrics(table, GRAIN, AP_HTPB_AL, nozzle, reference)})
    sens.write_csv(setting_rows, OUTPUT / "solver_settings.csv")

    # observed order of the raw solver: slope of log|error| against log(element size)
    final_row = next(r for r in rows if r["kind"] == "extrapolated" and r["size_in"] == final_pair[1]
                     and r["size_coarse_in"] == final_pair[0])
    raw_rows = [r for r in rows if r["kind"] == "raw"]
    order = {key: float(np.polyfit(np.log([r["size_in"] for r in raw_rows]),
                                   np.log([abs(r[key] - final_row[key]) for r in raw_rows]), 1)[0])
             for key in ("peak_pressure_psi", "burn_time_s", "web_burnout_in")}

    (OUTPUT / "design.json").write_text(json.dumps({
        "final_table": f"tables/extrapolated_h{final_pair[0]:g}_h{final_pair[1]:g}.csv",
        "final_pair_in": final_pair,
        "reference_table": f"tables/reference_fmm_dx{REFERENCE_GRIDS[-1]:g}.csv",
        "propellant": AP_HTPB_AL.name,
        "throat_diameter_in": nozzle.throat_diameter / IN,
        "expansion_ratio": nozzle.expansion_ratio,
        "design_peak_pressure_psi": PEAK_PRESSURE / PSI,
        "observed_order_raw": order,
    }, indent=2))

    keys = ("peak_pressure_psi", "burn_time_s", "total_impulse_lbf_s", "web_burnout_in", "area_rms_vs_ref_pct")
    print(f"{'case':22s}" + "".join(f"{k:>22s}" for k in keys))
    for row in rows:
        print(f"{row['kind'] + ' ' + row['label']:22s}" + "".join(f"{row[k]:22.4f}" for k in keys))


if __name__ == "__main__":
    main()
