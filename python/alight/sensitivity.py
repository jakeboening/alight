"""Mesh sensitivity of the alight solution and its Richardson extrapolation.

The burnback-3d scheme that alight runs carries a numerical-diffusion error that is first order in
the element size. `extrapolate` removes the leading term from two solutions on
different meshes; the study shows both the raw and the extrapolated convergence
against an independent fast-marching reference.
"""
from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from alight.ballistics import PSI, LBF, Nozzle, Propellant, simulate
from alight.geometry import Grain
from alight.postprocess import BurnbackTable, TetField
from alight.solve import Solution, run_case

LEVELS = (0.4, 0.3, 0.2, 0.15, 0.1, 0.075, 0.05)


def level_dir(workroot, size) -> Path:
    return Path(workroot) / f"h{size:g}"


def run_level(grain: Grain, workroot, size, **kw) -> Solution:
    return run_case(grain, level_dir(workroot, size), size, caddir=Path(workroot) / "cad", **kw)


def extrapolate(coarse: Solution, fine: Solution, size_coarse, size_fine, order=1.0) -> Solution:
    """Richardson-extrapolated burn-time field on the fine mesh."""
    ratio = (size_coarse / size_fine) ** order
    cache = fine.workdir / f"extrapolated_from_h{size_coarse:g}.npy"
    if cache.exists():
        u = np.load(cache)
    else:
        u_coarse = TetField(coarse.points, coarse.tets, coarse.u)(fine.points)
        u = np.maximum((ratio * fine.u - u_coarse) / (ratio - 1), 0.0)
        u[fine.u == 0.0] = 0.0   # inlet nodes stay on the initial surface
        np.save(cache, u)
    return replace(fine, u=u)


def cached_table(solution: Solution, name: str, n_levels=400) -> BurnbackTable:
    path = solution.workdir / f"{name}.csv"
    if not path.exists():
        solution.table(n_levels).save(path)
    return BurnbackTable.load(path)


def curve_difference(table: BurnbackTable, reference: BurnbackTable, web_limit: float):
    """RMS and maximum burn-area difference over 0..web_limit, as % of the reference mean area."""
    web = np.linspace(0.0, web_limit, 600)
    ref = reference.area_at(web)
    diff = table.area_at(web) - ref
    return 100 * np.sqrt(np.mean(diff**2)) / ref.mean(), 100 * np.abs(diff).max() / ref.mean()


def metrics(table: BurnbackTable, grain: Grain, propellant: Propellant, nozzle: Nozzle,
            reference: BurnbackTable) -> dict:
    result = simulate(table, propellant, nozzle)
    rms, worst = curve_difference(table, reference, 0.95 * (grain.case_radius - grain.port_radius))
    return {
        "initial_area_in2": table.burn_area[0],
        "peak_area_in2": table.burn_area.max(),
        "web_at_peak_in": table.web[table.burn_area.argmax()],
        "web_burnout_in": table.web_burnout,
        "burned_volume_in3": table.burned_volume[-1],
        "volume_error_pct": 100 * (table.burned_volume[-1] / grain.propellant_volume - 1),
        "area_rms_vs_ref_pct": rms,
        "area_max_vs_ref_pct": worst,
        "peak_pressure_psi": result.peak_pressure / PSI,
        "burn_time_s": result.burn_time,
        "total_impulse_lbf_s": result.total_impulse / LBF,
    }


def write_csv(rows, path) -> None:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})


def read_csv(path) -> list:
    def parse(value):
        try:
            return float(value)
        except ValueError:
            return value
    with open(path) as f:
        return [{k: parse(v) for k, v in row.items() if v != ""} for row in csv.DictReader(f)]
