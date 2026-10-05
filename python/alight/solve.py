"""Run one burnback case end to end: CAD -> mesh -> alight -> burn table."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alight.geometry import Grain
from alight.postprocess import BurnbackTable, extrapolate_tables, table_from_solution, triangle_area

MM_PER_IN = 25.4
ALIGHT_URL = "https://github.com/jakeboening/alight"


def work_dir(default) -> Path:
    """Folder for meshes and solver output: $ALIGHT_WORK if set (they are large), else `default`."""
    return Path(os.environ.get("ALIGHT_WORK", default))


def alight_executable() -> str:
    """Path of the alight solver: $ALIGHT, else the PATH, else a build in this checkout."""
    exe = os.environ.get("ALIGHT") or shutil.which("alight")
    if not exe:
        root = Path(__file__).resolve().parents[2]
        builds = [root / "build" / name for name in ("alight", "alight.exe", "Release/alight.exe")]
        exe = next((str(path) for path in builds if path.exists()), None)
    if not exe:
        raise RuntimeError("alight not found. Build it (cmake -S . -B build && cmake --build build --config Release) "
                           f"or download a release from {ALIGHT_URL}/releases, then put it on the PATH "
                           "or set the ALIGHT environment variable to the executable.")
    return exe


def alight_version() -> str:
    return subprocess.run([alight_executable(), "--version"], capture_output=True, text=True).stdout.strip()


def _run_module(module, *args):
    done = subprocess.run([sys.executable, "-m", module, *[str(a) for a in args]], capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{module} failed:\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}")
    return done.stdout


@dataclass
class Solution:
    """Burn-time field of one solved sector, in inches."""
    grain: Grain
    points: np.ndarray
    tets: np.ndarray
    u: np.ndarray
    inlet_area: float
    multiplicity: float
    meta: dict
    workdir: Path
    initial_port_volume: float

    def table(self, n_levels=400) -> BurnbackTable:
        return table_from_solution(self.points, self.tets, self.u, self.inlet_area, self.multiplicity,
                                   self.initial_port_volume, n_levels)


def make_cad(grain: Grain, caddir) -> dict:
    caddir = Path(caddir)
    caddir.mkdir(parents=True, exist_ok=True)
    grain_json = caddir / "grain.json"
    try:
        current = "port_volume_in3" in json.loads((caddir / "cad.json").read_text()) \
            and Grain.from_json(grain_json) == grain
    except (OSError, KeyError, TypeError, ValueError):
        current = False
    if not current:
        grain.to_json(grain_json)
        _run_module("alight.cad", grain_json, caddir)
    return json.loads((caddir / "cad.json").read_text())


def make_mesh(grain: Grain, caddir, workdir, size, sector="wedge", size_inlet=None,
              grow=None, curvature=0, threads=8) -> dict:
    caddir, workdir = Path(caddir), Path(workdir)
    if not (workdir / "mesh.npz").exists():
        step = caddir / ("wedge.step" if sector == "wedge" else "grain.step")
        args = [caddir / "grain.json", step, workdir, "--size", size, "--sector", sector,
                "--curvature", curvature, "--threads", threads]
        if size_inlet:
            args += ["--size-inlet", size_inlet]
        if grow:
            args += ["--grow", grow]
        _run_module("alight.mesh", *args)
    return json.loads((workdir / "mesh.info.json").read_text())


def solve(grain: Grain, workdir, sector="wedge", cfl=1.0, weight=1.0, iters=20000,
          tol=1e-4, tag="run", caddir=None) -> Solution:
    """Run alight on the mesh in `workdir` until the burn-time field is steady."""
    workdir = Path(workdir)
    prefix = workdir / tag
    meta_path = Path(f"{prefix}.meta.json")
    if not meta_path.exists():
        subprocess.run([alight_executable(), str(workdir / "mesh.json"), str(prefix), "--cfl", str(cfl),
                        "--weight", str(weight), "--iters", str(iters), "--tol", str(tol)],
                       capture_output=True, text=True)
        if not meta_path.exists():
            raise RuntimeError(f"alight produced no result in {workdir}")
    meta = json.loads(meta_path.read_text())
    if meta["diverged"]:
        raise RuntimeError(f"alight diverged in {workdir} (cfl={cfl}, weight={weight})")
    if not meta["converged"]:
        raise RuntimeError(f"alight did not converge in {iters} iterations in {workdir}")
    mesh = np.load(workdir / "mesh.npz")
    points = mesh["points"] / MM_PER_IN
    u = np.fromfile(f"{prefix}.u.f64") / MM_PER_IN
    inlet_area = triangle_area(points, mesh["triangles"][mesh["inlet"]]).sum()
    multiplicity = grain.wedge_count if sector == "wedge" else 1.0
    cad = json.loads((Path(caddir) if caddir else workdir.parent / "cad").joinpath("cad.json").read_text())
    return Solution(grain, points, mesh["tets"], u, inlet_area, multiplicity, meta, workdir, cad["port_volume_in3"])


def run_case(grain: Grain, workdir, size, sector="wedge", size_inlet=None, grow=None,
             curvature=0, caddir=None, threads=8, **solver) -> Solution:
    workdir = Path(workdir)
    caddir = Path(caddir) if caddir else workdir.parent / "cad"
    make_cad(grain, caddir)
    make_mesh(grain, caddir, workdir, size, sector, size_inlet, grow, curvature, threads)
    return solve(grain, workdir, sector, caddir=caddir, **solver)


def burn_table(grain: Grain, workroot, sizes, n_levels=400, **kw) -> BurnbackTable:
    """Burn table extrapolated to zero element size from two meshes, `sizes = (coarse, fine)` in inches."""
    coarse, fine = sizes
    tables = [run_case(grain, Path(workroot) / f"h{size:g}", size, caddir=Path(workroot) / "cad", **kw).table(n_levels)
              for size in (coarse, fine)]
    return extrapolate_tables(tables[0], tables[1], coarse / fine, n_levels=n_levels)
