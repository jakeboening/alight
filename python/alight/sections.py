"""Burn-time field sampled on section planes, for plots and the video."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alight.geometry import Grain
from alight.postprocess import TetField


@dataclass
class Sections:
    """Web-at-arrival on three planes. NaN outside the case, 0 in the initial port."""
    x: np.ndarray            # transverse coordinate, in
    z: np.ndarray            # axial coordinate, in
    z_plain: float           # station of the plain-port cross-section
    z_fin: float             # station of the finned cross-section
    plain: np.ndarray        # (len(x), len(x)), rows are y
    fin: np.ndarray
    longitudinal: np.ndarray  # (len(x), len(z)), the plane y = 0 through a fin

    def save(self, path) -> None:
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path) -> "Sections":
        data = np.load(path)
        return cls(**{k: (float(data[k]) if data[k].ndim == 0 else data[k]) for k in data.files})


def sample(grain: Grain, points, tets, u, n=420, nz=1500) -> Sections:
    """Evaluate a solved wedge field on full cross-sections and a longitudinal plane."""
    field = TetField(points, tets, u)
    radius = grain.case_radius
    x = np.linspace(-radius, radius, n)

    def evaluate(xx, yy, zz):
        xw, yw = grain.fold_to_wedge(xx, yy)
        values = field(np.column_stack([xw.ravel(), yw.ravel(), np.broadcast_to(zz, xx.shape).ravel()]))
        values = np.maximum(values.reshape(xx.shape), 0.0)
        values[grain.distance(xx, yy, zz) <= 0] = 0.0
        values[np.hypot(xx, yy) > radius] = np.nan
        return values

    xx, yy = np.meshgrid(x, x)
    z_fin = grain.fin_start + grain.fin_length / 2 if grain.n_fins else grain.length / 2
    z_plain = grain.fin_start / 2 if grain.n_fins and grain.fin_start > 0 else grain.length / 2
    z = np.linspace(0.0, grain.length, nz)
    zz, xl = np.meshgrid(z, x)
    return Sections(x=x, z=z, z_plain=z_plain, z_fin=z_fin,
                    plain=evaluate(xx, yy, z_plain), fin=evaluate(xx, yy, z_fin),
                    longitudinal=evaluate(xl, np.zeros_like(xl), zz))


def cached(grain: Grain, solution, path) -> Sections:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        sample(grain, solution.points, solution.tets, solution.u).save(path)
    return Sections.load(path)
