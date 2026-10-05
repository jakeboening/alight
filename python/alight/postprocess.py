"""Burn area vs web from a burn-time field on a tetrahedral mesh."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def iso_area(points, tets, u, levels, bins=None):
    """Area of the iso-surfaces u = level of a piecewise-linear field (marching tetrahedra).

    With `bins = (bin index of every tetrahedron, number of bins)` the area is returned per bin,
    as an array (levels, bins); otherwise as one total per level.
    """
    ut = u[tets]
    order = np.argsort(ut, axis=1)
    us = np.take_along_axis(ut, order, axis=1)
    ts = np.take_along_axis(tets, order, axis=1)
    areas = np.zeros(len(levels)) if bins is None else np.zeros((len(levels), bins[1]))
    for n, w in enumerate(levels):
        active = np.flatnonzero((us[:, 0] < w) & (us[:, 3] > w))
        if not active.size:
            continue
        a = us[active]
        p = points[ts[active]]
        below = (a < w).sum(axis=1)
        piece = np.zeros(len(active))

        def cut(rows, i, j):
            t = (w - a[rows, i]) / (a[rows, j] - a[rows, i])
            return p[rows, i] + t[:, None] * (p[rows, j] - p[rows, i])

        rows = np.flatnonzero(below == 1)
        if rows.size:
            q0, q1, q2 = cut(rows, 0, 1), cut(rows, 0, 2), cut(rows, 0, 3)
            piece[rows] = 0.5 * np.linalg.norm(np.cross(q1 - q0, q2 - q0), axis=1)
        rows = np.flatnonzero(below == 3)
        if rows.size:
            q0, q1, q2 = cut(rows, 0, 3), cut(rows, 1, 3), cut(rows, 2, 3)
            piece[rows] = 0.5 * np.linalg.norm(np.cross(q1 - q0, q2 - q0), axis=1)
        rows = np.flatnonzero(below == 2)
        if rows.size:
            # planar quadrilateral: half the cross product of its diagonals
            q0, q1, q2, q3 = cut(rows, 0, 2), cut(rows, 0, 3), cut(rows, 1, 3), cut(rows, 1, 2)
            piece[rows] = 0.5 * np.linalg.norm(np.cross(q2 - q0, q3 - q1), axis=1)
        if bins is None:
            areas[n] = piece.sum()
        else:
            areas[n] = np.bincount(bins[0][active], weights=piece, minlength=bins[1])
    return areas


def triangle_area(points, triangles):
    p = points[triangles]
    return 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)


def tet_volume(points, tets):
    p = points[tets]
    return np.abs(np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])) / 6


@dataclass
class BurnbackTable:
    """Full-grain burn geometry against web burned. Inches."""
    web: np.ndarray
    burn_area: np.ndarray
    burned_volume: np.ndarray
    initial_port_volume: float
    propellant_volume: float

    @property
    def web_burnout(self) -> float:
        return float(self.web[-1])

    @property
    def free_volume(self) -> np.ndarray:
        return self.initial_port_volume + self.burned_volume

    def area_at(self, web):
        return np.interp(web, self.web, self.burn_area, right=0.0)

    def save(self, path) -> None:
        header = f"initial_port_volume_in3={float(self.initial_port_volume)!r} propellant_volume_in3={float(self.propellant_volume)!r}\n" \
                 "web_in,burn_area_in2,burned_volume_in3"
        np.savetxt(path, np.column_stack([self.web, self.burn_area, self.burned_volume]),
                   delimiter=",", header=header, comments="# ")

    @classmethod
    def load(cls, path) -> "BurnbackTable":
        first = Path(path).read_text().splitlines()[0].lstrip("# ")
        meta = dict(item.split("=") for item in first.split())
        data = np.loadtxt(path, delimiter=",", comments="#")
        return cls(data[:, 0], data[:, 1], data[:, 2], float(meta["initial_port_volume_in3"]),
                   float(meta["propellant_volume_in3"]))


def make_table(web, burn_area, initial_port_volume, propellant_volume) -> BurnbackTable:
    """Assemble a table; the burned volume is the integral of area over web."""
    web = np.asarray(web, float)
    burn_area = np.asarray(burn_area, float)
    burned = np.concatenate([[0.0], np.cumsum(0.5 * (burn_area[1:] + burn_area[:-1]) * np.diff(web))])
    return BurnbackTable(web, burn_area, burned, initial_port_volume, propellant_volume)


def table_from_volume(web, volume, initial_area, initial_port_volume) -> BurnbackTable:
    """Table from burned volume sampled on a uniform web grid with an even number of intervals.

    The burn area is dV/dw, evaluated at the odd grid points by central difference.
    """
    step = web[1] - web[0]
    area = (volume[2::2] - volume[:-2:2]) / (2 * step)
    return BurnbackTable(np.concatenate([[0.0], web[1::2], [web[-1]]]),
                         np.concatenate([[initial_area], area, [0.0]]),
                         np.concatenate([[0.0], volume[1::2], [volume[-1]]]),
                         initial_port_volume, float(volume[-1]))


def table_from_solution(points, tets, u, inlet_area, multiplicity, initial_port_volume, n_levels=400):
    """Burn table from one solved sector: area of the burn-time iso-surfaces.

    points, u in inches; `inlet_area` is the sector's initial burning surface and
    `multiplicity` the number of sectors in the full grain.
    """
    web = np.linspace(0.0, float(u.max()), n_levels + 1)
    area = iso_area(points, tets, u, web)
    area[0] = inlet_area
    area[-1] = 0.0
    return make_table(web, area * multiplicity, initial_port_volume,
                      tet_volume(points, tets).sum() * multiplicity)


def extrapolate_tables(coarse: BurnbackTable, fine: BurnbackTable, size_ratio: float,
                       order: float = 1.0, n_levels: int = 400) -> BurnbackTable:
    """Richardson extrapolation of two burn tables to zero element size.

    The solver's error is a mesh-proportional lag in web at a given fraction of
    propellant burned, so the extrapolation is applied to web(fraction burned),
    not to area(web): that keeps the fin-burnout kink sharp instead of smearing
    it. The result is scaled to the meshed propellant volume.
    """
    from scipy.interpolate import PchipInterpolator

    fraction = np.linspace(0.0, 1.0, 4001)

    def web_of_fraction(table):
        volume, index = np.unique(table.burned_volume, return_index=True)
        return PchipInterpolator(volume / volume[-1], table.web[index])(fraction)

    r = size_ratio ** order
    web = np.maximum.accumulate((r * web_of_fraction(fine) - web_of_fraction(coarse)) / (r - 1))
    web, index = np.unique(web, return_index=True)
    uniform = np.linspace(0.0, web[-1], 2 * n_levels + 1)
    volume = PchipInterpolator(web, fraction[index])(uniform) * fine.propellant_volume
    table = table_from_volume(uniform, volume, fine.burn_area[0], fine.initial_port_volume)
    # Where the two meshes put a kink of the curve at slightly different fractions, the derivative can
    # show a one- or two-point spike. Replace points that stand more than 2 % of the peak area away from
    # their local median; genuine features are wider than that and are left alone.
    from scipy.signal import medfilt

    area = table.burn_area
    median = medfilt(area, 7)
    spikes = np.abs(area - median) > 0.02 * area.max()
    spikes[[0, 1, 2, -3, -2, -1]] = False
    area[spikes] = median[spikes]
    return table


class TetField:
    """Piecewise-linear field on a tetrahedral mesh, evaluated at arbitrary points."""

    def __init__(self, points, tets, u):
        from scipy.spatial import cKDTree

        self.u = u[tets]
        p = points[tets]
        self.origin = p[:, 0]
        self.inverse = np.linalg.inv(np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=2))
        self.tree = cKDTree(p.mean(axis=1))

    def __call__(self, x, k=16, chunk=200_000):
        x = np.atleast_2d(np.asarray(x, float))
        out = np.empty(len(x))
        for start in range(0, len(x), chunk):
            q = x[start:start + chunk]
            _, cand = self.tree.query(q, k=k, workers=-1)
            lam = np.einsum("nkij,nkj->nki", self.inverse[cand], q[:, None, :] - self.origin[cand])
            bary = np.concatenate([1 - lam.sum(axis=2, keepdims=True), lam], axis=2)
            # the containing tet has all barycentric coordinates >= 0; otherwise take the closest
            best = bary.min(axis=2).argmax(axis=1)
            rows = np.arange(len(q))
            out[start:start + chunk] = (bary[rows, best] * self.u[cand[rows, best]]).sum(axis=1)
        return out
