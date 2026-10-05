"""Independent burnback reference on a Cartesian grid.

The initial port is voxelised from the grain definition, scikit-fmm marches the
distance from it, and scikit-image extracts the burning surfaces.
"""
from __future__ import annotations

import numpy as np

from alight.geometry import Grain
from alight.postprocess import BurnbackTable, make_table


def grid(grain: Grain, dx: float):
    """Node coordinates. Even fin counts are mirror-symmetric in x and y, so a quadrant is enough."""
    quadrant = grain.n_fins % 2 == 0
    n = int(np.ceil(grain.case_radius / dx)) + 2
    x = np.arange(0 if quadrant else -n, n + 1) * dx
    nz = int(round(grain.length / dx))
    z = np.linspace(0.0, grain.length, nz + 1)
    return x, x.copy(), z, (4 if quadrant else 1)


def distance_field(grain: Grain, dx: float, method: str = "fmm"):
    """Web distance on the grid: `fmm` (scikit-fmm) or `exact` (closed-form distance)."""
    x, y, z, mult = grid(grain, dx)
    phi = np.empty((len(x), len(y), len(z)))
    xx, yy = np.meshgrid(x, y, indexing="ij")
    for k, zk in enumerate(z):  # slab by slab to bound memory
        phi[:, :, k] = grain.distance(xx, yy, zk)
    if method == "fmm":
        import skfmm
        phi = skfmm.distance(phi, dx=[x[1] - x[0], y[1] - y[0], z[1] - z[0]])
    elif method != "exact":
        raise ValueError(method)
    return (x, y, z), phi, mult


def reference_table(grain: Grain, dx: float, method: str = "fmm", n_levels: int = 120) -> BurnbackTable:
    """Burn area vs web from marching-cubes iso-surfaces of the grid distance field."""
    from skimage.measure import marching_cubes, mesh_surface_area

    (x, y, z), phi, mult = distance_field(grain, dx, method)
    spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    inside = np.hypot(xx, yy) < grain.case_radius
    web_max = float(phi[inside].max())
    mask = np.broadcast_to((np.hypot(xx, yy) < grain.case_radius + 2 * dx)[:, :, None], phi.shape)
    # an extra level just short of burnout keeps the final drop in area sharp
    web = np.linspace(0.0, web_max, n_levels + 1)
    web = np.concatenate([web[:-1], [web_max - 0.02 * dx, web_max]])
    area = np.zeros_like(web)
    for n, w in enumerate(web[:-1]):
        level = max(w, 1e-6)
        verts, faces, _, _ = marching_cubes(phi, level, spacing=spacing, mask=mask)
        verts += (x[0], y[0], z[0])
        centroid = verts[faces].mean(axis=1)
        keep = np.hypot(centroid[:, 0], centroid[:, 1]) < grain.case_radius
        area[n] = mesh_surface_area(verts, faces[keep])
    return make_table(web, area * mult, grain.initial_port_volume, grain.propellant_volume)


def cached_reference(grain: Grain, dx: float, method: str, cachedir, n_levels: int = 150) -> BurnbackTable:
    from pathlib import Path

    path = Path(cachedir) / f"{method}_dx{dx:g}.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        reference_table(grain, dx, method, n_levels).save(path)
    return BurnbackTable.load(path)
