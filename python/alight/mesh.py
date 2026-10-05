"""Tetrahedral mesh of the grain STEP, written in the mesh format alight reads.

Run as a module (separate process from build123d):
    python -m alight.mesh grain.json input.step outdir --size 0.2 [--sector wedge|full]

Boundary types follow burnback-3d: `inlet` for the initial burning surface,
`outlet` for inhibited or case-bonded faces, `symmetry` for the wedge planes.
Sizes are given in inches; the model is in millimetres. Writes mesh.json (for
alight), mesh.npz (the same arrays for post-processing) and mesh.info.json.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from alight.geometry import Grain

MM_PER_IN = 25.4


def _samples(gmsh, tag, n=5):
    """Points on the (untrimmed) geometry under a surface."""
    (u0, v0), (u1, v1) = [list(b) for b in gmsh.model.getParametrizationBounds(2, tag)]
    uu, vv = np.meshgrid(np.linspace(u0, u1, n + 2)[1:-1], np.linspace(v0, v1, n + 2)[1:-1])
    uv = np.column_stack([uu.ravel(), vv.ravel()]).ravel()
    return np.asarray(gmsh.model.getValue(2, tag, uv)).reshape(-1, 3)


def classify(gmsh, grain: Grain, sector: str) -> dict:
    """Sort the model surfaces into inlet / outlet / symmetry planes."""
    s = MM_PER_IN
    tol = 1e-4 * grain.case_radius * s
    a = grain.wedge_angle
    groups = {"inlet": [], "outlet": [], "symmetry_0": [], "symmetry_1": []}
    for _, tag in gmsh.model.getEntities(2):
        p = _samples(gmsh, tag)
        r = np.hypot(p[:, 0], p[:, 1])
        on = lambda values: bool(np.all(np.abs(values) < tol))
        head, aft = grain.inhibited_ends
        on_inhibited_end = (head and on(p[:, 2])) or (aft and on(p[:, 2] - grain.length * s))
        if on(r - grain.case_radius * s) or on_inhibited_end:
            groups["outlet"].append(tag)
        elif sector == "wedge" and on(p[:, 1]):
            groups["symmetry_0"].append(tag)
        elif sector == "wedge" and on(-p[:, 0] * math.sin(a) + p[:, 1] * math.cos(a)):
            groups["symmetry_1"].append(tag)
        else:
            groups["inlet"].append(tag)
    return groups


BOUNDARY_TYPES = {"inlet": "inlet", "outlet": "outlet", "symmetry_0": "symmetry", "symmetry_1": "symmetry"}


def generate(grain: Grain, step, outdir, size, sector="wedge", size_inlet=None,
             grow=None, curvature=0, threads=8) -> dict:
    """Mesh the STEP solid. `size` and `size_inlet` are element sizes in inches.

    With `size_inlet`, elements are that size on the burning surface and grow to
    `size` over the distance `grow` (inches, default 4 * size).
    """
    import gmsh

    s = MM_PER_IN
    outdir = Path(outdir)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", threads)
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        gmsh.model.occ.importShapes(str(step))
        gmsh.model.occ.synchronize()
        volumes = [tag for _, tag in gmsh.model.getEntities(3)]
        if len(volumes) != 1:
            raise RuntimeError(f"expected one solid in {step}, found {len(volumes)}")
        box = gmsh.model.getBoundingBox(3, volumes[0])
        if abs(box[5] - box[2] - grain.length * s) > 1e-3 * grain.length * s:
            raise RuntimeError(f"STEP length {box[5] - box[2]:.3f} mm does not match the grain")

        groups = classify(gmsh, grain, sector)
        if not groups["inlet"] or not groups["outlet"]:
            raise RuntimeError(f"surface classification failed: {groups}")
        if sector == "wedge" and not (groups["symmetry_0"] and groups["symmetry_1"]):
            raise RuntimeError(f"symmetry planes not found: {groups}")
        tags = {key: gmsh.model.addPhysicalGroup(2, surfaces, name=key)
                for key, surfaces in groups.items() if surfaces}
        gmsh.model.addPhysicalGroup(3, volumes, name="domain")

        gmsh.option.setNumber("Mesh.MeshSizeMax", size * s)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature)
        if size_inlet:
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            distance = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(distance, "SurfacesList", groups["inlet"])
            gmsh.model.mesh.field.setNumber(distance, "Sampling", 200)
            threshold = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
            gmsh.model.mesh.field.setNumber(threshold, "SizeMin", size_inlet * s)
            gmsh.model.mesh.field.setNumber(threshold, "SizeMax", size * s)
            gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(threshold, "DistMax", (grow or 4 * size) * s)
            gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10 if threads > 1 else 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.model.mesh.generate(3)

        node_tags, xyz, _ = gmsh.model.mesh.getNodes()
        index = np.zeros(int(node_tags.max()) + 1, np.int64)
        index[node_tags.astype(np.int64)] = np.arange(len(node_tags))
        points = xyz.reshape(-1, 3)
        triangles, conditions = [], []
        for key, tag in tags.items():
            for surface in groups[key]:
                _, _, nodes = gmsh.model.mesh.getElements(2, surface)
                tri = index[np.asarray(nodes[0], np.int64)].reshape(-1, 3)
                triangles.append(tri)
                conditions.append(np.full(len(tri), tag))
        triangles, conditions = np.concatenate(triangles), np.concatenate(conditions)
        tets = index[np.asarray(gmsh.model.mesh.getElementsByType(4)[1], np.int64)].reshape(-1, 4)
    finally:
        gmsh.finalize()

    # alight's mesh format: 1-based connectivity, one boundary tag per triangle
    outdir.mkdir(parents=True, exist_ok=True)
    document = {
        "metaData": {"nodes": len(points), "triangles": len(triangles), "tetrahedra": len(tets), "version": "0.1"},
        "mesh": {"nodes": points.tolist(), "triangles": (triangles + 1).tolist(), "tetrahedra": (tets + 1).tolist()},
        "conditions": {"boundary": [{"tag": tag, "type": BOUNDARY_TYPES[key], "description": key}
                                    for key, tag in tags.items()],
                       "recession": [], "triangle": conditions.tolist()},
    }
    (outdir / "mesh.json").write_text(json.dumps(document, separators=(",", ":")))
    np.savez(outdir / "mesh.npz", points=points, tets=tets, triangles=triangles, inlet=conditions == tags["inlet"])
    info = {"surfaces": {k: len(v) for k, v in groups.items()}, "nodes": len(points), "tetrahedra": len(tets),
            "size_in": size, "size_inlet_in": size_inlet, "sector": sector}
    (outdir / "mesh.info.json").write_text(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("grain")
    ap.add_argument("step")
    ap.add_argument("outdir")
    ap.add_argument("--size", type=float, required=True)
    ap.add_argument("--size-inlet", type=float, default=None)
    ap.add_argument("--grow", type=float, default=None)
    ap.add_argument("--curvature", type=float, default=0)
    ap.add_argument("--sector", default="wedge", choices=("wedge", "full"))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    print(json.dumps(generate(Grain.from_json(args.grain), args.step, args.outdir, args.size,
                              args.sector, args.size_inlet, args.grow, args.curvature, args.threads)))
