"""Initial grain CAD in build123d, exported as STEP in millimetres.

Run as a module so build123d's OpenCASCADE stays out of the Gmsh process:
    python -m alight.cad grain.json outdir
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from alight.geometry import FinocylGrain, Grain, StarGrain, TaperedFinocylGrain, TubeGrain

MM_PER_IN = 25.4


def _port(grain: Grain):
    """The initial port as a build123d solid (mm), to be cut from the case cylinder."""
    from build123d import (Align, Box, BuildLine, BuildSketch, Cylinder, Kind, Line, Pos, Rot,
                           ThreePointArc, extrude, make_face, offset)

    base = (Align.CENTER, Align.CENTER, Align.MIN)
    s = MM_PER_IN
    if isinstance(grain, TubeGrain):
        return Cylinder(grain.port_radius * s, grain.length * s, align=base)
    if isinstance(grain, FinocylGrain):
        port = Cylinder(grain.port_radius * s, grain.length * s, align=base)
        for i in range(grain.n_fins):
            slot = Box(grain.tip_center * s, grain.fin_width * s, grain.fin_length * s,
                       align=(Align.MIN, Align.CENTER, Align.MIN)) \
                + Pos(grain.tip_center * s, 0, 0) * Cylinder(grain.fin_width / 2 * s, grain.fin_length * s, align=base)
            port += Rot(0, 0, 360.0 * i / grain.n_fins) * Pos(0, 0, grain.fin_start * s) * slot
        return port
    if isinstance(grain, StarGrain):
        rp = grain.skeleton_radius * s
        point = lambda a: (rp * math.cos(a), rp * math.sin(a))
        vertices = grain.skeleton_vertices()
        with BuildSketch() as sketch:
            with BuildLine():
                for i, (start, end, meet) in enumerate(vertices):
                    meet = (meet[0] * s, meet[1] * s)
                    ThreePointArc(point(start), point((start + end) / 2), point(end))
                    Line(point(end), meet)
                    Line(meet, point(vertices[(i + 1) % len(vertices)][0]))
            make_face()
            offset(amount=grain.fillet_radius * s, kind=Kind.ARC)
        return extrude(sketch.sketch, grain.length * s)
    if isinstance(grain, TaperedFinocylGrain):
        from build123d import Axis, Plane, Polygon, revolve

        outline = [(0.0, grain.bore[0][0])] + [(r, z) for z, r in grain.bore] + [(0.0, grain.bore[-1][0])]
        port = revolve(Plane.XZ * Polygon(*[(x * s, z * s) for x, z in outline], align=None), Axis.Z)
        if grain.slots:
            from build123d import Circle, Rectangle, loft

            half = grain.slot_width / 2 * s
            sections = []
            for z, tip in grain.slots:      # the slot cross-section at each station: a bar with a round tip
                centre = tip * s - half
                profile = Rectangle(centre, 2 * half, align=(Align.MIN, Align.CENTER)) + Pos(centre, 0) * Circle(half)
                sections.append(Plane.XY.offset(z * s) * profile)
            slot = loft(sections, ruled=True)
            for i in range(grain.n_slots):
                port += Rot(0, 0, 360.0 * i / grain.n_slots) * slot
        return port
    raise TypeError(f"no CAD for {type(grain).__name__}")


def build_grain(grain: Grain):
    """Full propellant grain as a build123d solid (mm)."""
    from build123d import Align, Cylinder

    case = Cylinder(grain.case_radius * MM_PER_IN, grain.length * MM_PER_IN,
                    align=(Align.CENTER, Align.CENTER, Align.MIN))
    return case - _port(grain)


def build_wedge(grain: Grain, solid=None):
    """One symmetry wedge of the grain, 0 <= theta <= grain.wedge_angle."""
    from build123d import Polygon, extrude

    solid = build_grain(grain) if solid is None else solid
    reach = 2 * grain.case_radius * MM_PER_IN
    a = grain.wedge_angle
    profile = Polygon((0, 0), (reach, 0), (reach * math.cos(a), reach * math.sin(a)), align=None)
    return solid & extrude(profile, grain.length * MM_PER_IN)


def export(grain: Grain, outdir) -> dict:
    from build123d import export_step

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    solid = build_grain(grain)
    wedge = build_wedge(grain, solid)
    export_step(solid, str(outdir / "grain.step"))
    export_step(wedge, str(outdir / "wedge.step"))
    info = {
        "grain_volume_in3": solid.volume / MM_PER_IN**3,
        "wedge_volume_in3": wedge.volume / MM_PER_IN**3,
        "port_volume_in3": grain.case_area * grain.length - solid.volume / MM_PER_IN**3,
    }
    (outdir / "cad.json").write_text(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    print(json.dumps(export(Grain.from_json(sys.argv[1]), sys.argv[2])))
