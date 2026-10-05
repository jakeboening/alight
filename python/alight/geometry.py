"""Grain definitions, their closed-form burn geometry and exact distance fields.

Lengths are inches. The grain axis is z, with z = 0 at the head end. Every grain
is case-bonded on its outer diameter. `inhibited_ends` says which end faces do
not burn.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

KINDS = {}


def _register(cls):
    KINDS[cls.kind] = cls
    return cls


@dataclass(frozen=True)
class Grain:
    """Common interface of all grains."""
    kind = "grain"

    @property
    def case_radius(self) -> float:
        return self.case_diameter / 2

    @property
    def wedge_count(self) -> float:
        """Number of symmetry wedges in the full grain."""
        return 2 * math.pi / self.wedge_angle

    @property
    def inhibited_ends(self) -> tuple:
        """(head end, aft end): True where the end face does not burn."""
        return (self.ends_inhibited, self.ends_inhibited)

    @property
    def case_area(self) -> float:
        return math.pi * self.case_radius**2

    @property
    def initial_port_volume(self) -> float:
        return self.port_area * self.length

    @property
    def propellant_volume(self) -> float:
        return self.case_area * self.length - self.initial_port_volume

    def fold_to_wedge(self, x, y):
        """Map points to the equivalent point in the solved wedge 0 <= theta <= wedge_angle."""
        r = np.hypot(x, y)
        period = 2 * self.wedge_angle
        theta = np.mod(np.arctan2(y, x), period)
        theta = np.where(theta > self.wedge_angle, period - theta, theta)
        return r * np.cos(theta), r * np.sin(theta)

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps({"kind": self.kind, **asdict(self)}, indent=2))

    @staticmethod
    def from_json(path) -> "Grain":
        data = json.loads(Path(path).read_text())
        return KINDS[data.pop("kind")](**data)


@_register
@dataclass(frozen=True)
class TubeGrain(Grain):
    """Hollow cylinder. With burning ends this is a BATES grain."""
    kind = "tube"
    case_diameter: float = 10.0
    port_diameter: float = 4.0
    length: float = 10.0
    ends_inhibited: bool = True

    def __post_init__(self):
        if not 0 < self.port_diameter < self.case_diameter:
            raise ValueError("port must be inside the case")

    @property
    def port_radius(self) -> float:
        return self.port_diameter / 2

    @property
    def wedge_angle(self) -> float:
        return math.pi / 12   # axisymmetric: any wedge will do

    @property
    def port_area(self) -> float:
        return math.pi * self.port_radius**2

    @property
    def web(self) -> float:
        """Web burned at burnout."""
        radial = self.case_radius - self.port_radius
        return radial if self.ends_inhibited else min(radial, self.length / 2)

    @property
    def initial_burn_area(self) -> float:
        return float(self.burn_area(0.0))

    def burn_area(self, web):
        """Closed-form burn area for 0 <= web < burnout web."""
        web = np.asarray(web, float)
        r = self.port_radius + web
        if self.ends_inhibited:
            return 2 * math.pi * r * self.length
        return 2 * math.pi * r * (self.length - 2 * web) + 2 * math.pi * (self.case_radius**2 - r**2)

    def distance(self, x, y, z):
        d = np.hypot(x, y) - self.port_radius
        if self.ends_inhibited:
            return d + 0 * np.asarray(z, float)
        return np.minimum(d, np.minimum(z, self.length - np.asarray(z, float)))


@_register
@dataclass(frozen=True)
class StarGrain(Grain):
    """Classical star: N propellant points with flat flanks, ends inhibited.

    The port is everything within the fillet radius of a "skeleton" region made,
    per point pitch, of an arc of radius `skeleton_radius` spanning the fraction
    1 - `angular_fraction` of the pitch and two straight flanks that meet on the
    line between two arcs at the full `point_angle` (degrees). Burning offsets
    that skeleton, which is what the textbook phase formulas describe.
    """
    kind = "star"
    case_diameter: float = 10.0
    length: float = 1.0
    n_points: int = 6
    skeleton_radius: float = 3.0
    fillet_radius: float = 0.25
    angular_fraction: float = 0.7
    point_angle: float = 70.0
    ends_inhibited: bool = True

    def __post_init__(self):
        if not self.ends_inhibited:
            raise ValueError("the star grain is modelled with inhibited ends")
        if self.n_points < 3 or not 0 < self.angular_fraction < 1:
            raise ValueError("need at least 3 points and 0 < angular fraction < 1")
        if self.half_point_angle <= self.flank_angle or self.half_point_angle >= math.pi / 2:
            raise ValueError("point angle too small or too large for this pitch and angular fraction")
        if self.skeleton_radius + self.fillet_radius >= self.case_radius:
            raise ValueError("port reaches the case")

    @property
    def wedge_angle(self) -> float:
        return math.pi / self.n_points

    @property
    def half_point_angle(self) -> float:
        return math.radians(self.point_angle) / 2

    @property
    def arc_half_angle(self) -> float:
        """Half the angle spanned by one skeleton arc."""
        return math.pi * (1 - self.angular_fraction) / self.n_points

    @property
    def flank_angle(self) -> float:
        """Angle at the axis subtended by one flank."""
        return math.pi * self.angular_fraction / self.n_points

    @property
    def flank_length(self) -> float:
        return self.skeleton_radius * math.sin(self.flank_angle) / math.sin(self.half_point_angle)

    @property
    def point_radius(self) -> float:
        """Radius at which two flanks meet (tip of the propellant point, before filleting)."""
        return self.skeleton_radius * math.sin(self.half_point_angle - self.flank_angle) / math.sin(self.half_point_angle)

    @property
    def web(self) -> float:
        """Web burned when the arcs reach the case; slivers remain after this."""
        return self.case_radius - self.skeleton_radius - self.fillet_radius

    def perimeter(self, web):
        """Closed-form burning perimeter for 0 <= web <= self.web (phases I and II)."""
        d = np.asarray(web, float) + self.fillet_radius
        n, rp, half = self.n_points, self.skeleton_radius, self.half_point_angle
        arc = (rp + d) * self.arc_half_angle
        flank_gone = rp * math.sin(self.flank_angle) / math.cos(half)
        phase1 = arc + d * (math.pi / 2 - half + self.flank_angle) + self.flank_length - d / math.tan(half)
        ratio = np.clip(rp * math.sin(self.flank_angle) / np.maximum(d, 1e-300), -1.0, 1.0)
        phase2 = arc + d * (self.flank_angle + np.arcsin(ratio))
        return 2 * n * np.where(d < flank_gone, phase1, phase2)

    def burn_area(self, web):
        return self.perimeter(web) * self.length

    @property
    def initial_burn_area(self) -> float:
        return float(self.burn_area(0.0))

    @property
    def port_area(self) -> float:
        """Initial port cross-section: skeleton area plus its offset by the fillet radius."""
        n, rp, f, half = self.n_points, self.skeleton_radius, self.fillet_radius, self.half_point_angle
        skeleton = n * (rp**2 * self.arc_half_angle + rp * self.point_radius * math.sin(self.flank_angle))
        if f >= rp * math.sin(self.flank_angle) / math.cos(half):
            raise ValueError("fillet radius removes the flanks; port area formula does not apply")
        # per half pitch: strip along the arc, strip along the flank less the lost corner at the
        # reflex point, and the circular sector around the convex corner
        strips = (rp * f + f**2 / 2) * self.arc_half_angle + f * self.flank_length - f**2 / (2 * math.tan(half)) \
            + f**2 / 2 * (math.pi / 2 - half + self.flank_angle)
        return skeleton + 2 * n * strips

    def skeleton_vertices(self):
        """Per pitch: arc start angle, arc end angle and the flank meeting point (x, y)."""
        pitch = 2 * math.pi / self.n_points
        out = []
        for i in range(self.n_points):
            centre = i * pitch
            meet = centre + pitch / 2
            out.append((centre - self.arc_half_angle, centre + self.arc_half_angle,
                        (self.point_radius * math.cos(meet), self.point_radius * math.sin(meet))))
        return out


@_register
@dataclass(frozen=True)
class FinocylGrain(Grain):
    """Cylindrical port with straight, parallel-sided fins over part of the length.

    Fin i is a slot centred on the angle 2*pi*i/N with a semicircular tip, cut
    from the axis out to the fin tip radius over an axial span.
    """
    kind = "finocyl"
    case_diameter: float = 10.0
    port_diameter: float = 4.0
    length: float = 60.0
    n_fins: int = 6
    fin_width: float = 1.0
    fin_tip_radius: float = 4.0
    fin_length: float = 20.0
    fin_start: float = 40.0
    ends_inhibited: bool = True

    def __post_init__(self):
        if not self.ends_inhibited:
            raise ValueError("the finocyl grain is modelled with inhibited ends")
        if not 0 < self.port_radius < self.case_radius:
            raise ValueError("port must be inside the case")
        if self.n_fins == 0:
            return
        if self.fin_width >= 2 * self.port_radius:
            raise ValueError("fin width must be smaller than the port diameter")
        if not self.port_radius < self.fin_tip_radius < self.case_radius:
            raise ValueError("fin tip must lie between the port and the case")
        if self.fin_tip_radius - self.fin_width / 2 <= 0:
            raise ValueError("fin tip radius too small for the fin width")
        if self.fin_start < 0 or self.fin_end > self.length + 1e-9 or self.fin_length <= 0:
            raise ValueError("fin span must lie inside the grain")
        if self.half_angle_at_port >= math.pi / self.n_fins:
            raise ValueError("fins overlap at the port")

    @property
    def port_radius(self) -> float:
        return self.port_diameter / 2

    @property
    def fin_end(self) -> float:
        return self.fin_start + self.fin_length

    @property
    def tip_center(self) -> float:
        """Radius of the centre of the semicircular fin tip."""
        return self.fin_tip_radius - self.fin_width / 2

    @property
    def half_angle_at_port(self) -> float:
        """Half the angle a fin subtends where it meets the port."""
        return math.asin(self.fin_width / (2 * self.port_radius))

    @property
    def wedge_angle(self) -> float:
        """Smallest symmetry sector: half a fin pitch."""
        return math.pi / self.n_fins if self.n_fins else math.pi / 6

    @property
    def web(self) -> float:
        return self.case_radius - self.port_radius

    # ---- closed-form quantities -------------------------------------------------
    @property
    def fin_section_area(self) -> float:
        """Cross-section area of one fin slot outside the port."""
        half_w = self.fin_width / 2
        x0 = math.sqrt(self.port_radius**2 - half_w**2)
        segment = self.port_radius**2 * self.half_angle_at_port - half_w * x0
        return self.fin_width * (self.tip_center - x0) + math.pi * half_w**2 / 2 - segment

    @property
    def finned_perimeter(self) -> float:
        """Burning perimeter of the finned cross-section."""
        half_w = self.fin_width / 2
        x0 = math.sqrt(self.port_radius**2 - half_w**2)
        per_half_fin = (self.tip_center - x0) + math.pi * half_w / 2 \
            + self.port_radius * (math.pi / self.n_fins - self.half_angle_at_port)
        return 2 * self.n_fins * per_half_fin

    @property
    def n_fin_end_faces(self) -> int:
        """Fin end faces that lie inside the grain (not on an inhibited end)."""
        if not self.n_fins:
            return 0
        return int(self.fin_start > 1e-9) + int(self.fin_end < self.length - 1e-9)

    @property
    def initial_burn_area(self) -> float:
        port = 2 * math.pi * self.port_radius
        if not self.n_fins:
            return port * self.length
        return port * (self.length - self.fin_length) + self.finned_perimeter * self.fin_length \
            + self.n_fin_end_faces * self.n_fins * self.fin_section_area

    @property
    def initial_port_volume(self) -> float:
        fins = self.n_fins * self.fin_section_area * self.fin_length if self.n_fins else 0.0
        return math.pi * self.port_radius**2 * self.length + fins

    # ---- distance field ---------------------------------------------------------
    def distance(self, x, y, z):
        """Exact distance from points in the propellant to the initial port.

        Negative inside the port (magnitude not exact there). With a uniform burn
        rate this is the web burned when the flame reaches the point.
        """
        x, y, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float))
        d = np.hypot(x, y) - self.port_radius
        if not self.n_fins:
            return d
        dz = np.maximum(np.maximum(self.fin_start - z, z - self.fin_end), 0.0)
        for i in range(self.n_fins):
            angle = 2 * math.pi * i / self.n_fins
            c, s = math.cos(angle), math.sin(angle)
            along = x * c + y * s
            across = -x * s + y * c
            nearest = np.clip(along, 0.0, self.tip_center)
            d2 = np.hypot(along - nearest, across) - self.fin_width / 2
            d = np.minimum(d, np.where(dz > 0, np.hypot(np.maximum(d2, 0.0), dz), d2))
        return d


@_register
@dataclass(frozen=True)
class TaperedFinocylGrain(Grain):
    """Bore of varying radius with straight slots of varying depth, defined at axial stations.

    `bore` is a sequence of (z, radius) and `slots` a sequence of (z, tip radius); both
    are joined by straight lines. Slot i is centred on the angle 2*pi*i/N, parallel-sided
    with a round tip. The slots exist between their first and last station and vanish
    wherever the bore is wider than their tip.
    """
    kind = "tapered_finocyl"
    case_diameter: float = 5.0
    length: float = 10.0
    bore: tuple = ((0.0, 1.0), (10.0, 1.0))
    slots: tuple = ()
    n_slots: int = 6
    slot_width: float = 0.4
    head_end_inhibited: bool = True
    aft_end_inhibited: bool = True

    def __post_init__(self):
        for name in ("bore", "slots"):   # JSON gives lists; keep the grain hashable and comparable
            object.__setattr__(self, name, tuple(tuple(float(v) for v in station) for station in getattr(self, name)))
        if abs(self.bore[0][0]) > 1e-9 or abs(self.bore[-1][0] - self.length) > 1e-9:
            raise ValueError("bore stations must run from z = 0 to z = length")
        for stations in (self.bore, self.slots):
            if any(b[0] <= a[0] for a, b in zip(stations, stations[1:])):
                raise ValueError("stations must be in increasing z")
            if any(not 0 < r < self.case_radius for _, r in stations):
                raise ValueError("station radii must lie inside the case")
        if self.slots and (self.slots[0][0] < 0 or self.slots[-1][0] > self.length):
            raise ValueError("slot stations must lie inside the grain")
        if self.slots and min(r for _, r in self.slots) <= self.slot_width:
            raise ValueError("slot tip radius too small for the slot width")

    @property
    def inhibited_ends(self) -> tuple:
        return (self.head_end_inhibited, self.aft_end_inhibited)

    @property
    def wedge_angle(self) -> float:
        return math.pi / self.n_slots

    @property
    def web(self) -> float:
        """Largest radial web: case radius less the smallest bore radius."""
        return self.case_radius - min(r for _, r in self.bore)
