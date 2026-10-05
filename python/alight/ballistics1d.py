"""Quasi-steady 1-D internal ballistics with erosive burning.

The port is divided into axial stations. Each station has its own web burned and takes its burning
perimeter and port area from the 3-D burnback solution at that web. Gas is added along the port, so
mass flux and Mach number grow and static pressure falls toward the nozzle; the local burn rate is
the pressure term a P^n plus an erosive term that depends on the local mass flux.

Approximations: each cross-section regresses by its own web as it would under uniform burning (the
usual 1-D ballistics assumption; axial coupling of the regression is ignored); the flow is
quasi-steady with constant stagnation temperature; chamber filling, dynamic burning, the ignition
transient and throat erosion are not modelled.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from alight.ballistics import IN, Nozzle, Propellant
from alight.postprocess import BurnbackTable, extrapolate_tables, iso_area, make_table, tet_volume, triangle_area


@dataclass
class Stations:
    """Burn geometry per axial station against local web. Inches."""
    z: np.ndarray            # station centres
    dz: np.ndarray           # station lengths
    web: np.ndarray          # common web grid
    perimeter: np.ndarray    # (stations, webs) burn area per unit length
    port_area: np.ndarray    # (stations, webs)
    case_area: float

    def at(self, web):
        """Burning perimeter and port area of every station at its own web."""
        step = self.web[1] - self.web[0]
        x = np.clip(web / step, 0, len(self.web) - 1 - 1e-9)
        j = x.astype(int)
        f = x - j
        rows = np.arange(len(self.z))
        pick = lambda table: (1 - f) * table[rows, j] + f * table[rows, j + 1]
        return pick(self.perimeter), pick(self.port_area)

    def save(self, path) -> None:
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path) -> "Stations":
        data = np.load(path)
        return cls(**{k: (float(data[k]) if data[k].ndim == 0 else data[k]) for k in data.files})

    @property
    def burn_table(self) -> BurnbackTable:
        """Whole-grain burn area against a web common to all stations (uniform burning)."""
        area = (self.perimeter * self.dz[:, None]).sum(axis=0)
        port = (self.port_area[:, 0] * self.dz).sum()
        return make_table(self.web, area, port, self.case_area * self.dz.sum() - port)


def _station_tables(solution, edges, n_levels):
    """One burn table per station from a solved sector, all on the same web grid."""
    n = len(edges) - 1
    centroid_z = solution.points[solution.tets][:, :, 2].mean(axis=1)
    bins = np.clip(np.searchsorted(edges, centroid_z) - 1, 0, n - 1)
    web = np.linspace(0.0, float(solution.u.max()), n_levels + 1)
    area = iso_area(solution.points, solution.tets, solution.u, web, bins=(bins, n)) * solution.multiplicity
    volume = np.bincount(bins, weights=tet_volume(solution.points, solution.tets), minlength=n) * solution.multiplicity
    return web, area, volume


def stations_from_solutions(coarse, fine, size_ratio, n_stations=70, n_levels=300, inlet_triangles=None) -> Stations:
    """Station geometry extrapolated to zero element size from two solutions of the same grain."""
    grain = fine.grain
    edges = np.linspace(0.0, grain.length, n_stations + 1)
    dz = np.diff(edges)
    tables = []
    for solution in (coarse, fine):
        web, area, volume = _station_tables(solution, edges, n_levels)
        # the initial surface of each station, from the mesh's inlet triangles
        mesh = np.load(solution.workdir / "mesh.npz")
        tri = solution.points[mesh["triangles"][mesh["inlet"]]]
        bins = np.clip(np.searchsorted(edges, tri[:, :, 2].mean(axis=1)) - 1, 0, n_stations - 1)
        initial = np.bincount(bins, weights=triangle_area(solution.points, mesh["triangles"][mesh["inlet"]]),
                              minlength=n_stations) * solution.multiplicity
        area[0] = initial
        tables.append([make_table(web, area[:, k], grain.case_area * dz[k] - volume[k], volume[k])
                       for k in range(n_stations)])
    final = [extrapolate_tables(c, f, size_ratio, n_levels=n_levels) for c, f in zip(*tables)]
    web = np.linspace(0.0, max(t.web_burnout for t in final), n_levels + 1)
    perimeter = np.array([t.area_at(web) / dz[k] for k, t in enumerate(final)])
    port = np.array([(t.initial_port_volume + np.interp(web, t.web, t.burned_volume)) / dz[k]
                     for k, t in enumerate(final)])
    return Stations(z=(edges[:-1] + edges[1:]) / 2, dz=dz, web=web, perimeter=perimeter, port_area=port,
                    case_area=grain.case_area)


@dataclass(frozen=True)
class LenoirRobillard:
    """Erosive burning added to the pressure term: r_e = alpha G^0.8 / L^0.2 exp(-beta r rho_p / G).

    `length` selects the characteristic length L: "distance" from the head end (the original model),
    "hydraulic" diameter of the port (Lawrence), or "spp", the fit f(D_h) used by the Solid
    Performance Program with D_h in inches: f = 0.90 + 0.189 D_h [1 + 0.043 D_h (1 + 0.023 D_h)].
    SI units throughout; f is taken to be a length in inches.
    """
    alpha: float
    beta: float = 53.0
    length: str = "spp"

    @classmethod
    def from_properties(cls, propellant: Propellant, flame_temperature, viscosity, solid_specific_heat,
                        surface_temperature, initial_temperature=294.0, prandtl=None, **kw):
        """alpha from gas transport properties, as in Lenoir and Robillard's heat-transfer derivation."""
        g = propellant.gamma
        gas_constant = propellant.gas_rt / flame_temperature
        cp = g * gas_constant / (g - 1)
        prandtl = 4 * g / (9 * g - 5) if prandtl is None else prandtl      # Eucken's estimate
        alpha = 0.0288 * cp * viscosity**0.2 * prandtl ** (-2 / 3) / (propellant.density * solid_specific_heat) \
            * (flame_temperature - surface_temperature) / (surface_temperature - initial_temperature)
        return cls(alpha=alpha, **kw)

    def characteristic_length(self, z, hydraulic_diameter):
        if self.length == "distance":
            return np.maximum(z, 1e-6)
        if self.length == "hydraulic":
            return hydraulic_diameter
        d = hydraulic_diameter / IN
        return (0.90 + 0.189 * d * (1 + 0.043 * d * (1 + 0.023 * d))) * IN


@dataclass
class Ballistics1DResult:
    t: np.ndarray
    head_pressure: np.ndarray      # Pa
    aft_pressure: np.ndarray       # Pa, static, at the port exit
    max_mach: np.ndarray           # largest Mach number along the port
    mass_flow: np.ndarray          # kg/s through the nozzle
    max_rate: np.ndarray           # m/s, largest burn rate along the port
    web: np.ndarray                # (times, stations), in
    rate: np.ndarray               # (times, stations), m/s
    z: np.ndarray                  # station centres, in

    @property
    def expelled_mass(self) -> float:
        return float(np.trapezoid(self.mass_flow, self.t))


def simulate_1d(stations: Stations, propellant: Propellant, nozzle: Nozzle, erosive: LenoirRobillard | None = None,
                dt: float = 2e-3, t_max: float = 60.0, max_web_step: float = 0.004) -> Ballistics1DResult:
    """March the station webs in time; at each step solve the steady port flow for the head-end pressure."""
    g = propellant.gamma
    half = (g - 1) / 2
    rt = propellant.gas_rt
    a = propellant.burn_rate_ref / propellant.p_ref ** propellant.exponent
    z, dz = stations.z * IN, stations.dz * IN
    n = len(z)
    web_end = np.array([stations.web[np.flatnonzero(stations.perimeter[k] > 0)[-1] + 1] if stations.perimeter[k].any()
                        else 0.0 for k in range(n)])
    exponent, density, c_star, throat = propellant.exponent, propellant.density, propellant.c_star, nozzle.throat_area
    phi_sonic = (1 + g) / math.sqrt(1 + half)
    scale = math.sqrt(rt / g)

    def mach_from(phi):
        """Subsonic Mach number with (1 + g M^2) / (M sqrt(1 + (g-1)/2 M^2)) = phi."""
        lo, hi = 1e-7, 1.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if (1 + g * mid * mid) / (mid * math.sqrt(1 + half * mid * mid)) > phi:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def flow(p_head, perimeter, area, length):
        """Steady flow along the port for a head-end pressure, station by station.

        Returns the mass-flow mismatch at the nozzle (generated less discharged) and the station
        pressures, Mach numbers and burn rates, or None if the port chokes: then no steady flow
        exists at this head-end pressure and it must be higher.
        """
        mdot, impulse, p = 0.0, p_head * area[0], p_head
        ps, machs, rates = [], [], []
        for k in range(n):
            if k:
                impulse += p * (area[k] - area[k - 1])     # wall force at the area change
            base = a * p**exponent
            rate = base
            if erosive is not None and perimeter[k] > 0:
                for _ in range(60):
                    flux = (mdot + 0.5 * density * rate * perimeter[k] * dz[k]) / area[k]
                    new = base + erosive.alpha * flux**0.8 / length[k]**0.2 * math.exp(-erosive.beta * rate * density / flux)
                    if abs(new - rate) < 1e-9 * rate:
                        break
                    rate = 0.5 * (rate + new)
            # the gas that fills the volume the propellant vacates does not leave the port
            mdot += (density - p / rt) * rate * perimeter[k] * dz[k]
            if mdot > 0:
                phi = impulse / (mdot * scale)
                if phi <= phi_sonic:
                    return None
                mach = mach_from(phi)
            else:
                mach = 0.0
            p = impulse / (area[k] * (1 + g * mach * mach))
            ps.append(p)
            machs.append(mach)
            rates.append(rate)
        stagnation = p * (1 + half * mach * mach) ** (g / (g - 1))
        return mdot - stagnation * throat / c_star, np.array(ps), np.array(machs), np.array(rates), mdot

    def solve(p_guess, perimeter, area, length):
        """Head-end pressure at which generated and discharged mass flow balance (bracketed secant)."""
        mismatch = lambda result: math.inf if result is None else result[0]
        lo = hi = p_guess
        f_lo = f_hi = mismatch(flow(p_guess, perimeter, area, length))
        while f_hi > 0:                    # too little discharge, or choked: raise the pressure
            lo, f_lo = hi, f_hi
            hi *= 1.25
            f_hi = mismatch(flow(hi, perimeter, area, length))
        while f_lo < 0:
            hi, f_hi = lo, f_lo
            lo /= 1.25
            f_lo = mismatch(flow(lo, perimeter, area, length))
        result = None
        for _ in range(60):
            secant = hi - f_hi * (hi - lo) / (f_hi - f_lo) if math.isfinite(f_lo) else math.nan
            p = secant if lo < secant < hi else 0.5 * (lo + hi)
            result = flow(p, perimeter, area, length)
            f = mismatch(result)
            if f > 0:
                lo, f_lo = p, f
            else:
                hi, f_hi = p, f
            if result is not None and (abs(f) < 1e-8 * result[4] or hi - lo < 1e-7 * hi):
                return p, result
        return hi, flow(hi, perimeter, area, length)

    web = np.zeros(n)
    initial_area = (stations.perimeter[:, 0] * stations.dz).sum()
    perimeter, area = stations.at(web)
    # starting guess: the steady pressure for uniform burning
    p_head = (density * a * c_star * (perimeter * stations.dz).sum() * IN**2 / throat) ** (1 / (1 - exponent))
    out = {key: [] for key in ("t", "head", "aft", "mach", "flow", "rate", "web", "rates")}
    t = 0.0
    while t < t_max:
        perimeter, area = stations.at(web)
        perimeter, area = perimeter * IN, area * IN**2
        if (perimeter * dz).sum() / IN**2 < 0.002 * initial_area:
            break
        wetted = np.where(perimeter > 0, perimeter, 2 * np.sqrt(math.pi * area))
        length = None if erosive is None else erosive.characteristic_length(z, 4 * area / wetted)
        p_head, (_, p, mach, rate, mdot) = solve(p_head, perimeter, area, length)
        if p_head < 1.5 * nozzle.ambient_pressure:
            break
        for key, value in zip(out, (t, p_head, p[-1], mach.max(), mdot, rate.max(), web.copy(), rate)):
            out[key].append(value)
        step = min(dt, max_web_step * IN / rate.max())
        web = np.minimum(web + np.where(perimeter > 0, rate, 0.0) * step / IN, web_end)
        t += step
    return Ballistics1DResult(t=np.array(out["t"]), head_pressure=np.array(out["head"]), aft_pressure=np.array(out["aft"]),
                              max_mach=np.array(out["mach"]), mass_flow=np.array(out["flow"]),
                              max_rate=np.array(out["rate"]), web=np.array(out["web"]),
                              rate=np.array(out["rates"]), z=stations.z)
