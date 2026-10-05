"""Lumped (0-D) internal ballistics driven by a burn-area table.

SI units inside; the burn table is in inches. Not modelled: erosive burning and
axial pressure drop (see ballistics1d for those), throat erosion, ignition transient.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from alight.postprocess import BurnbackTable

IN = 0.0254
PSI = 6894.757
LBF = 4.448222
G0 = 9.80665


@dataclass(frozen=True)
class Propellant:
    name: str
    density: float          # kg/m^3
    burn_rate_ref: float    # m/s at p_ref
    p_ref: float            # Pa
    exponent: float
    c_star: float           # m/s
    gamma: float

    def burn_rate(self, p):
        return self.burn_rate_ref * (np.maximum(p, 0.0) / self.p_ref) ** self.exponent

    @property
    def gas_rt(self) -> float:
        """R * T_c of the combustion gas, from c*."""
        g = self.gamma
        return (self.c_star * math.sqrt(g) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))) ** 2


# Aluminized AP/HTPB composite (about 68 % AP, 18 % Al, 14 % HTPB binder).
# Representative handbook-level values, not a specific qualified formulation.
AP_HTPB_AL = Propellant(name="AP/HTPB/Al composite (68/18/14)", density=1800.0,
                        burn_rate_ref=0.35 * IN, p_ref=1000 * PSI, exponent=0.35,
                        c_star=1580.0, gamma=1.14)


@dataclass(frozen=True)
class Nozzle:
    throat_diameter: float           # m
    expansion_ratio: float = 8.0
    ambient_pressure: float = 101325.0

    @property
    def throat_area(self) -> float:
        return math.pi * self.throat_diameter**2 / 4


def exit_pressure_ratio(gamma: float, expansion_ratio: float) -> float:
    """p_exit / p_chamber for supersonic isentropic flow at the given area ratio."""
    g = gamma

    def residual(pr):
        return ((g + 1) / 2) ** (1 / (g - 1)) * pr ** (1 / g) \
            * math.sqrt((g + 1) / (g - 1) * (1 - pr ** ((g - 1) / g))) - 1 / expansion_ratio

    return brentq(residual, 1e-9, (2 / (g + 1)) ** (g / (g - 1)))


def thrust(p, propellant: Propellant, nozzle: Nozzle):
    """Ideal thrust with the nozzle flowing full; clipped at zero during tail-off."""
    g = propellant.gamma
    pr = exit_pressure_ratio(g, nozzle.expansion_ratio)
    momentum = math.sqrt(2 * g * g / (g - 1) * (2 / (g + 1)) ** ((g + 1) / (g - 1)) * (1 - pr ** ((g - 1) / g)))
    force = nozzle.throat_area * (p * (momentum + pr * nozzle.expansion_ratio)
                                  - nozzle.ambient_pressure * nozzle.expansion_ratio)
    return np.maximum(force, 0.0)


def equilibrium_pressure(burn_area_m2, propellant: Propellant, nozzle: Nozzle):
    """Steady-state chamber pressure for a given burn area."""
    a = propellant.burn_rate_ref / propellant.p_ref ** propellant.exponent
    kn = burn_area_m2 / nozzle.throat_area
    return (propellant.density * a * propellant.c_star * kn) ** (1 / (1 - propellant.exponent))


def size_throat(table: BurnbackTable, propellant: Propellant, peak_pressure: float,
                expansion_ratio: float = 8.0, ambient_pressure: float = 101325.0) -> "Nozzle":
    """Nozzle whose throat puts the simulated peak chamber pressure at `peak_pressure`."""
    a = propellant.burn_rate_ref / propellant.p_ref ** propellant.exponent
    area_max = table.burn_area.max() * IN**2
    # steady-state estimate, then refine against the transient simulation
    guess = math.sqrt(4 / math.pi * propellant.density * a * propellant.c_star * area_max
                      / peak_pressure ** (1 - propellant.exponent))

    def excess(diameter):
        return simulate(table, propellant, Nozzle(diameter, expansion_ratio, ambient_pressure)).peak_pressure \
            - peak_pressure

    return Nozzle(brentq(excess, 0.9 * guess, 1.1 * guess, xtol=1e-7), expansion_ratio, ambient_pressure)


@dataclass
class BallisticsResult:
    t: np.ndarray
    pressure: np.ndarray    # Pa
    thrust: np.ndarray      # N
    web: np.ndarray         # in
    burn_area: np.ndarray   # in^2
    propellant_mass: float  # kg
    expelled_mass: float    # kg

    @property
    def peak_pressure(self) -> float:
        return float(self.pressure.max())

    @property
    def total_impulse(self) -> float:
        return float(np.trapezoid(self.thrust, self.t))

    @property
    def burn_time(self) -> float:
        """Action time: first to last crossing of 10 % of peak pressure."""
        above = np.flatnonzero(self.pressure >= 0.1 * self.peak_pressure)
        return float(self.t[above[-1]] - self.t[above[0]])

    @property
    def specific_impulse(self) -> float:
        return self.total_impulse / (self.propellant_mass * G0)

    def at(self, t):
        return {name: np.interp(t, self.t, getattr(self, name))
                for name in ("pressure", "thrust", "web", "burn_area")}

    def summary(self) -> dict:
        mean_p = np.trapezoid(self.pressure, self.t) / self.t[-1]
        return {
            "peak_pressure_psi": self.peak_pressure / PSI,
            "mean_pressure_psi": float(mean_p) / PSI,
            "burn_time_s": self.burn_time,
            "peak_thrust_lbf": float(self.thrust.max()) / LBF,
            "total_impulse_lbf_s": self.total_impulse / LBF,
            "total_impulse_kN_s": self.total_impulse / 1000,
            "specific_impulse_s": self.specific_impulse,
            "propellant_mass_kg": self.propellant_mass,
            "expelled_mass_kg": self.expelled_mass,
        }


def simulate(table: BurnbackTable, propellant: Propellant, nozzle: Nozzle,
             t_max: float = 120.0, dt_out: float = 0.002) -> BallisticsResult:
    web_tab = table.web * IN
    area_tab = table.burn_area * IN**2
    volume_tab = table.free_volume * IN**3
    web_end = web_tab[-1]
    rt = propellant.gas_rt
    p_amb = nozzle.ambient_pressure

    def state(web):
        return (np.interp(web, web_tab, area_tab, right=0.0), np.interp(web, web_tab, volume_tab))

    def rhs(t, y):
        web, p = y
        area, volume = state(web)
        rate = float(propellant.burn_rate(p)) if web < web_end else 0.0
        # the nozzle only discharges once the chamber is above ambient
        m_out = p * nozzle.throat_area / propellant.c_star if p > p_amb else 0.0
        dp = rt / volume * ((propellant.density - p / rt) * area * rate - m_out)
        return [rate, dp]

    def tailed_off(t, y):
        # End of the run: chamber pressure falling through 1.5 atmospheres. Thin slivers can keep burning
        # slowly at ambient pressure for a long time; they add nothing to the motor's performance.
        return y[1] - 1.5 * p_amb
    tailed_off.terminal = True
    tailed_off.direction = -1

    sol = solve_ivp(rhs, (0.0, t_max), [0.0, p_amb], method="LSODA", events=tailed_off,
                    max_step=0.01, rtol=1e-8, atol=[1e-10, 1e-2], dense_output=True)
    t = np.arange(0.0, sol.t[-1], dt_out)
    web, p = sol.sol(t)
    web = np.minimum(web, web_end)
    force = thrust(p, propellant, nozzle)
    m_out = np.where(p > p_amb, p * nozzle.throat_area / propellant.c_star, 0.0)
    return BallisticsResult(t=t, pressure=p, thrust=force, web=web / IN,
                            burn_area=np.interp(web, web_tab, area_tab, right=0.0) / IN**2,
                            propellant_mass=propellant.density * table.burned_volume[-1] * IN**3,
                            expelled_mass=float(np.trapezoid(m_out, t)))
