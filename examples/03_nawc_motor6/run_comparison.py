#!/usr/bin/env python3
"""NAWC tactical motor no. 6: comparison with a published 3-D burnback simulation and a motor firing.

Sources
  [1] M. A. Willcox, M. Q. Brewster, K. C. Tang, D. S. Stewart and I. Kuznetsov, "Solid Rocket Motor
      Internal Ballistics Simulation Using Three-Dimensional Grain Burnback", Journal of Propulsion
      and Power 23(3), 2007, 575-584. Figure 7 gives the grain, Figure 10 the pressure traces.
  [2] F. S. Blomshield, "Pulsed Motor Firings", NAWCWD TP 8444, Naval Air Warfare Center Weapons
      Division, China Lake, 2000 (DTIC ADA382239). Table 3 gives the propellant.

Grain (Figure 7 of [1], inches): 5.0 OD, 69.415 long, forward face inhibited. A cylindrical bore at
the head end changes over stations C to E into a six-slot star that runs to station F and opens to
a cylinder at G. Station values of the bore radius R1 and the slot tip radius R3 are joined by
straight lines here. The 0.213 in rounding where a slot meets the bore is left out, and the aft
face is taken as burning because only the forward face is marked inhibited.

Propellant "A" of [2]: 1.80 g/cm3, 0.678 cm/s at 6.9 MPa, exponent 0.36, speed of sound 1083 m/s in
the chamber. Neither source gives c*; it is derived from the speed of sound with an assumed
gamma = 1.2. Throat diameter 1.84 in, no erosion.

[1] simulates the motor with the Rocgrain 3-D burnback code and a 0-D chamber model. Its run with
the strand burn rate is the like-for-like reference for this pipeline. The measured pressure is
higher and the burn shorter; [1] attributes this to erosive and dynamic burning, which neither its
0-D strand-rate run nor this pipeline models.

Three comparisons that do not depend on the unknown c* are made besides the pressure trace itself:
  - the pressure integral, which a mass balance fixes at (propellant mass) * c* / (throat area);
  - the web burned when the pressure peaks, found by integrating the burn-rate law along the trace;
  - the burn area against web implied by the published trace (quasi-steady inversion), with c*
    taken from the published trace's own pressure integral.

The measured trace is then compared with a 1-D run that includes erosive burning (alight.ballistics1d):
the Lenoir-Robillard model in the form [1] quotes from the Solid Performance Program, with beta = 53
and alpha computed from gas properties. The viscosity, the solid's specific heat and the surface
temperature are not in the sources; typical values are assumed and their effect is reported.
Throat erosion is tried as well. The sources give neither the throat material nor an erosion rate,
so two rates are run to show the effect; they are not part of the baseline.

Writes results/burn_table.csv, results/comparison.json, results/nawc_motor6.png and
results/nawc_motor6_erosive.png (PDFs in report/fig).

Usage: python run_comparison.py
"""
from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from alight import style
from alight.ballistics import IN, PSI, Nozzle, Propellant, ThroatErosion, simulate
from alight.ballistics1d import LenoirRobillard, Stations, simulate_1d, stations_from_solutions
from alight.geometry import TaperedFinocylGrain
from alight.solve import burn_table, run_case, work_dir

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE.parents[1] / "report" / "fig"
WORK = work_dir(HERE / "_work")
SIZES = (0.1, 0.05)   # coarse and fine element size, in
ATM = 101325.0

FACE = 0.19           # station coordinate of the grain's forward face, in
GRAIN = TaperedFinocylGrain(
    case_diameter=5.0, length=69.415,
    #       forward face   station B            C                    D                     F                     G                     aft face
    bore=((0.0, 1.900), (1.19 - FACE, 0.937), (5.78 - FACE, 0.932), (11.78 - FACE, 0.618), (65.78 - FACE, 0.620), (66.78 - FACE, 1.415), (69.415, 1.415)),
    #        C                     D                      E                      F                      G
    slots=((5.78 - FACE, 0.932), (11.78 - FACE, 0.932), (14.78 - FACE, 1.367), (65.78 - FACE, 1.415), (66.78 - FACE, 1.415)),
    n_slots=6, slot_width=2 * 0.192, head_end_inhibited=True, aft_end_inhibited=False)

GAMMA = 1.2           # assumed
SOUND_SPEED = 1083.0  # m/s at 6.9 MPa, Table 3 of [2]
C_STAR = SOUND_SPEED / (GAMMA * (2 / (GAMMA + 1)) ** ((GAMMA + 1) / (2 * (GAMMA - 1))))
PROPELLANT = Propellant(name='NAWC reduced-smoke propellant "A"', density=1800.0, burn_rate_ref=0.678e-2,
                        p_ref=6.9e6, exponent=0.36, c_star=C_STAR, gamma=GAMMA)
NOZZLE = Nozzle(1.84 * IN)
SOLVER = dict(threads=6, cfl=2.0, iters=100000)
FLAME_TEMPERATURE = 2713.0 + 273.15   # K, Table 3 of [2]
# assumed, not in the sources: gas viscosity (Pa s), specific heat of the solid (J/kg/K), surface temperature (K)
EROSION = dict(viscosity=8.5e-5, solid_specific_heat=1500.0, surface_temperature=1000.0)
THROAT_EROSION_RATES = (0.3, 0.8)   # mm/s at 6.9 MPa; illustrative, no throat data in the sources


def published() -> dict:
    curves = {}
    with open(HERE / "reference" / "fig10_pressure.csv") as f:
        for row in csv.DictReader(f):
            curves.setdefault(row["series"], []).append((float(row["time_s"]), float(row["pressure_atm"])))
    return {name: np.array(points) for name, points in curves.items()}


def invert(t, p_atm, mass):
    """Web, burn area and c* implied by a pressure trace under the strand burn-rate law.

    Web is the time integral of a P^n. The pressure integral times the throat area equals the
    propellant mass times c*. Quasi-steady burn area then follows from P^(1-n).
    """
    grid = np.linspace(0.0, t.max(), 6000)
    p = np.interp(grid, t, p_atm, left=p_atm[0]) * ATM
    a = PROPELLANT.burn_rate_ref / PROPELLANT.p_ref ** PROPELLANT.exponent
    rate = a * p ** PROPELLANT.exponent
    web = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(grid))])
    integral = float(np.trapezoid(p, grid))
    c_star = NOZZLE.throat_area * integral / mass
    area = p ** (1 - PROPELLANT.exponent) * NOZZLE.throat_area / (PROPELLANT.density * a * c_star)
    return {"web_in": web / IN, "area_in2": area / IN**2, "pressure_integral_atm_s": integral / ATM,
            "c_star_m_s": c_star, "web_at_peak_in": float(web[p.argmax()] / IN), "web_at_end_in": float(web[-1] / IN)}


def facts(t, p):
    after_peak = np.flatnonzero((p < 0.5 * p.max()) & (t > t[p.argmax()]))
    return {"pressure_at_1s_atm": float(np.interp(1.0, t, p)), "pressure_at_3s_atm": float(np.interp(3.0, t, p)),
            "peak_pressure_atm": float(p.max()), "time_of_peak_s": float(t[p.argmax()]),
            "time_to_half_peak_s": float(t[after_peak[0]])}


def main():
    coarse = run_case(GRAIN, WORK / f"h{SIZES[0]:g}", SIZES[0], caddir=WORK / "cad", **SOLVER).table()
    # the tapered slots leave a few thin elements, which set a small pseudo-time step: SOLVER allows more
    # iterations and uses the larger stable CFL number (the converged field does not depend on it)
    table = burn_table(GRAIN, WORK, SIZES, **SOLVER)
    RESULTS.mkdir(exist_ok=True)
    table.save(RESULTS / "burn_table.csv")
    mass = PROPELLANT.density * table.propellant_volume * IN**3
    reference = published()
    strand, measured = reference["simulated_0d_strand_rate"], reference["measured"]

    def run(burn):
        result = simulate(burn, PROPELLANT, NOZZLE)
        return result.t, result.pressure / ATM

    t, p = run(table)
    t_coarse, p_coarse = run(coarse)
    ours, theirs = invert(t, p, mass), invert(*strand.T, mass)
    plateau = np.linspace(0.5, 3.0, 100)
    level = lambda time, pressure: float(np.mean(100 * (np.interp(plateau, time, pressure) / np.interp(plateau, *strand.T) - 1)))
    summary = {
        "propellant_volume_in3": table.propellant_volume, "propellant_mass_kg": mass,
        "initial_burn_area_in2": float(table.burn_area[0]), "peak_burn_area_in2": float(table.burn_area.max()),
        "web_at_peak_area_in": float(table.web[table.burn_area.argmax()]), "web_burnout_in": table.web_burnout,
        "gamma_assumed": GAMMA, "c_star_derived_m_s": C_STAR,
        "alight": {**facts(t, p), "pressure_integral_atm_s": ours["pressure_integral_atm_s"],
                   "web_at_peak_pressure_in": ours["web_at_peak_in"], "plateau_difference_from_published_pct": level(t, p)},
        "alight_coarse_mesh_only": {**facts(t_coarse, p_coarse), "element_size_in": SIZES[0],
                                    "plateau_difference_from_published_pct": level(t_coarse, p_coarse)},
        "published_0d_strand_rate": {**facts(*strand.T), "pressure_integral_atm_s": theirs["pressure_integral_atm_s"],
                                     "web_at_peak_pressure_in": theirs["web_at_peak_in"], "web_at_end_in": theirs["web_at_end_in"],
                                     "c_star_implied_by_pressure_integral_m_s": theirs["c_star_m_s"]},
        "measured": {"pressure_at_2s_atm": float(np.interp(2.0, *measured.T)), "design_pressure_atm": 1000 * PSI / ATM,
                     "time_to_half_plateau_s": float(measured[(measured[:, 0] > 2.0) & (measured[:, 1] < 0.5 * np.interp(2.0, *measured.T))][0, 0])},
    }
    # ---- 1-D ballistics with erosive burning, against the measured head-end pressure ----
    station_file = WORK / "stations.npz"
    if not station_file.exists():
        solutions = [run_case(GRAIN, WORK / f"h{size:g}", size, caddir=WORK / "cad", **SOLVER) for size in SIZES]
        stations_from_solutions(*solutions, SIZES[0] / SIZES[1]).save(station_file)
    stations = Stations.load(station_file)
    erosion = LenoirRobillard.from_properties(PROPELLANT, flame_temperature=FLAME_TEMPERATURE, **EROSION)
    runs = {"no_erosion": simulate_1d(stations, PROPELLANT, NOZZLE),
            "erosive": simulate_1d(stations, PROPELLANT, NOZZLE, erosive=erosion),
            "erosive_alpha_minus_25pct": simulate_1d(stations, PROPELLANT, NOZZLE, erosive=replace(erosion, alpha=0.75 * erosion.alpha)),
            "erosive_alpha_plus_25pct": simulate_1d(stations, PROPELLANT, NOZZLE, erosive=replace(erosion, alpha=1.25 * erosion.alpha)),
            "erosive_hydraulic_diameter": simulate_1d(stations, PROPELLANT, NOZZLE, erosive=replace(erosion, length="hydraulic")),
            **{f"erosive_throat_erosion_{rate:g}mm_s": simulate_1d(stations, PROPELLANT, NOZZLE, erosive=erosion,
                                                                   throat_erosion=ThroatErosion(rate * 1e-3))
               for rate in THROAT_EROSION_RATES}}
    steady = measured[measured[:, 0] > 0.25]            # after the ignition spike, which the model has no physics for
    window = np.linspace(0.3, 2.8, 120)                 # the quasi-steady part of the firing
    integral_grid = np.linspace(0.0, 4.0, 2000)         # the measured trace ends near 4 s

    def against_measured(run):
        head = run.head_pressure / ATM
        difference = 100 * (np.interp(window, run.t, head) / np.interp(window, *steady.T) - 1)
        plateau = np.interp(2.0, run.t, head)
        falling = np.flatnonzero((head < 0.5 * plateau) & (run.t > 2.0))
        return {"head_pressure_atm": {f"{time:g}s": float(np.interp(time, run.t, head)) for time in (0.0, 0.2, 0.5, 1.0, 2.0, 3.0)},
                "mean_difference_0p3_to_2p8s_pct": float(difference.mean()),
                "rms_difference_0p3_to_2p8s_pct": float(np.sqrt(np.mean(difference**2))),
                "time_to_half_plateau_s": float(run.t[falling[0]]), "largest_burn_rate_cm_s": float(run.max_rate.max() * 100),
                "largest_mach_number": float(run.max_mach.max()), "expelled_mass_kg": run.expelled_mass,
                "pressure_integral_to_4s_atm_s": float(np.trapezoid(np.interp(integral_grid, run.t, head, right=0.0), integral_grid)),
                "final_throat_diameter_in": float(run.throat_diameter[-1] / IN)}

    summary["one_dimensional"] = {
        "erosive_model": {"form": "Lenoir-Robillard with the SPP characteristic length", "beta": erosion.beta,
                          "alpha_si": erosion.alpha, "flame_temperature_K": FLAME_TEMPERATURE, "assumed": EROSION},
        **{name: against_measured(run) for name, run in runs.items()},
        "measured": {"head_pressure_atm": {f"{time:g}s": float(np.interp(time, *measured.T)) for time in (0.2, 0.5, 1.0, 2.0, 3.0)},
                     "peak_atm": float(measured[:, 1].max()), "time_to_half_plateau_s": summary["measured"]["time_to_half_plateau_s"],
                     "pressure_integral_to_4s_atm_s": float(np.trapezoid(np.interp(integral_grid, *measured[np.argsort(measured[:, 0])].T), integral_grid))},
    }
    (RESULTS / "comparison.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style.apply(9.5)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=150, gridspec_kw=dict(wspace=0.26))
    left.plot(table.web, table.burn_area, color="#d62728", lw=1.8, label="alight, extrapolated")
    left.plot(coarse.web, coarse.burn_area, color="#d62728", lw=1.0, ls=":", label=f"alight, {SIZES[0]:g} in mesh only")
    left.plot(theirs["web_in"], theirs["area_in2"], color="#1f3f8f", lw=1.4, ls="--",
              label="implied by the published 0-D trace")
    left.set_xlabel("web burned (in)")
    left.set_ylabel("burn area (in²)")
    left.set_title("NAWC motor no. 6: burn area")
    left.set_xlim(0, 1.9)
    left.set_ylim(0, 1250)
    left.grid(True, alpha=0.25)
    left.legend(fontsize=8, frameon=False, loc="lower left")
    right.plot(t, p, color="#d62728", lw=1.8, label="alight + 0-D, extrapolated")
    right.plot(t_coarse, p_coarse, color="#d62728", lw=1.0, ls=":", label=f"alight + 0-D, {SIZES[0]:g} in mesh only")
    right.plot(*strand.T, color="#1f3f8f", lw=1.4, ls="--", label="Willcox et al., Rocgrain + 0-D, strand burn rate")
    right.plot(*measured.T, color="0.35", lw=1.0, ls="-.", label="Willcox et al., measured")
    right.axhline(1000 * PSI / ATM, color="0.6", lw=0.8)
    right.text(7.4, 1000 * PSI / ATM + 1.5, "design pressure", fontsize=7.5, color="0.4", ha="right")
    right.set_xlabel("time (s)")
    right.set_ylabel("chamber pressure (atm)")
    right.set_title("Chamber pressure")
    right.set_xlim(0, 7.5)
    right.set_ylim(0, 145)
    right.grid(True, alpha=0.25)
    right.legend(fontsize=8, frameon=False, loc="upper right")
    for folder, suffix in ((RESULTS, ".png"), (FIGURES, ".pdf")):
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"nawc_motor6{suffix}", bbox_inches="tight")
    plt.close(fig)

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=150, gridspec_kw=dict(wspace=0.26))
    band = [runs["erosive_alpha_minus_25pct"], runs["erosive_alpha_plus_25pct"]]
    grid = np.linspace(0.0, min(run.t[-1] for run in band), 600)
    left.fill_between(grid, *(np.interp(grid, run.t, run.head_pressure / ATM) for run in band), color="#d62728", alpha=0.18,
                      lw=0, label="alpha ± 25 %")
    left.plot(runs["erosive"].t, runs["erosive"].head_pressure / ATM, color="#d62728", lw=1.8, label="alight 1-D with erosive burning")
    left.plot(runs["no_erosion"].t, runs["no_erosion"].head_pressure / ATM, color="#d62728", lw=1.0, ls=":", label="alight 1-D, no erosion")
    worn = runs[f"erosive_throat_erosion_{THROAT_EROSION_RATES[-1]:g}mm_s"]
    left.plot(worn.t, worn.head_pressure / ATM, color="#1f3f8f", lw=1.0, ls="--",
              label=f"with erosive burning and throat erosion, {THROAT_EROSION_RATES[-1]:g} mm/s")
    left.plot(*measured.T, color="k", lw=1.2, ls="-.", label="Willcox et al., measured")
    left.set_xlabel("time (s)")
    left.set_ylabel("head-end pressure (atm)")
    left.set_title("NAWC motor no. 6: head-end pressure")
    left.set_xlim(0, 6)
    left.set_ylim(0, 170)
    left.grid(True, alpha=0.25)
    left.legend(fontsize=8, frameon=False, loc="upper right")
    for name, line, label in (("erosive", "-", "with erosive burning"), ("no_erosion", ":", "no erosion")):
        run = runs[name]
        for time, color in ((0.15, "#d62728"), (1.0, "#e58a2c"), (2.5, "#1f3f8f")):
            row = np.abs(run.t - time).argmin()
            right.plot(run.z * 2.54, run.rate[row] * 100, color=color, ls=line, lw=1.5 if line == "-" else 1.0,
                       label=f"t = {time:g} s, {label}")
    right.set_xlabel("axial location (cm)")
    right.set_ylabel("burn rate (cm/s)")
    right.set_title("Burn rate along the port")
    right.set_ylim(0, None)
    right.grid(True, alpha=0.25)
    right.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper left")
    for folder, suffix in ((RESULTS, ".png"), (FIGURES, ".pdf")):
        fig.savefig(folder / f"nawc_motor6_erosive{suffix}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
