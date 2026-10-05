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
gamma = 1.2. Throat diameter 1.84 in, no erosion. Because c* is the one input not taken from the
sources, a second run fits it to the plateau of the published simulation; that single constant
sets the pressure level, and the shape and timing of the trace are then a test of the burn area.

[1] simulates the motor with the Rocgrain 3-D burnback code and a 0-D chamber model. Its run with
the strand burn rate is the like-for-like reference for this pipeline. The measured pressure is
higher and the burn shorter; [1] attributes this to erosive and dynamic burning, which neither its
0-D strand-rate run nor this pipeline models.

Writes results/burn_table.csv, results/comparison.json and results/nawc_motor6.png (PDF in report/fig).

Usage: python run_comparison.py
"""
from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from alight import style
from alight.ballistics import IN, Nozzle, Propellant, simulate
from alight.geometry import TaperedFinocylGrain
from alight.solve import burn_table, work_dir

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


def published() -> dict:
    curves = {}
    with open(HERE / "reference" / "fig10_pressure.csv") as f:
        for row in csv.DictReader(f):
            curves.setdefault(row["series"], []).append((float(row["time_s"]), float(row["pressure_atm"])))
    return {name: np.array(points) for name, points in curves.items()}


def burn_duration(t, p):
    """Time at which the pressure falls through half its maximum on the way down."""
    falling = np.flatnonzero((p[1:] < 0.5 * p.max()) & (np.arange(1, len(p)) > p.argmax()))
    return float(t[falling[0] + 1])


def main():
    # the tapered slots leave a few thin elements, which set a small pseudo-time step: allow more
    # iterations and use the larger stable CFL number (the converged field does not depend on it)
    table = burn_table(GRAIN, WORK, SIZES, threads=6, cfl=2.0, iters=100000)
    RESULTS.mkdir(exist_ok=True)
    table.save(RESULTS / "burn_table.csv")
    reference = published()
    strand, measured = reference["simulated_0d_strand_rate"], reference["measured"]
    plateau = np.linspace(0.3, 3.0, 100)        # after the published run's start-up, before its final rise

    def run(c_star):
        result = simulate(table, replace(PROPELLANT, c_star=c_star), NOZZLE)
        return result.t, result.pressure / ATM

    def facts(t, p):
        difference = 100 * (np.interp(plateau, t, p) / np.interp(plateau, *strand.T) - 1)
        return {"peak_pressure_atm": float(p.max()), "time_of_peak_s": float(t[p.argmax()]),
                "time_to_half_peak_s": burn_duration(t, p), "plateau_mean_difference_pct": float(difference.mean()),
                "plateau_rms_difference_pct": float(np.sqrt(np.mean(difference**2)))}

    t, p = run(C_STAR)
    # pressure scales as c*^(1/(1-n)), so the plateau ratio gives the fitted value directly
    ratio = np.mean(np.interp(plateau, *strand.T) / np.interp(plateau, t, p))
    c_star_fitted = C_STAR * ratio ** (1 - PROPELLANT.exponent)
    t_fit, p_fit = run(c_star_fitted)
    summary = {
        "propellant_volume_in3": table.propellant_volume, "initial_burn_area_in2": float(table.burn_area[0]),
        "peak_burn_area_in2": float(table.burn_area.max()), "web_burnout_in": table.web_burnout,
        "gamma_assumed": GAMMA, "c_star_derived_m_s": C_STAR, "c_star_fitted_m_s": float(c_star_fitted),
        "alight_c_star_derived": facts(t, p), "alight_c_star_fitted": facts(t_fit, p_fit),
        "published_0d_strand_rate": {"peak_pressure_atm": float(strand[:, 1].max()),
                                     "time_of_peak_s": float(strand[strand[:, 1].argmax(), 0]),
                                     "time_to_half_peak_s": burn_duration(*strand.T)},
        "measured": {"plateau_pressure_atm": float(np.interp(2.0, *measured.T)),
                     "time_to_half_plateau_s": float(measured[(measured[:, 0] > 2.0) & (measured[:, 1] < 0.5 * np.interp(2.0, *measured.T))][0, 0])},
    }
    (RESULTS / "comparison.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style.apply(9.5)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=150, gridspec_kw=dict(wspace=0.26))
    left.plot(table.web, table.burn_area, color="0.15", lw=1.6)
    left.set_xlabel("web burned (in)")
    left.set_ylabel("burn area (in²)")
    left.set_title("NAWC motor no. 6: burn area from alight")
    left.set_xlim(0, None)
    left.set_ylim(0, None)
    left.grid(True, alpha=0.25)
    right.plot(t, p, color="#d62728", lw=1.8, label=f"alight + 0-D, c* = {C_STAR:.0f} m/s (derived)")
    right.plot(t_fit, p_fit, color="#d62728", lw=1.2, ls="-.", label=f"alight + 0-D, c* = {c_star_fitted:.0f} m/s (fitted to plateau)")
    right.plot(*strand.T, color="#1f3f8f", lw=1.4, ls="--", label="Willcox et al., Rocgrain + 0-D, strand burn rate")
    right.plot(*measured.T, color="0.35", lw=1.0, ls=":", label="Willcox et al., measured")
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


if __name__ == "__main__":
    main()
