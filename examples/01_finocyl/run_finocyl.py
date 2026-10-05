#!/usr/bin/env python3
"""Final results for the baseline finocyl grain: ballistics, video and quad chart.

Needs the output of run_sensitivity.py (burn tables, sensitivity.csv, design.json).

Usage: python run_finocyl.py [--no-video] [--display-pair 0.2 0.1]
Writes into results/: burn_table.csv, ballistics.csv, summary.json, burnback.mp4 / .jpg,
quad_chart.png / .pdf.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from alight import quad_chart, sections, video
from alight import sensitivity as sens
from alight.ballistics import AP_HTPB_AL, IN, LBF, PSI, Nozzle, simulate
from alight.geometry import FinocylGrain
from alight.postprocess import BurnbackTable
from alight.solve import work_dir

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results"
WORK = work_dir(HERE / "_work")
FIGURES = HERE.parents[1] / "report" / "fig"
GRAIN = FinocylGrain()
MESH_SHOWN = 0.3


def report_figures(rows, tables, final_key, reference_key, views, table, result):
    """The sensitivity and result figures of the report, as PDFs in report/fig."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from alight import style
    from alight.video import FRONT, PRESSURE, THRUST

    style.apply(9.5)
    FIGURES.mkdir(parents=True, exist_ok=True)
    raw = sorted((r for r in rows if r["kind"] == "raw"), key=lambda r: -r["size_in"])
    ext = sorted((r for r in rows if r["kind"] == "extrapolated"), key=lambda r: -r["size_in"])
    reference = next(r for r in rows if r["kind"] == "reference-fmm" and f"dx{r['size_in']:g}" in reference_key)
    final = ext[-1]

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.7), dpi=150, gridspec_kw=dict(wspace=0.26))
    inset = left.inset_axes([0.37, 0.12, 0.33, 0.38])
    shades = plt.cm.Blues(np.linspace(0.3, 0.95, len(raw)))
    for axis in (left, inset):
        for row, shade in zip(raw, shades):
            curve = tables[f"raw_h{row['size_in']:g}"]
            axis.plot(curve.web, curve.burn_area, color=shade, lw=1.1,
                      label=f"raw, h = {row['size_in']:g} in" if row in (raw[0], raw[-1]) else None)
        axis.plot(table.web, table.burn_area, color=FRONT, lw=1.9, label="extrapolated, h → 0")
        axis.plot(tables[reference_key].web, tables[reference_key].burn_area, color="k", lw=1.0, ls="--",
                  label="fast-marching reference")
    peak = table.web[table.burn_area.argmax()]
    inset.set_xlim(peak - 0.15, peak + 0.25)
    inset.set_ylim(0.9 * table.burn_area.max(), 1.02 * table.burn_area.max())
    inset.tick_params(labelsize=7)
    inset.grid(True, alpha=0.25)
    left.indicate_inset_zoom(inset, edgecolor="0.4")
    left.set_xlabel("web burned (in)")
    left.set_ylabel("burn area (in²)")
    left.set_xlim(0, 3.2)
    left.set_ylim(0, 1.42 * table.burn_area.max())
    left.grid(True, alpha=0.25)
    left.legend(fontsize=7.8, loc="upper center", ncol=2, frameon=False)
    right.axhspan(-1, 1, color="0.88", lw=0)
    for key, name, color, marker in (("peak_pressure_psi", "peak pressure", PRESSURE, "o"),
                                     ("burn_time_s", "burn time", "#2a9d5c", "s"),
                                     ("total_impulse_lbf_s", "total impulse", THRUST, "^")):
        pct = lambda row: 100 * (row[key] / final[key] - 1)
        right.plot([r["size_in"] for r in raw], [pct(r) for r in raw], color=color, marker=marker, mfc="white", ms=5,
                   lw=1.2, label=f"{name}, raw")
        right.plot([r["size_in"] for r in ext], [pct(r) for r in ext], color=color, marker=marker, ms=5, lw=1.2,
                   ls="--", label=f"{name}, extrapolated pair")
        right.plot([0], [pct(reference)], color="k", marker=marker, ms=7, mfc=color, ls="none", clip_on=False, zorder=5)
    right.axhline(0, color="k", lw=0.6)
    right.set_xlim(0, 1.05 * raw[0]["size_in"])
    right.set_ylim(-12.5, 5.5)
    right.set_xlabel("element size h (in)")
    right.set_ylabel("difference from final (%)")
    right.grid(True, alpha=0.25)
    right.legend(fontsize=7.4, loc="lower left", ncol=2, frameon=False)
    fig.savefig(FIGURES / "finocyl_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(10.5, 3.6), dpi=150)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.2, 1.25], height_ratios=[1, 1], wspace=0.42, hspace=0.5)
    levels = np.arange(0.0, table.web_burnout + 0.25, 0.25)
    propellant = lambda field: np.ma.masked_where(~(field > 0), field)
    radius = GRAIN.case_radius
    ax = fig.add_subplot(grid[:, 0])
    ax.contourf(views.x, views.x, propellant(views.fin), levels=levels, cmap="YlOrRd", extend="max")
    ax.contour(views.x, views.x, propellant(views.fin), levels=levels[1:], colors="k", linewidths=0.35)
    ax.add_patch(plt.Circle((0, 0), radius, fill=False, ec="k", lw=1.3))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Burning surface every 0.25 in, z = {views.z_fin:g} in", fontsize=9.5)
    ax = fig.add_subplot(grid[0, 1])
    ax.contourf(views.z, views.x, propellant(views.longitudinal), levels=levels, cmap="YlOrRd", extend="max")
    ax.contour(views.z, views.x, propellant(views.longitudinal), levels=levels[1::2], colors="k", linewidths=0.3)
    ax.add_patch(plt.Rectangle((0, -radius), GRAIN.length, 2 * radius, fill=False, ec="k", lw=1.1))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Longitudinal section through a fin pair", fontsize=9.5)
    ax = fig.add_subplot(grid[1, 1])
    ax.plot(table.web, table.burn_area, color="0.15", lw=1.6)
    ax.set_xlabel("web burned (in)")
    ax.set_ylabel("burn area (in²)")
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.25)
    ax = fig.add_subplot(grid[:, 2])
    ax.plot(result.t, result.pressure / PSI, color=PRESSURE, lw=1.8)
    ax.set_ylabel("chamber pressure (psi)", color=PRESSURE)
    ax.set_xlabel("time (s)")
    ax.set_xlim(0, result.t[-1])
    ax.set_ylim(0, 1.12 * result.peak_pressure / PSI)
    ax.grid(True, alpha=0.25)
    twin = ax.twinx()
    twin.plot(result.t, result.thrust / LBF, color=THRUST, lw=1.8)
    twin.set_ylabel("thrust (lbf)", color=THRUST)
    twin.set_ylim(0, 1.12 * result.thrust.max() / LBF)
    fig.savefig(FIGURES / "finocyl_results.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--display-pair", type=float, nargs=2, default=(0.2, 0.1),
                    help="element sizes whose extrapolated field is drawn in the section views")
    args = ap.parse_args()

    design = json.loads((OUTPUT / "design.json").read_text())
    tables = {p.stem: BurnbackTable.load(p) for p in sorted((OUTPUT / "tables").glob("*.csv"))}
    final_key = design["final_table"].split("/")[-1].removesuffix(".csv")
    reference_key = design["reference_table"].split("/")[-1].removesuffix(".csv")
    table = tables[final_key]
    nozzle = Nozzle(design["throat_diameter_in"] * IN, design["expansion_ratio"])
    result = simulate(table, AP_HTPB_AL, nozzle)

    shutil.copyfile(OUTPUT / design["final_table"], OUTPUT / "burn_table.csv")
    np.savetxt(OUTPUT / "ballistics.csv",
               np.column_stack([result.t, result.pressure / PSI, result.thrust / LBF, result.web, result.burn_area]),
               delimiter=",", header="time_s,pressure_psi,thrust_lbf,web_in,burn_area_in2", comments="", fmt="%.6g")
    summary = {**result.summary(), "throat_diameter_in": design["throat_diameter_in"],
               "expansion_ratio": design["expansion_ratio"], "propellant": AP_HTPB_AL.name,
               "initial_burn_area_in2": float(table.burn_area[0]), "peak_burn_area_in2": float(table.burn_area.max()),
               "web_at_peak_area_in": float(table.web[table.burn_area.argmax()]),
               "web_burnout_in": table.web_burnout, "final_pair_in": design["final_pair_in"]}
    summary = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in summary.items()}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
    for key, value in summary.items():
        print(f"{key:28s} {value}")

    coarse, fine = args.display_pair
    field = sens.extrapolate(sens.run_level(GRAIN, WORK, coarse), sens.run_level(GRAIN, WORK, fine), coarse, fine)
    views = sections.cached(GRAIN, field, WORK / f"sections_h{coarse:g}_h{fine:g}.npz")

    rows = sens.read_csv(OUTPUT / "sensitivity.csv")
    print(quad_chart.render(GRAIN, AP_HTPB_AL, nozzle, rows, tables, final_key, reference_key,
                            design["final_pair_in"], views, result,
                            sens.level_dir(WORK, MESH_SHOWN) / "mesh.npz", MESH_SHOWN, OUTPUT / "quad_chart"))
    report_figures(rows, tables, final_key, reference_key, views, table, result)
    if not args.no_video:
        title = (f"Finocyl grain burnback: {GRAIN.case_diameter:g} in OD × {GRAIN.length:g} in, "
                 f"Ø{GRAIN.port_diameter:g} in port, {GRAIN.n_fins} fins × {GRAIN.fin_width:g} in, AP/HTPB/Al")
        print(video.render(GRAIN, views, table, result, OUTPUT / "burnback.mp4", title))


if __name__ == "__main__":
    main()
