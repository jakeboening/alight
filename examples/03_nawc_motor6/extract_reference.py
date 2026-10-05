#!/usr/bin/env python3
"""Read the published pressure traces of NAWC motor no. 6 out of the paper's PDF.

M. A. Willcox, M. Q. Brewster, K. C. Tang, D. S. Stewart and I. Kuznetsov, "Solid Rocket Motor
Internal Ballistics Simulation Using Three-Dimensional Grain Burnback", Journal of Propulsion and
Power, Vol. 23, No. 3, 2007, pp. 575-584 (open copy: https://www.ideals.illinois.edu/items/14484).

Figure 10 plots chamber pressure against time: the measured trace and two 0-D simulations, one with
the propellant's strand burn rate ("unmodified") and one with an adjusted rate. The figure is vector
graphics, so the curves are read from the PDF paths and scaled with the axis tick labels. Writes
reference/fig10_pressure.csv (series, time in s, pressure in atm).

Usage: python extract_reference.py paper.pdf      (needs pymupdf)
"""
import sys
from pathlib import Path

import numpy as np
import pymupdf

PAGE = 6                      # zero-based page of Figure 10
REGION = (325, 570, 540, 740)  # the figure's part of the page, points
CROSSING = 1.3                 # s, where the measured trace drops below the modified-rate simulation


def main():
    page = pymupdf.open(sys.argv[1])[PAGE]
    if "Fig. 10" not in page.get_text():
        raise SystemExit("Figure 10 is not on the expected page")
    x0, y0, x1, y1 = REGION
    inside = lambda r: r.x0 > x0 and r.y0 > y0 and r.x1 < x1 and r.y1 < y1
    drawings = [d for d in page.get_drawings() if inside(d["rect"])]
    frame = max((d["rect"] for d in drawings), key=lambda r: r.width * r.height)   # the axes path

    # axis scales: straight-line fit of tick-label value against label position
    words = [((w[0] + w[2]) / 2, (w[1] + w[3]) / 2, float(w[4])) for w in page.get_text("words")
             if x0 < w[0] < x1 and y0 < w[1] < y1 and w[4].isdigit()]
    below = [(x, v) for x, y, v in words if frame.y1 < y < frame.y1 + 14]
    left = [(y, v) for x, y, v in words if x < frame.x0]
    time_of = np.poly1d(np.polyfit(*zip(*below), 1))
    pressure_of = np.poly1d(np.polyfit(*zip(*left), 1))

    # the full line is one long stroked path; dashes and dots are drawn as small filled shapes
    solid = max((d for d in drawings if d["type"] == "s" and (d.get("width") or 0) > 0.6), key=lambda d: len(d["items"]))
    line = [(item[1].x, item[1].y) for item in solid["items"] if item[0] == "l"] + [tuple(solid["items"][-1][2])]
    marks = np.array([((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, max(r.width, r.height))
                      for r in (d["rect"] for d in drawings if d["type"] == "f")
                      if r.x0 >= frame.x0 - 1 and r.x1 <= frame.x1 + 1 and r.y1 <= frame.y1 + 1
                      and not (r.y0 < frame.y0 + 34 and frame.x0 + 18 < r.x0 < frame.x0 + 48)])   # legend samples

    # Two broken curves share the marks: the measured trace, which starts high, and the modified-rate
    # simulation, which starts low. Walk through time in small bins and split each bin at its largest
    # gap in pressure; a bin with a single cluster goes to the curve whose last point is nearest.
    t, pressure = time_of(marks[:, 0]), pressure_of(marks[:, 1])
    keep = ~((t > 0.45) & (t < 1.35) & (pressure > 105))          # legend samples inside the plot area
    marks, t, pressure = marks[keep], t[keep], pressure[keep]
    last = {"measured": pressure[t < 0.2].max(), "modified": pressure[t < 0.2].min()}
    rows_of = {"measured": [], "modified": []}
    for start in np.arange(0.0, t.max() + 0.1, 0.1):
        rows = np.flatnonzero((t >= start) & (t < start + 0.1))
        if not rows.size:
            continue
        rows = rows[np.argsort(pressure[rows])]
        gaps = np.diff(pressure[rows])
        clusters = [rows[:gaps.argmax() + 1], rows[gaps.argmax() + 1:]] if gaps.size and gaps.max() > 3.0 else [rows]
        if len(clusters) == 2:
            lower, upper = clusters
            # the curves cross once, near 1.3 s: the measured trace is the upper one before, the lower after
            high, low = ("measured", "modified") if start < CROSSING else ("modified", "measured")
            assignment = {high: upper, low: lower}
        else:
            mean = pressure[rows].mean()
            # the measured trace has ended by 4.1 s; later marks are the tail of the simulation
            near = "modified" if start > 4.1 else min(last, key=lambda name: abs(last[name] - mean))
            other = "modified" if near == "measured" else "measured"
            # where the two curves coincide the cluster belongs to both
            assignment = {near: rows, other: rows} if abs(last[other] - mean) < 3.0 else {near: rows}
        for name, members in assignment.items():
            rows_of[name] += list(members)
            last[name] = pressure[members].mean()
    measured, modified = rows_of["measured"], rows_of["modified"]
    order = lambda rows: marks[np.array(sorted(rows, key=lambda i: (marks[i, 0], marks[i, 1])))][:, :2]
    series = {"measured": order(measured), "simulated_0d_strand_rate": np.array(line),
              "simulated_0d_modified_rate": order(modified)}
    out = Path(__file__).resolve().parent / "reference"
    out.mkdir(exist_ok=True)
    with open(out / "fig10_pressure.csv", "w") as f:
        f.write("series,time_s,pressure_atm\n")
        for name, xy in series.items():
            for x, y in xy:
                f.write(f"{name},{time_of(x):.4f},{pressure_of(y):.2f}\n")
            t, p = time_of(xy[:, 0]), pressure_of(xy[:, 1])
            print(f"{name}: {len(xy)} points, t {t.min():.2f}..{t.max():.2f} s, p {p.min():.1f}..{p.max():.1f} atm")


if __name__ == "__main__":
    main()
