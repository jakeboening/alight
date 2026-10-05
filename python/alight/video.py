"""Burnback video: section views of the regressing grain beside the ballistics traces.

Frames are evenly spaced in time, so the surfaces move at the computed burn rate.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from alight import style
from alight.ballistics import LBF, PSI, BallisticsResult
from alight.geometry import Grain
from alight.postprocess import BurnbackTable
from alight.sections import Sections

PROPELLANT = "#9a7b4f"
GAS = "#ffe9b0"
FRONT = "#d62728"
PRESSURE = "#1f3f8f"
THRUST = "#c8641e"


def _two_tone():
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([GAS, PROPELLANT])
    cmap.set_bad("white")
    return cmap


class SectionView:
    """One section panel: propellant / burnt fill, faint web contours, live front."""

    def __init__(self, ax, horizontal, vertical, field, web_marks):
        self.ax, self.h, self.v, self.field = ax, horizontal, vertical, field
        self.masked = np.ma.masked_invalid(field)
        extent = (horizontal[0], horizontal[-1], vertical[0], vertical[-1])
        self.image = ax.imshow(self._remaining(0.0), extent=extent, origin="lower", cmap=_two_tone(),
                               vmin=0, vmax=1, interpolation="antialiased", aspect="equal")
        ax.contour(horizontal, vertical, self.masked, levels=web_marks, colors="k", linewidths=0.4, alpha=0.35)
        self.front = None

    def _remaining(self, web):
        return np.ma.masked_invalid(np.where(np.isnan(self.field), np.nan, (self.field > web).astype(float)))

    def update(self, web):
        self.image.set_data(self._remaining(web))
        if self.front is not None:
            self.front.remove()
            self.front = None
        level = max(web, 1e-3)
        if level < 0.995 * np.nanmax(self.field):   # no front once the last sliver is gone
            self.front = self.ax.contour(self.h, self.v, self.masked, levels=[level], colors=FRONT, linewidths=1.6)


def _case_circle(ax, radius):
    import matplotlib.pyplot as plt
    ax.add_patch(plt.Circle((0, 0), radius, fill=False, ec="k", lw=1.5))
    ax.set_xlim(-1.06 * radius, 1.06 * radius)
    ax.set_ylim(-1.06 * radius, 1.06 * radius)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render(grain: Grain, sections: Sections, table: BurnbackTable, result: BallisticsResult,
           out, title: str, fps: int = 20, hold_start: float = 1.0, hold_end: float = 2.0) -> dict:
    import imageio_ffmpeg
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    style.apply(10)
    out = Path(out)
    radius = grain.case_radius
    web_marks = np.arange(0.5, radius - grain.port_radius, 0.5)
    t_end = float(result.t[-1])
    times = np.arange(0.0, t_end + 0.5 / fps, 1.0 / fps)
    frames = [0.0] * int(hold_start * fps) + list(times) + [times[-1]] * int(hold_end * fps)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    left = fig.add_gridspec(2, 2, left=0.03, right=0.60, top=0.84, bottom=0.135, height_ratios=[3.0, 1.0],
                            hspace=0.22, wspace=0.04)
    right = fig.add_gridspec(2, 1, left=0.675, right=0.925, top=0.84, bottom=0.09, hspace=0.36)
    ax_plain, ax_fin = fig.add_subplot(left[0, 0]), fig.add_subplot(left[0, 1])
    ax_long = fig.add_subplot(left[1, :])
    ax_p, ax_a = fig.add_subplot(right[0]), fig.add_subplot(right[1])

    views = [SectionView(ax_plain, sections.x, sections.x, sections.plain, web_marks),
             SectionView(ax_fin, sections.x, sections.x, sections.fin, web_marks),
             SectionView(ax_long, sections.z, sections.x, sections.longitudinal, web_marks)]
    for ax, station, name in ((ax_plain, sections.z_plain, "plain port"), (ax_fin, sections.z_fin, "finned")):
        _case_circle(ax, radius)
        ax.set_title(f"Section at z = {station:g} in ({name})", fontsize=10)
    ax_long.add_patch(plt.Rectangle((0, -radius), grain.length, 2 * radius, fill=False, ec="k", lw=1.5))
    ax_long.set_xlim(-0.01 * grain.length, 1.01 * grain.length)
    ax_long.set_ylim(-1.08 * radius, 1.08 * radius)
    ax_long.set_yticks([])
    ax_long.set_xlabel("z (in)   head end  →  aft end (nozzle)")
    ax_long.set_title("Longitudinal section through a fin pair", fontsize=10)
    for station in (sections.z_plain, sections.z_fin):
        ax_long.axvline(station, color="k", ls=":", lw=0.9)
    for spine in ax_long.spines.values():
        spine.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=PROPELLANT), plt.Rectangle((0, 0), 1, 1, fc=GAS, ec="0.6"),
               plt.Line2D([], [], color=FRONT, lw=1.6), plt.Line2D([], [], color="k", lw=0.4, alpha=0.5)]
    fig.legend(handles, ["propellant", "burnt / gas", "burning surface", "web contours every 0.5 in"],
               loc="lower left", bbox_to_anchor=(0.03, 0.0), ncol=4, frameon=False, fontsize=9)

    ax_p.plot(result.t, result.pressure / PSI, color=PRESSURE, lw=1.8)
    ax_p.set_ylabel("chamber pressure (psi)", color=PRESSURE)
    ax_p.set_xlabel("time (s)")
    ax_p.set_xlim(0, t_end)
    ax_p.set_ylim(0, 1.12 * result.peak_pressure / PSI)
    ax_p.grid(True, alpha=0.25)
    ax_f = ax_p.twinx()
    ax_f.plot(result.t, result.thrust / LBF, color=THRUST, lw=1.8)
    ax_f.set_ylabel("thrust (lbf)", color=THRUST)
    ax_f.set_ylim(0, 1.12 * result.thrust.max() / LBF)
    cursor = ax_p.axvline(0.0, color="k", lw=1.0)
    dot_p, = ax_p.plot([], [], "o", color=PRESSURE, ms=6)
    dot_f, = ax_f.plot([], [], "o", color=THRUST, ms=6)

    ax_a.plot(table.web, table.burn_area, color="0.15", lw=1.8)
    ax_a.set_xlabel("web burned (in)")
    ax_a.set_ylabel("burn area (in²)")
    ax_a.set_xlim(0, 1.03 * table.web_burnout)
    ax_a.set_ylim(0, 1.12 * table.burn_area.max())
    ax_a.grid(True, alpha=0.25)
    dot_a, = ax_a.plot([], [], "o", color=FRONT, ms=7)

    fig.text(0.03, 0.955, title, fontsize=13, fontweight="bold", va="center")
    # one text item per counter at a fixed position, so the proportional font does not jitter
    counter_items = [fig.text(x, 0.905, "", fontsize=12.5, va="center")
                     for x in (0.03, 0.125, 0.25, 0.44, 0.57, 0.70)]
    mass_total = table.burned_volume[-1]

    def draw(t):
        now = result.at(t)
        web = float(now["web"])
        for view in views:
            view.update(web)
        cursor.set_xdata([t, t])
        dot_p.set_data([t], [now["pressure"] / PSI])
        dot_f.set_data([t], [now["thrust"] / LBF])
        dot_a.set_data([web], [now["burn_area"]])
        left_pct = 100 * (1 - np.interp(web, table.web, table.burned_volume) / mass_total)
        for item, text in zip(counter_items, (
                f"t = {t:.2f} s", f"web = {web:.2f} in", f"burn area = {now['burn_area']:,.0f} in²",
                f"Pc = {now['pressure'] / PSI:,.0f} psi", f"F = {now['thrust'] / LBF:,.0f} lbf",
                f"propellant left = {left_pct:.0f} %")):
            item.set_text(text)

    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=-1,
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "22", "-movflags", "+faststart"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:   # encode locally, then copy to the output folder
        local = Path(tmp) / out.name
        with writer.saving(fig, str(local), dpi=100):
            for t in frames:
                draw(t)
                writer.grab_frame()
        shutil.copyfile(local, out)
    poster = out.with_suffix(".jpg")
    draw(float(np.interp(0.35 * table.web_burnout, result.web, result.t)))
    fig.savefig(poster, dpi=100, pil_kwargs={"quality": 88, "optimize": True})
    plt.close(fig)
    return {"video": str(out), "poster": str(poster), "frames": len(frames), "fps": fps,
            "duration_s": len(frames) / fps}
