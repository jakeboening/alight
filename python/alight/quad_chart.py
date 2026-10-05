"""Quad chart: problem, approach, mesh sensitivity, results."""
from __future__ import annotations

import textwrap
from importlib.metadata import version
from pathlib import Path

import numpy as np

from alight.ballistics import IN, LBF, PSI, BallisticsResult, Nozzle, Propellant
from alight.fmm_check import grid as fmm_grid
from alight.geometry import Grain
from alight.postprocess import BurnbackTable
from alight import style
from alight.sections import Sections
from alight.solve import alight_version
from alight.video import FRONT, PRESSURE, PROPELLANT, THRUST

PANEL = "#f6f7f9"
INLET, OUTLET, SYMMETRY, CUT = "#e0453a", "#8a8f98", "#5b9bd5", "#e9dcc3"


def _panel(subfig, number, title):
    subfig.set_facecolor(PANEL)
    subfig.text(0.015, 0.955, f"{number}  {title}", fontsize=15, fontweight="bold", va="center")


def _dimension(ax, p0, p1, text, offset=(0, 0), **kw):
    ax.annotate("", xy=p0, xytext=p1, arrowprops=dict(arrowstyle="<->", color="k", lw=1.0, shrinkA=0, shrinkB=0))
    mid = (np.asarray(p0) + np.asarray(p1)) / 2 + np.asarray(offset)
    ax.text(*mid, text, ha="center", va="center", fontsize=9,
            bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.9), **kw)


def _problem(subfig, grain: Grain, propellant: Propellant, nozzle: Nozzle):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    _panel(subfig, 1, "Problem")
    R, Rp, w = grain.case_radius, grain.port_radius, grain.fin_width
    solid_color = ListedColormap([PROPELLANT])
    ax = subfig.add_axes([0.01, 0.33, 0.33, 0.56])
    x = np.linspace(-R, R, 500)
    xx, yy = np.meshgrid(x, x)
    solid = np.where((grain.distance(xx, yy, grain.fin_start + grain.fin_length / 2) > 0) & (np.hypot(xx, yy) < R), 1.0, np.nan)
    ax.imshow(solid, extent=(-R, R, -R, R), origin="lower", cmap=solid_color, interpolation="antialiased")
    ax.add_patch(plt.Circle((0, 0), R, fill=False, ec="k", lw=1.6))
    _dimension(ax, (-R, -R - 0.8), (R, -R - 0.8), f"grain OD {grain.case_diameter:g} in")
    _dimension(ax, (0, -Rp), (0, Rp), f"port\nØ{grain.port_diameter:g} in")
    xw = 0.5 * (Rp + grain.tip_center) + 0.35
    _dimension(ax, (xw, -w / 2), (xw, w / 2), f"{w:g} in", offset=(0.75, 0))
    a = np.radians(120)
    tip = np.array([np.cos(a), np.sin(a)])
    ax.annotate(f"fin tip R {grain.fin_tip_radius:g} in*", xy=grain.fin_tip_radius * tip, xytext=(-R - 0.3, R + 0.9),
                fontsize=9, ha="left", va="center", arrowprops=dict(arrowstyle="->", color="k", lw=0.9))
    ax.annotate(f"{R - grain.fin_tip_radius:g} in web*", xy=(grain.fin_tip_radius + 0.5 * (R - grain.fin_tip_radius)) * np.array([0.5, 3**0.5 / 2]),
                xytext=(R - 2.2, R + 0.9), fontsize=9, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="k", lw=0.9))
    ax.set_xlim(-R - 0.5, R + 0.5)
    ax.set_ylim(-R - 1.5, R + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax = subfig.add_axes([0.01, 0.085, 0.98, 0.225])
    z = np.linspace(0, grain.length, 1200)
    zz, xl = np.meshgrid(z, x)
    solid = np.where(grain.distance(xl, np.zeros_like(xl), zz) > 0, 1.0, np.nan)
    ax.imshow(solid, extent=(0, grain.length, -R, R), origin="lower", cmap=solid_color,
              interpolation="antialiased", aspect="equal")
    ax.add_patch(plt.Rectangle((0, -R), grain.length, 2 * R, fill=False, ec="k", lw=1.6))
    _dimension(ax, (0, -R - 2.2), (grain.length, -R - 2.2), f"grain length {grain.length:g} in")
    _dimension(ax, (grain.fin_start, R + 2.0), (grain.fin_end, R + 2.0), f"fins, aft {grain.fin_length:g} in*")
    ax.text(0.0, R + 2.0, "head end (inhibited*)", fontsize=9, ha="left", va="center")
    ax.text(grain.length + 1.0, 0, "nozzle →", fontsize=9, ha="left", va="center")
    ax.text(grain.fin_start / 2, 0, f"plain Ø{grain.port_diameter:g} in port", fontsize=9, ha="center", va="center")
    ax.text((grain.fin_start + grain.fin_end) / 2, 0, "finned section", fontsize=9, ha="center", va="center")
    ax.set_xlim(-22, grain.length + 22)
    ax.set_ylim(-R - 3.6, R + 3.4)
    ax.axis("off")

    rate = propellant.burn_rate_ref / IN
    lines = [
        ("Find", "burn area vs web burned, chamber pressure and thrust\nvs time for a finocyl grain, and show that the answer\ndoes not depend on the mesh."),
        ("Grain", f"{grain.case_diameter:g} in OD × {grain.length:g} in long, Ø{grain.port_diameter:g} in port,\n"
                  f"{grain.n_fins}* fins × {w:g} in wide with semicircular tips*,\nOD case-bonded, both ends inhibited*"),
        ("Propellant", f"AP / HTPB / Al composite (68 / 18 / 14 %)\n"
                       f"ρ = {propellant.density:g} kg/m³,  c* = {propellant.c_star:g} m/s,  γ = {propellant.gamma:g}\n"
                       f"r = {rate:g} in/s at {propellant.p_ref / PSI:g} psi,  n = {propellant.exponent:g}"),
        ("Nozzle", f"throat Ø{nozzle.throat_diameter / IN:.2f} in (sized for 1000 psi peak),\n"
                   f"expansion ratio {nozzle.expansion_ratio:g}, sea level"),
    ]
    y = 0.885
    for head, body in lines:
        subfig.text(0.37, y, head, fontsize=10, fontweight="bold", va="top")
        subfig.text(0.485, y, body, fontsize=9.6, va="top", linespacing=1.3)
        y -= 0.040 + 0.0395 * (body.count("\n") + 1)
    subfig.text(0.015, 0.04, "* Not specified in the request: assumed, adjustable input.\n"
                "Propellant values are representative handbook-level numbers, not a specific qualified formulation.",
                fontsize=8.3, va="center", color="0.3", linespacing=1.3)


def _mesh_view(ax, grain: Grain, mesh_npz, z_range, elev=38, azim=-125):
    """A slice of the solved tetrahedral mesh around the fin start, coloured by boundary type."""
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    data = np.load(mesh_npz)
    points = data["points"] / (IN * 1000)
    tets = data["tets"]
    zc = points[tets][:, :, 2].mean(axis=1)
    tets = tets[(zc > z_range[0]) & (zc < z_range[1])]
    # outer faces of the slice: those that belong to one tetrahedron only
    local = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    faces = np.sort(tets[:, local].reshape(-1, 3), axis=1)
    opposite = tets.reshape(-1)
    n = len(points)
    key = lambda f: (f[:, 0] * n + f[:, 1]) * n + f[:, 2]
    _, first, count = np.unique(key(faces), return_index=True, return_counts=True)
    faces, opposite = faces[first[count == 1]], opposite[first[count == 1]]
    inlet = np.isin(key(faces), key(np.sort(data["triangles"][data["inlet"]], axis=1)))
    tri = points[faces]
    a = grain.wedge_angle
    on = lambda values: np.all(np.abs(values) < 1e-6, axis=1)
    symmetry = on(tri[:, :, 1]) | on(-tri[:, :, 0] * np.sin(a) + tri[:, :, 1] * np.cos(a))
    case = on(np.hypot(tri[:, :, 0], tri[:, :, 1]) - grain.case_radius)
    base = np.array([to_rgb(c) for c in (CUT, OUTLET, SYMMETRY, INLET)])
    colors = base[np.where(inlet, 3, np.where(symmetry, 2, np.where(case, 1, 0)))]

    # grain axis horizontal, wedge mirrored so the fin slot opens upward: (z, x, -y)
    to_view = lambda q: np.stack([q[..., 2], q[..., 0], -q[..., 1]], axis=-1)
    verts = to_view(tri)
    normal = np.cross(verts[:, 1] - verts[:, 0], verts[:, 2] - verts[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    outward = np.einsum("ij,ij->i", normal, verts[:, 0] - to_view(points[opposite])) > 0
    normal[~outward] *= -1
    e, z = np.radians(elev), np.radians(azim)
    camera = np.array([np.cos(z) * np.cos(e), np.sin(z) * np.cos(e), np.sin(e)])
    light = np.array([0.35, -0.55, 0.76])
    visible = normal @ camera > 0   # back faces are hidden, which also keeps the depth sort clean
    shade = 0.62 + 0.38 * np.clip(normal @ light, 0, 1)
    ax.add_collection3d(Poly3DCollection(verts[visible], facecolors=colors[visible] * shade[visible, None],
                                         edgecolors="#1b1f27", linewidths=0.25))
    height = grain.case_radius * np.sin(a)
    ax.set_proj_type("ortho")
    ax.set_xlim(*z_range)
    ax.set_ylim(0, grain.case_radius)
    ax.set_zlim(-height, 0)
    ax.set_box_aspect((z_range[1] - z_range[0], grain.case_radius, height), zoom=0.92)
    ax.view_init(elev=elev, azim=azim)
    ax.patch.set_alpha(0.0)
    ax.set_axis_off()


def _approach(subfig, grain: Grain, mesh_npz, mesh_size, final_pair, counts):
    import matplotlib.pyplot as plt

    _panel(subfig, 2, "Approach")
    steps = [
        ("CAD", "build123d parametric solid → STEP; one 1/12 symmetry wedge (half a fin)."),
        ("Mesh", "Gmsh tetrahedra. Boundaries: inlet = port and fin faces, outlet = case wall and\nend faces, symmetry = wedge planes."),
        ("Burnback", r"alight solves $|\nabla u| = 1$, where u is the web burned when the flame arrives" "\n(the burnback-3d kernels behind a command line)."),
        ("Burn area", "A(w) = area of the surface u = w, by marching tetrahedra, × 12."),
        ("Mesh error", "The solver is first-order accurate (numerical diffusion), so two meshes are\n"
                       f"Richardson-extrapolated to zero element size (final pair: {final_pair[0]:g} and {final_pair[1]:g} in)."),
        ("Check", "Independent fast-marching distance field (scikit-fmm) with marching-cubes\nsurfaces (scikit-image) on a Cartesian grid."),
        ("Ballistics", r"0-D chamber:  $dP/dt = (RT/V)\,[\rho\,A(w)\,r - P A_t/c^*]$,   $r = a P^n$,   $F = C_F\,P A_t$"),
    ]
    y = 0.885
    for i, (head, body) in enumerate(steps, 1):
        subfig.text(0.022, y - 0.004, f"{i}", fontsize=9, fontweight="bold", va="top", color="white",
                    bbox=dict(boxstyle="circle,pad=0.22", fc="#33415c", ec="none"))
        subfig.text(0.058, y, head, fontsize=10, fontweight="bold", va="top")
        subfig.text(0.20, y, body, fontsize=9.6, va="top", linespacing=1.25)
        y -= 0.026 + 0.0355 * (body.count("\n") + 1)
    ax = subfig.add_axes([0.47, -0.14, 0.56, 0.56], projection="3d")
    _mesh_view(ax, grain, mesh_npz, (grain.fin_start - 3.5, grain.fin_start + 4.5))
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="k", lw=0.4) for c in (INLET, OUTLET, SYMMETRY, CUT)]
    subfig.legend(handles, ["inlet: initial burning surface", "outlet: case wall (and inhibited ends)",
                            "symmetry plane", "cut through the mesh (display only)"],
                  loc="lower left", bbox_to_anchor=(0.012, 0.155), frameon=False, fontsize=9, handlelength=1.2)
    subfig.text(0.015, 0.125, f"8 in slice of the solved wedge at the fin start,\n{mesh_size:g} in elements shown "
                f"(finest mesh: {counts['tetrahedra'] / 1e6:.1f} M tetrahedra)", fontsize=9, va="center", linespacing=1.3)
    subfig.text(0.015, 0.04, f"Solver: {alight_version()}, github.com/jakeboening/alight, a fork of burnback-3d "
                "(codeberg.org/iff/burnback-3d, AGPL-3.0).\n"
                "Not modelled: erosive burning, axial pressure drop, throat erosion, ignition transient.",
                fontsize=8.3, color="0.3", va="center", linespacing=1.3)


def _sensitivity(subfig, grain, rows, tables, final_key, reference_key, acceptance=1.0):
    import matplotlib.pyplot as plt

    _panel(subfig, 3, "Mesh sensitivity")
    gs = subfig.add_gridspec(1, 2, left=0.085, right=0.985, top=0.86, bottom=0.445, wspace=0.26)
    raw = sorted((r for r in rows if r["kind"] == "raw"), key=lambda r: -r["size_in"])
    ext = sorted((r for r in rows if r["kind"] == "extrapolated"), key=lambda r: -r["size_in"])
    final = ext[-1]
    reference = next(r for r in rows if r["kind"] == "reference-fmm" and f"dx{r['size_in']:g}" in reference_key)
    shades = plt.cm.Blues(np.linspace(0.3, 0.95, len(raw)))

    ax = subfig.add_subplot(gs[0])
    inset = ax.inset_axes([0.37, 0.13, 0.33, 0.36])
    for axis in (ax, inset):
        for row, shade in zip(raw, shades):
            table = tables[f"raw_h{row['size_in']:g}"]
            axis.plot(table.web, table.burn_area, color=shade, lw=1.1,
                      label=f"raw, h = {row['size_in']:g} in" if row in (raw[0], raw[-1]) else None)
        axis.plot(tables[final_key].web, tables[final_key].burn_area, color=FRONT, lw=2.0, label="extrapolated, h → 0 (final)")
        axis.plot(tables[reference_key].web, tables[reference_key].burn_area, color="k", lw=1.1, ls="--",
                  label="fast-marching reference")
    peak = tables[final_key].web[tables[final_key].burn_area.argmax()]
    inset.set_xlim(peak - 0.15, peak + 0.25)
    inset.set_ylim(0.9 * tables[final_key].burn_area.max(), 1.02 * tables[final_key].burn_area.max())
    inset.tick_params(labelsize=7)
    inset.grid(True, alpha=0.25)
    ax.indicate_inset_zoom(inset, edgecolor="0.4")
    ax.set_xlabel("web burned (in)")
    ax.set_ylabel("burn area (in²)")
    ax.set_xlim(0, 3.2)
    ax.set_ylim(0, 1.42 * tables[final_key].burn_area.max())
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7.6, loc="upper center", ncol=2, frameon=False, columnspacing=1.0, borderaxespad=0.2)
    ax.set_title("Burn area vs web", fontsize=10)

    ax = subfig.add_subplot(gs[1])
    quantities = (("peak_pressure_psi", "peak pressure", PRESSURE, "o"), ("burn_time_s", "burn time", "#2a9d5c", "s"),
                  ("total_impulse_lbf_s", "total impulse", THRUST, "^"))
    ax.axhspan(-acceptance, acceptance, color="0.87", lw=0)
    for key, name, color, marker in quantities:
        pct = lambda row: 100 * (row[key] / final[key] - 1)
        ax.plot([r["size_in"] for r in raw], [pct(r) for r in raw], color=color, marker=marker, mfc="white", ms=5, lw=1.2)
        ax.plot([r["size_in"] for r in ext], [pct(r) for r in ext], color=color, marker=marker, ms=5, lw=1.2, ls="--")
        ax.plot([0], [pct(reference)], color="k", marker=marker, ms=7, mfc=color, ls="none", clip_on=False, zorder=5)
    line = plt.Line2D
    handles = [line([], [], color=c, lw=2) for _, _, c, _ in quantities] + [
        line([], [], color="0.3", marker="o", mfc="white", ms=5, lw=1.2),
        line([], [], color="0.3", marker="o", ms=5, lw=1.2, ls="--"),
        line([], [], color="k", marker="o", mfc="0.6", ms=7, ls="none"),
        plt.Rectangle((0, 0), 1, 1, fc="0.87")]
    labels = [q[1] for q in quantities] + ["raw alight", "extrapolated pair (at finer h)",
                                           "fast-marching ref. (at h = 0)", f"±{acceptance:g} % band"]
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlim(0, 1.05 * raw[0]["size_in"])
    ax.set_xlabel("element size h (in)")
    ax.set_ylabel("difference from final (%)")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(-12.5, 5.5)
    ax.legend(handles, labels, fontsize=7.4, loc="lower left", ncol=2, frameon=False, columnspacing=0.8,
              borderpad=0.2, labelspacing=0.3)
    ax.set_title("Ballistic results vs element size", fontsize=10)

    def spread(group, key):
        values = [r[key] for r in group]
        return 100 * (max(values) - min(values)) / final[key]

    def step(group, key):
        return 100 * abs(group[-1][key] / group[-2][key] - 1)

    keys = [q[0] for q in quantities]
    usable = ext[2:] if len(ext) > 3 else ext
    gx, gy, gz, _ = fmm_grid(grain, reference["size_in"])
    cells = len(gx) * len(gy) * len(gz)
    exact = next(r for r in rows if r["kind"] == "reference-exact")
    bullets = [
        f"Raw alight results drift with the mesh: peak pressure spans {spread(raw, keys[0]):.1f} % over "
        f"h = {raw[0]['size_in']:g} to {raw[-1]['size_in']:g} in and still moves {step(raw, keys[0]):.2f} % at the last "
        f"refinement. The error falls linearly with h (first order).",
        f"Extrapolating two meshes removes it: for every pair from {usable[0]['label'].replace(' ', '')} in down, peak "
        f"pressure, burn time and total impulse stay within {max(spread(usable, k) for k in keys):.2f} %, and the last two "
        f"pairs differ by at most {max(step(ext, k) for k in keys):.2f} %.",
        f"The final result differs from the independent fast-marching reference by "
        f"{100 * abs(final[keys[0]] / reference[keys[0]] - 1):.2f} % in peak pressure, "
        f"{100 * abs(final[keys[1]] / reference[keys[1]] - 1):.2f} % in burn time and "
        f"{100 * abs(final[keys[2]] / reference[keys[2]] - 1):.2f} % in total impulse. CFL and convergence "
        f"tolerance do not change the converged field.",
        f"Reference case source: computed in this study, not taken from literature. scikit-fmm {version('scikit-fmm')} "
        f"marches the distance from the initial port, voxelised from the grain definition on a {reference['size_in']:g} in "
        f"Cartesian grid ({cells / 1e6:.0f} M nodes, one quadrant); scikit-image {version('scikit-image')} marching cubes "
        f"gives the burn area. The closed-form distance on the same grid agrees with it within "
        f"{100 * abs(exact[keys[0]] / reference[keys[0]] - 1):.2f} % in peak pressure.",
    ]
    text = "\n".join("•  " + textwrap.fill(b, 138, subsequent_indent="    ") for b in bullets)
    subfig.text(0.015, 0.34, text, fontsize=8.7, va="top", linespacing=1.3)


def _results(subfig, grain: Grain, sections: Sections, table: BurnbackTable, result: BallisticsResult):
    import matplotlib.pyplot as plt

    _panel(subfig, 4, "Results")
    R = grain.case_radius
    levels = np.arange(0.0, table.web_burnout + 0.25, 0.25)
    cmap = plt.cm.YlOrRd
    propellant_only = lambda field: np.ma.masked_where(~(field > 0), field)

    ax = subfig.add_axes([0.05, 0.31, 0.34, 0.52])
    field = propellant_only(sections.fin)
    filled = ax.contourf(sections.x, sections.x, field, levels=levels, cmap=cmap, extend="max")
    ax.contour(sections.x, sections.x, field, levels=levels[1:], colors="k", linewidths=0.35)
    ax.add_patch(plt.Circle((0, 0), R, fill=False, ec="k", lw=1.4))
    ax.set_xlim(-1.03 * R, 1.03 * R)
    ax.set_ylim(-1.03 * R, 1.03 * R)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Burning surface every 0.25 in of web (z = {sections.z_fin:g} in)", fontsize=9.5, pad=3)
    cax = subfig.add_axes([0.035, 0.37, 0.012, 0.40])
    bar = subfig.colorbar(filled, cax=cax, ticks=np.arange(0, table.web_burnout + 0.1, 1.0))
    cax.yaxis.set_ticks_position("left")
    cax.set_title("web (in)", fontsize=8, pad=4)
    bar.ax.tick_params(labelsize=8)

    ax = subfig.add_axes([0.02, 0.10, 0.40, 0.15])
    field = propellant_only(sections.longitudinal)
    ax.contourf(sections.z, sections.x, field, levels=levels, cmap=cmap, extend="max")
    ax.contour(sections.z, sections.x, field, levels=levels[1::2], colors="k", linewidths=0.3)
    ax.add_patch(plt.Rectangle((0, -R), grain.length, 2 * R, fill=False, ec="k", lw=1.2))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Longitudinal section through a fin pair", fontsize=10, pad=3)

    ax = subfig.add_axes([0.525, 0.50, 0.365, 0.35])
    ax.plot(result.t, result.pressure / PSI, color=PRESSURE, lw=1.8)
    ax.set_ylabel("chamber pressure (psi)", color=PRESSURE)
    ax.set_xlabel("time (s)", labelpad=1)
    ax.set_xlim(0, result.t[-1])
    ax.set_ylim(0, 1.12 * result.peak_pressure / PSI)
    ax.grid(True, alpha=0.25)
    twin = ax.twinx()
    twin.plot(result.t, result.thrust / LBF, color=THRUST, lw=1.8)
    twin.set_ylabel("thrust (lbf)", color=THRUST)
    twin.set_ylim(0, 1.12 * result.thrust.max() / LBF)
    ax.set_title("Chamber pressure and thrust", fontsize=10)

    s = result.summary()
    peak_web = table.web[table.burn_area.argmax()]
    cells = [
        ("Peak / mean pressure", f"{s['peak_pressure_psi']:.0f} / {s['mean_pressure_psi']:.0f} psi"),
        ("Peak thrust", f"{s['peak_thrust_lbf']:,.0f} lbf"),
        ("Burn time (10 % to 10 %)", f"{s['burn_time_s']:.2f} s"),
        ("Total impulse", f"{s['total_impulse_lbf_s']:,.0f} lbf·s  ({s['total_impulse_kN_s']:.0f} kN·s)"),
        ("Specific impulse, sea level", f"{s['specific_impulse_s']:.0f} s"),
        ("Propellant mass", f"{s['propellant_mass_kg']:.1f} kg  ({s['propellant_mass_kg'] / 0.45359237:.0f} lbm)"),
        ("Burn area, initial / peak", f"{table.burn_area[0]:,.0f} / {table.burn_area.max():,.0f} in²"),
        ("Web at fin burn-through / burnout", f"{peak_web:.2f} / {table.web_burnout:.2f} in"),
    ]
    for i, (name, value) in enumerate(cells):
        y = 0.385 - 0.0375 * i
        subfig.text(0.47, y, name, fontsize=9.2, color="0.3", va="center")
        subfig.text(0.745, y, value, fontsize=9.8, fontweight="bold", va="center")
    subfig.text(0.47, 0.085, "Burn area peaks when the fin tips reach the case wall, falls as the\n"
                "fin slots burn out, then rises again in the plain port until burnout.", fontsize=8.9, va="top",
                linespacing=1.3)


def render(grain: Grain, propellant: Propellant, nozzle: Nozzle, rows, tables, final_key, reference_key,
           final_pair, sections: Sections, result: BallisticsResult, mesh_npz, mesh_size, out) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style.apply(9.5)
    plt.rcParams["axes.spines.top"] = False
    fig = plt.figure(figsize=(16, 9), dpi=150)
    fig.text(0.012, 0.976, "Finocyl grain burnback and internal ballistics", fontsize=19, fontweight="bold", va="center")
    fig.text(0.988, 0.976, f"{grain.case_diameter:g} in OD × {grain.length:g} in  ·  Ø{grain.port_diameter:g} in port  ·  "
             f"{grain.n_fins} fins × {grain.fin_width:g} in  ·  {propellant.name}", fontsize=11, va="center", ha="right", color="0.25")
    outer = fig.add_gridspec(3, 2, height_ratios=[0.075, 1, 1], wspace=0.012, hspace=0.022)
    quads = np.array([[fig.add_subfigure(outer[i + 1, j]) for j in range(2)] for i in range(2)])
    for sub in quads.ravel():
        sub.set_edgecolor("0.75")
        sub.set_linewidth(0.8)
    finest = max((r for r in rows if r["kind"] == "raw"), key=lambda r: r["tetrahedra"])
    _problem(quads[0, 0], grain, propellant, nozzle)
    _approach(quads[0, 1], grain, mesh_npz, mesh_size, final_pair, finest)
    _sensitivity(quads[1, 0], grain, rows, tables, final_key, reference_key)
    _results(quads[1, 1], grain, sections, tables[final_key], result)
    outputs = []
    for suffix in (".png", ".pdf"):
        path = Path(out).with_suffix(suffix)
        fig.savefig(path, dpi=150)
        outputs.append(str(path))
    plt.close(fig)
    return outputs
