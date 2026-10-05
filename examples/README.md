# Examples

Numbered in the order to read them. Each folder has its scripts and a `results/` folder with what
they produced. Meshes and solver output go to `_work/` beside the script (not tracked); set
`ALIGHT_WORK` to put them elsewhere, since the finest meshes are several hundred megabytes.

| # | Folder | What it is |
| --- | --- | --- |
| 01 | [`01_finocyl/`](01_finocyl) | **The worked example.** A 10 in finocyl grain, 60 in long, with six fins over the aft 20 in and an AP/HTPB/Al propellant. `run_sensitivity.py` solves seven element sizes, extrapolates neighbouring pairs, computes an independent fast-marching reference and sizes the nozzle; `run_finocyl.py` produces the ballistics, the burnback video and the quad chart. |
| 02 | [`02_analytic_validation/`](02_analytic_validation) | Tube, BATES and star grains against their closed-form burn area, each swept over its geometry (15 cases). |
| 03 | [`03_nawc_motor6/`](03_nawc_motor6) | NAWC tactical motor no. 6 (cylinder forward, six-slot star aft) against the 3-D burnback simulation and the firing published by Willcox et al., *Journal of Propulsion and Power*, 2007. `extract_reference.py` reads the published curves from the paper's PDF. |
| | [`gmsh/`](gmsh) | burnback-3d's own Gmsh examples for the GUI. |

```shell
python examples/01_finocyl/run_sensitivity.py       # about 40 min, dominated by the 0.05 in mesh
python examples/01_finocyl/run_finocyl.py
python examples/02_analytic_validation/run_validation.py
python examples/03_nawc_motor6/run_comparison.py
```
