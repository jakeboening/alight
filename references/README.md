# References

Sources used by the examples and the report.

| Source | Used for | File |
| --- | --- | --- |
| F. S. Blomshield, *Pulsed Motor Firings*, NAWCWD TP 8444, Naval Air Warfare Center Weapons Division, China Lake, CA, August 2000. DTIC ADA382239. Approved for public release; distribution is unlimited. | Propellant data of NAWC motor no. 6 ([example 03](../examples/03_nawc_motor6)) | [PDF](Blomshield_2000_Pulsed_Motor_Firings_NAWCWD_TP_8444_ADA382239.pdf) |
| M. A. Willcox, M. Q. Brewster, K. C. Tang, D. S. Stewart and I. Kuznetsov, "Solid Rocket Motor Internal Ballistics Simulation Using Three-Dimensional Grain Burnback", *Journal of Propulsion and Power*, Vol. 23, No. 3, 2007, pp. 575–584. doi:10.2514/1.22971 | Grain geometry, published 0-D simulation and measured pressure of NAWC motor no. 6 | Not redistributed here: the article is copyrighted. Open copy from the University of Illinois: <https://www.ideals.illinois.edu/items/14484> |
| D. E. Coats et al., *A Computer Program for the Prediction of Solid Propellant Rocket Motor Performance* (SPP), AFRPL-TR-75-36, Vols. I–II, 1975. DTIC ADA015140, ADA015141. | Background: the Solid Performance Program. Its motor comparisons give pressure traces but not the grain inputs, so they could not be reproduced. | DTIC |
| iff, *burnback-3d*, <https://codeberg.org/iff/burnback-3d> | The solver this repository forks | |

`examples/03_nawc_motor6/extract_reference.py` reads the pressure curves of Figure 10 out of the
Willcox et al. PDF; the extracted numbers are in `examples/03_nawc_motor6/reference/`.
