"""Checks of the geometry, post-processing, ballistics and the burnback-3d pipeline."""
import math

import numpy as np
import pytest

from alight.ballistics import AP_HTPB_AL, IN, PSI, Nozzle, ThroatErosion, equilibrium_pressure, simulate, size_throat
from alight.geometry import FinocylGrain, Grain, StarGrain, TaperedFinocylGrain, TubeGrain
from alight.postprocess import extrapolate_tables, iso_area, make_table
from alight.solve import make_cad, run_case


def test_closed_form_geometry_matches_numerical_integration():
    g = FinocylGrain()
    x = np.linspace(-5, 5, 2001)
    xx, yy = np.meshgrid(x, x)
    port = (g.distance(xx, yy, 50.0) <= 0).sum() * (x[1] - x[0]) ** 2
    assert port == pytest.approx(math.pi * g.port_radius**2 + g.n_fins * g.fin_section_area, rel=2e-3)
    # distance field: on the axis of the plain section it is minus the port radius, at the case it is the web
    assert g.distance(0.0, 0.0, 10.0) == pytest.approx(-g.port_radius)
    assert g.distance(g.case_radius, 0.0, 10.0) == pytest.approx(g.case_radius - g.port_radius)
    assert g.distance(g.case_radius, 0.0, 50.0) == pytest.approx(g.case_radius - g.fin_tip_radius)


def test_invalid_grain_rejected():
    with pytest.raises(ValueError):
        FinocylGrain(n_fins=12, fin_width=1.2)   # fins overlap at the port
    with pytest.raises(ValueError):
        FinocylGrain(fin_tip_radius=5.5)


def test_bates_closed_form():
    g = TubeGrain(case_diameter=10.0, port_diameter=4.0, length=10.0, ends_inhibited=False)
    assert g.web == pytest.approx(3.0)
    assert g.burn_area(0.0) == pytest.approx(2 * math.pi * 2 * 10 + 2 * math.pi * (25 - 4))
    # volume burned is the integral of the burn area over the web
    web = np.linspace(0, g.web, 20001)
    assert np.trapezoid(g.burn_area(web), web) == pytest.approx(g.propellant_volume, rel=1e-6)
    assert g.distance(3.0, 0.0, 0.4) == pytest.approx(0.4)      # nearest surface is the head-end face


def test_star_phases_join_and_match_the_cad(tmp_path):
    g = StarGrain()
    switch = g.skeleton_radius * math.sin(g.flank_angle) / math.cos(g.half_point_angle) - g.fillet_radius
    assert 0 < switch < g.web
    assert g.perimeter(switch - 1e-9) == pytest.approx(float(g.perimeter(switch + 1e-9)), rel=1e-6)
    cad = make_cad(g, tmp_path)
    assert cad["grain_volume_in3"] == pytest.approx(g.propellant_volume, rel=1e-5)
    assert cad["port_volume_in3"] == pytest.approx(g.initial_port_volume, rel=1e-5)
    with pytest.raises(ValueError):
        StarGrain(point_angle=30.0)     # flanks cannot meet at so sharp a point


def test_grain_round_trips_through_json(tmp_path):
    grains = [TubeGrain(ends_inhibited=False), StarGrain(n_points=7), FinocylGrain(),
              TaperedFinocylGrain(bore=((0, 1.0), (10, 1.2)), slots=((2, 1.5), (8, 1.8)))]
    for i, grain in enumerate(grains):
        grain.to_json(tmp_path / f"{i}.json")
        assert Grain.from_json(tmp_path / f"{i}.json") == grain


def test_tapered_finocyl_meshes_with_a_burning_aft_end(tmp_path):
    grain = TaperedFinocylGrain(case_diameter=5.0, length=6.0, bore=((0, 1.2), (2, 0.7), (6, 0.7)),
                                slots=((1.5, 0.9), (3, 1.5), (6, 1.5)), n_slots=6, slot_width=0.4,
                                head_end_inhibited=True, aft_end_inhibited=False)
    table = run_case(grain, tmp_path / "h0.25", 0.25).table(100)
    assert table.propellant_volume + table.initial_port_volume == pytest.approx(grain.case_area * grain.length, rel=1e-2)
    assert table.burn_area[0] > 2 * math.pi * 0.7 * 4       # more than the plain bore: slots and aft face burn
    # the slots and the wide head-end bore leave less than the full radial web anywhere
    assert 1.3 < table.web_burnout < grain.web


def test_iso_area_of_linear_field_in_a_cube():
    # unit cube split into 6 tetrahedra, u = x: every iso-surface is a unit square
    points = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
    index = lambda i, j, k: 4 * i + 2 * j + k
    paths = [((1, 0, 0), (1, 1, 0)), ((1, 0, 0), (1, 0, 1)), ((0, 1, 0), (1, 1, 0)),
             ((0, 1, 0), (0, 1, 1)), ((0, 0, 1), (1, 0, 1)), ((0, 0, 1), (0, 1, 1))]
    tets = np.array([[index(0, 0, 0), index(*a), index(*b), index(1, 1, 1)] for a, b in paths])
    assert iso_area(points, tets, points[:, 0], [0.25, 0.5, 0.9]) == pytest.approx([1.0, 1.0, 1.0])


def test_extrapolation_removes_a_first_order_web_lag():
    web = np.linspace(0, 3, 601)
    volume = math.pi * (5**2 - 2**2) * 60
    exact = make_table(web, 2 * math.pi * (2 + web) * 60, 100.0, volume)

    def lagged(h):  # the same surfaces reached (1 + 0.3 h) times later
        return make_table(web * (1 + 0.3 * h), exact.burn_area / (1 + 0.3 * h), 100.0, volume)

    result = extrapolate_tables(lagged(0.2), lagged(0.1), 2.0)
    assert result.web_burnout == pytest.approx(3.0, rel=1e-3)
    assert result.area_at(1.5) == pytest.approx(exact.area_at(1.5), rel=2e-3)


def test_ballistics_plateau_and_mass_balance():
    web = np.linspace(0, 2, 201)
    table = make_table(web, np.full_like(web, 1200.0), 900.0, 2400.0)
    table.burn_area[-1] = 0.0
    nozzle = Nozzle(2.5 * IN)
    result = simulate(table, AP_HTPB_AL, nozzle)
    plateau = result.pressure[len(result.t) // 2]
    assert plateau == pytest.approx(equilibrium_pressure(1200 * IN**2, AP_HTPB_AL, nozzle), rel=1e-2)
    assert result.expelled_mass == pytest.approx(result.propellant_mass, rel=5e-3)
    assert simulate(table, AP_HTPB_AL, size_throat(table, AP_HTPB_AL, 800 * PSI)).peak_pressure == pytest.approx(800 * PSI, rel=1e-4)


@pytest.fixture(scope="module")
def cylinder_tables(tmp_path_factory):
    """Plain cylindrical port through CAD, Gmsh and burnback-3d at two element sizes."""
    root = tmp_path_factory.mktemp("cylinder")
    grain = FinocylGrain(n_fins=0, length=12.0)
    return grain, {h: run_case(grain, root / f"h{h:g}", h).table(200) for h in (0.4, 0.2)}


def test_cylinder_matches_analytic(cylinder_tables):
    grain, tables = cylinder_tables
    table = extrapolate_tables(tables[0.4], tables[0.2], 2.0)
    web = np.linspace(0.1, 2.8, 28)
    exact = 2 * math.pi * (grain.port_radius + web) * grain.length
    assert table.area_at(web) == pytest.approx(exact, rel=0.02)
    assert table.web_burnout == pytest.approx(grain.case_radius - grain.port_radius, rel=0.01)
    assert table.burned_volume[-1] == pytest.approx(grain.propellant_volume, rel=2e-3)
    # the raw solution lags (first-order error) and the finer mesh lags less
    assert tables[0.4].web_burnout > tables[0.2].web_burnout > table.web_burnout


def test_wedge_matches_full_grain(tmp_path):
    grain = FinocylGrain(length=6.0, fin_start=3.0, fin_length=3.0)
    wedge = run_case(grain, tmp_path / "wedge", 0.4).table(100)
    full = run_case(grain, tmp_path / "full", 0.4, sector="full").table(100)
    assert wedge.burn_area[0] == pytest.approx(full.burn_area[0], rel=5e-3)
    assert wedge.web_burnout == pytest.approx(full.web_burnout, rel=0.01)
    web = np.linspace(0.2, 2.6, 13)
    assert wedge.area_at(web) == pytest.approx(full.area_at(web), rel=0.05)


def _tube_stations(port_radius, case_radius=2.5, length=40.0, n=40):
    from alight.ballistics1d import Stations

    web = np.linspace(0.0, case_radius - port_radius, 201)
    radius = port_radius + web
    perimeter = np.tile(2 * math.pi * radius, (n, 1))
    perimeter[:, -1] = 0.0
    return Stations(z=(np.arange(n) + 0.5) * length / n, dz=np.full(n, length / n), web=web, perimeter=perimeter,
                    port_area=np.tile(math.pi * radius**2, (n, 1)), case_area=math.pi * case_radius**2)


def test_1d_ballistics_reduces_to_0d_in_a_wide_port():
    from alight.ballistics1d import simulate_1d

    stations = _tube_stations(port_radius=1.5, length=10.0, n=20)
    nozzle = Nozzle(0.6 * IN)                       # port-to-throat area ratio 25: negligible port velocity
    lumped = simulate(stations.burn_table, AP_HTPB_AL, nozzle)
    resolved = simulate_1d(stations, AP_HTPB_AL, nozzle)
    for t in (0.5, 1.0, 1.5):
        assert np.interp(t, resolved.t, resolved.head_pressure) == pytest.approx(np.interp(t, lumped.t, lumped.pressure), rel=0.02)
    assert resolved.expelled_mass == pytest.approx(lumped.propellant_mass, rel=0.01)
    assert resolved.max_mach.max() < 0.1


def test_erosive_burning_raises_pressure_in_a_narrow_port():
    from alight.ballistics1d import LenoirRobillard, simulate_1d

    stations = _tube_stations(port_radius=0.45)
    nozzle = Nozzle(0.75 * IN)                      # port-to-throat area ratio 1.44 at ignition
    erosion = LenoirRobillard.from_properties(AP_HTPB_AL, flame_temperature=3400.0, viscosity=9e-5,
                                              solid_specific_heat=1500.0, surface_temperature=1000.0)
    plain = simulate_1d(stations, AP_HTPB_AL, nozzle)
    erosive = simulate_1d(stations, AP_HTPB_AL, nozzle, erosive=erosion)
    assert plain.head_pressure[0] > plain.aft_pressure[0] * 1.1         # pressure drop along the port
    assert erosive.head_pressure[0] > 1.2 * plain.head_pressure[0]      # erosion adds mass early
    assert erosive.rate[0, -1] > erosive.rate[0, 0]                     # and most near the nozzle end
    # erosion fades as the port opens, and the same propellant is burned sooner
    late = 0.8 * erosive.t[-1]
    assert erosive.max_rate[np.abs(erosive.t - late).argmin()] < 0.75 * erosive.max_rate[0]
    assert erosive.expelled_mass == pytest.approx(plain.expelled_mass, rel=0.02)
    assert erosive.t[-1] < plain.t[-1]


def test_throat_erosion_lowers_pressure_and_conserves_mass():
    from alight.ballistics1d import simulate_1d

    web = np.linspace(0, 2, 201)
    table = make_table(web, np.full_like(web, 1200.0), 900.0, 2400.0)
    table.burn_area[-1] = 0.0
    nozzle = Nozzle(2.5 * IN)
    erosion = ThroatErosion(rate_ref=0.3e-3)                 # 0.3 mm/s at 6.9 MPa
    fixed, eroding = simulate(table, AP_HTPB_AL, nozzle), simulate(table, AP_HTPB_AL, nozzle, throat_erosion=erosion)
    assert eroding.throat_diameter[0] == pytest.approx(2.5 * IN)
    growth = np.trapezoid([erosion.rate(p) for p in eroding.pressure], eroding.t)
    assert eroding.throat_diameter[-1] - 2.5 * IN == pytest.approx(2 * growth, rel=0.02)
    late = 0.9 * fixed.t[-1]
    assert np.interp(late, eroding.t, eroding.pressure) < 0.95 * np.interp(late, fixed.t, fixed.pressure)
    assert eroding.t[-1] > fixed.t[-1]                       # lower pressure burns slower
    assert eroding.expelled_mass == pytest.approx(eroding.propellant_mass, rel=5e-3)

    stations = _tube_stations(port_radius=1.5, length=10.0, n=20)
    small = Nozzle(0.6 * IN)
    plain, worn = simulate_1d(stations, AP_HTPB_AL, small), simulate_1d(stations, AP_HTPB_AL, small, throat_erosion=erosion)
    assert worn.throat_diameter[-1] > worn.throat_diameter[0]
    assert np.interp(1.5, worn.t, worn.head_pressure) < np.interp(1.5, plain.t, plain.head_pressure)
    assert worn.expelled_mass == pytest.approx(plain.expelled_mass, rel=0.01)
