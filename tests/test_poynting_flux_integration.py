import warnings

import pytest
import torch

import solwa
from solwa.utils import poynting_flux


class ConstantPoyntingSimulation:
    _dtype = torch.complex128
    _device = torch.device("cpu")

    def __init__(self, L=(1.0, 1.0)):
        self.L = L

    def poynting_xy(self, layer, x_axis, y_axis, z_prop):
        del layer, z_prop
        shape = (x_axis.numel(), y_axis.numel())
        sx = torch.zeros(shape, dtype=x_axis.dtype, device=x_axis.device)
        sy = torch.zeros(shape, dtype=x_axis.dtype, device=x_axis.device)
        sz = torch.ones(shape, dtype=x_axis.dtype, device=x_axis.device)
        return sx, sy, sz


class ConstantPoyntingSimulation32(ConstantPoyntingSimulation):
    _dtype = torch.complex64


def _geometry(n=100, L=1.0, dtype=torch.float64):
    geo = solwa.geometry(
        Lx=L,
        Ly=L,
        nx=n,
        ny=n,
        dtype=dtype,
        device=torch.device("cpu"),
    )
    geo.grid()
    return geo


def test_poynting_flux_integrates_full_cell_centered_grid_area():
    geo = _geometry()
    sim = ConstantPoyntingSimulation()

    flux = poynting_flux(
        sim,
        layer=0,
        x_cells=geo.x,
        y_cells=geo.y,
        z_prop=0.0,
    )

    assert torch.isclose(flux, torch.tensor(1.0, dtype=geo.dtype), atol=1e-12)


def test_poynting_flux_integrates_selected_subset_of_cells():
    geo = _geometry()
    sim = ConstantPoyntingSimulation()

    x_cells = geo.x[20:50]
    y_cells = geo.y[30:70]
    flux = poynting_flux(
        sim,
        layer=0,
        x_cells=x_cells,
        y_cells=y_cells,
        z_prop=0.0,
    )

    expected_area = torch.tensor(0.3 * 0.4, dtype=geo.dtype)
    assert torch.isclose(flux, expected_area, atol=1e-12)


def test_poynting_flux_integrates_actual_points_with_trapezoid():
    sim = ConstantPoyntingSimulation()
    x_points = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    y_points = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flux = poynting_flux(
            sim,
            layer=0,
            x_points=x_points,
            y_points=y_points,
            z_prop=0.0,
        )

    assert torch.isclose(flux, torch.tensor(1.0, dtype=torch.float64), atol=1e-12)


def test_rcwa_wrapper_forwards_cell_coordinates():
    geo = _geometry()
    sim = ConstantPoyntingSimulation()

    flux = solwa.rcwa.poynting_flux(
        sim,
        layer_num=0,
        x_cells=geo.x,
        y_cells=geo.y,
        z_prop=0.0,
    )

    assert torch.isclose(flux, torch.tensor(1.0, dtype=geo.dtype), atol=1e-12)


def test_poynting_flux_accepts_float32_cell_grid_with_rounding():
    # float32 centers of a 300-wide cell with 1000 cells carry spacing
    # roundoff of ~2e-5, which must not be mistaken for uneven cells.
    geo = _geometry(n=1000, L=300.0, dtype=torch.float32)
    sim = ConstantPoyntingSimulation32()

    flux = poynting_flux(sim, layer=0, x_cells=geo.x, y_cells=geo.y)

    assert torch.isclose(flux, torch.tensor(300.0**2), rtol=1e-5)


def test_poynting_flux_accepts_float32_cells_in_complex128_simulation():
    # Promoting float32 centers to float64 keeps their float32 rounding, so
    # the spacing check must not apply a float64 tolerance to them.
    geo = _geometry(dtype=torch.float32)
    sim = ConstantPoyntingSimulation()

    flux = poynting_flux(sim, layer=0, x_cells=geo.x, y_cells=geo.y)

    assert flux.dtype == torch.float64
    assert torch.isclose(flux, torch.tensor(1.0, dtype=torch.float64), rtol=1e-6)


@pytest.mark.parametrize(
    "x_cells",
    [
        [1000000.5, 1000001.5, 1000001.5],
        [1000000.0, 1000001.0, 1000000.5],
    ],
    ids=["duplicate", "backward"],
)
def test_poynting_flux_rejects_non_increasing_float32_cell_centers(x_cells):
    sim = ConstantPoyntingSimulation32()
    x_cells = torch.tensor(x_cells, dtype=torch.float32)
    y_cells = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32)

    with pytest.raises(ValueError, match="strictly increasing"):
        poynting_flux(sim, layer=0, x_cells=x_cells, y_cells=y_cells)


def test_poynting_flux_rejects_float32_cells_too_coarse_to_check_spacing():
    # At this magnitude float32 rounding is comparable to the step, so the
    # 1 vs 2 gaps cannot be told apart from rounding and must be rejected.
    sim = ConstantPoyntingSimulation32()
    x_cells = torch.tensor([1000000.5, 1000001.5, 1000003.5], dtype=torch.float32)
    y_cells = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32)

    with pytest.raises(ValueError, match="precision"):
        poynting_flux(sim, layer=0, x_cells=x_cells, y_cells=y_cells)


@pytest.mark.parametrize(
    "x_cells",
    [
        [0.5, float("nan"), 2.5],
        [0.5, 1.5, float("inf")],
        [float("-inf"), 1.5, 2.5],
    ],
    ids=["nan", "inf", "neg-inf"],
)
def test_poynting_flux_rejects_non_finite_cell_centers(x_cells):
    sim = ConstantPoyntingSimulation()
    x_cells = torch.tensor(x_cells, dtype=torch.float64)
    y_cells = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64)

    with pytest.raises(ValueError, match="finite"):
        poynting_flux(sim, layer=0, x_cells=x_cells, y_cells=y_cells)


def test_poynting_flux_rejects_points_lost_in_simulation_precision():
    # Casting these x coordinates to float32 would merge them, so the fields
    # would be sampled away from the quadrature nodes.
    sim = ConstantPoyntingSimulation32()
    x_points = torch.tensor([1e8, 1e8 + 1.0], dtype=torch.float64)
    y_points = torch.tensor([0.0, 1.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="precision"):
        poynting_flux(sim, layer=0, x_points=x_points, y_points=y_points)


def test_poynting_flux_accepts_float64_points_in_complex64_simulation():
    sim = ConstantPoyntingSimulation32()
    points = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)

    flux = poynting_flux(sim, layer=0, x_points=points, y_points=points)

    assert torch.isclose(flux, torch.tensor(1.0, dtype=flux.dtype))


def test_poynting_flux_warns_for_float32_full_cell_grid():
    geo = _geometry(n=1000, L=300.0, dtype=torch.float32)
    sim = ConstantPoyntingSimulation32(L=(300.0, 300.0))

    with pytest.warns(UserWarning, match="x_cells/y_cells"):
        poynting_flux(sim, layer=0, x_points=geo.x, y_points=geo.y)


def test_poynting_flux_rejects_nonuniform_small_scale_cell_centers():
    sim = ConstantPoyntingSimulation()
    x_cells = torch.tensor([1e-9, 2e-9, 4e-9], dtype=torch.float64)
    y_cells = torch.tensor([1e-9, 2e-9, 3e-9], dtype=torch.float64)

    with pytest.raises(ValueError, match="equally spaced"):
        poynting_flux(sim, layer=0, x_cells=x_cells, y_cells=y_cells)


def test_poynting_flux_warns_when_points_look_cell_centered():
    geo = _geometry()
    sim = ConstantPoyntingSimulation()

    with pytest.warns(UserWarning, match="x_cells/y_cells"):
        flux = poynting_flux(sim, layer=0, x_points=geo.x, y_points=geo.y)

    expected = torch.tensor(0.99**2, dtype=geo.dtype)
    assert torch.isclose(flux, expected, atol=1e-12)


def test_poynting_flux_does_not_warn_for_subset_of_cell_centers():
    # A subset of centers is indistinguishable from an endpoint grid over a
    # sub-region, so it must not warn.
    geo = _geometry()
    sim = ConstantPoyntingSimulation()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        poynting_flux(sim, layer=0, x_points=geo.x[20:50], y_points=geo.y[30:70])


def test_poynting_flux_does_not_warn_for_symmetric_endpoint_points():
    # linspace(-L/2, L/2, n) has a half-step phase but is an endpoint grid.
    sim = ConstantPoyntingSimulation()
    points = torch.linspace(-0.5, 0.5, 100, dtype=torch.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flux = poynting_flux(sim, layer=0, x_points=points, y_points=points)

    assert torch.isclose(flux, torch.tensor(1.0, dtype=torch.float64))


def test_poynting_flux_does_not_warn_for_small_scale_endpoint_points():
    sim = ConstantPoyntingSimulation()
    x_points = torch.linspace(0.0, 1e-8, 100, dtype=torch.float64)
    y_points = torch.linspace(0.0, 1e-8, 100, dtype=torch.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        poynting_flux(sim, layer=0, x_points=x_points, y_points=y_points)


def test_poynting_flux_rejects_mixed_point_and_cell_axes():
    geo = _geometry()
    sim = ConstantPoyntingSimulation()

    with pytest.raises(ValueError, match="either x_points/y_points or x_cells/y_cells"):
        poynting_flux(
            sim,
            layer=0,
            x_points=geo.x,
            y_cells=geo.y,
        )


def test_poynting_flux_rejects_incomplete_axis_pairs():
    geo = _geometry()
    sim = ConstantPoyntingSimulation()

    with pytest.raises(ValueError, match="x_cells and y_cells must be provided together"):
        poynting_flux(sim, layer=0, x_cells=geo.x)


def test_poynting_flux_rejects_nonuniform_cell_centers():
    sim = ConstantPoyntingSimulation()
    x_cells = torch.tensor([0.1, 0.2, 0.35], dtype=torch.float64)
    y_cells = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)

    with pytest.raises(ValueError, match="equally spaced"):
        poynting_flux(
            sim,
            layer=0,
            x_cells=x_cells,
            y_cells=y_cells,
        )
