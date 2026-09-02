import pytest
import torch

import solwa
from solwa.utils import poynting_flux


class ConstantPoyntingSimulation:
    _dtype = torch.complex128
    _device = torch.device("cpu")

    def poynting_xy(self, layer, x_axis, y_axis, z_prop):
        del layer, z_prop
        shape = (x_axis.numel(), y_axis.numel())
        sx = torch.zeros(shape, dtype=x_axis.dtype, device=x_axis.device)
        sy = torch.zeros(shape, dtype=x_axis.dtype, device=x_axis.device)
        sz = torch.ones(shape, dtype=x_axis.dtype, device=x_axis.device)
        return sx, sy, sz


def _geometry(n=100):
    geo = solwa.geometry(
        Lx=1.0,
        Ly=1.0,
        nx=n,
        ny=n,
        dtype=torch.float64,
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
