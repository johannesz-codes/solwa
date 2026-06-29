import torch

import solwa
from solwa.utils import poynting_flux


class ConstantPoyntingSimulation:
    def poynting_xy(self, layer, x_points, y_points, z_prop):
        del layer, z_prop
        shape = (x_points.numel(), y_points.numel())
        sx = torch.zeros(shape, dtype=x_points.dtype, device=x_points.device)
        sy = torch.zeros(shape, dtype=x_points.dtype, device=x_points.device)
        sz = torch.ones(shape, dtype=x_points.dtype, device=x_points.device)
        return sx, sy, sz


def test_poynting_flux_integrates_cell_centered_periodic_grid_area():
    n = 100
    geo = solwa.geometry(
        Lx=1.0,
        Ly=1.0,
        nx=n,
        ny=n,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    geo.grid()

    sim = ConstantPoyntingSimulation()
    flux = poynting_flux(sim, layer=0, x_points=geo.x, y_points=geo.y, z_prop=0.0)

    assert torch.isclose(
        flux,
        torch.tensor(1.0, dtype=geo.dtype, device=geo.device),
        rtol=0.0,
        atol=1e-12,
    )
