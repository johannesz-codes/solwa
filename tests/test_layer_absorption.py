"""Interface fluxes and layer absorption from mode amplitudes (``flux_interfaces``)."""

import pytest
import torch

import solwa

EPS_METAL = -10.0 + 1.0j
EPS_LOSSY = 4.0 + 0.3j
EPS_TIO2 = 6.25 + 0j
EPS_SUB = 2.25

DTYPES = {
    "fp32": (torch.complex64, torch.float32, 1e-4),
    "fp64": (torch.complex128, torch.float64, 1e-10),
}


@pytest.fixture(params=list(DTYPES))
def prec(request):
    return DTYPES[request.param]


def build(
    dtype,
    rdtype,
    device,
    *,
    R=0.3,
    eps_lossy=EPS_LOSSY,
    eps_disk=EPS_METAL,
    order=(3, 3),
    n=64,
    pol=(1.0, 0.0),
):
    geo = solwa.geometry(Lx=1.0, Ly=1.0, nx=n, ny=n, dtype=rdtype, device=device)
    sim = solwa.rcwa(
        freq=1 / 1.3, order=list(order), L=[1.0, 1.0], dtype=dtype, device=device
    )
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=EPS_SUB)
    sim.set_incident_angle(inc_ang=0.3, azi_ang=0.2)
    disk = geo.circle(R=R, Cx=0.5, Cy=0.5)
    sim.add_layer(thickness=0.1, eps=1.0 + (eps_disk - 1.0) * disk)
    sim.add_layer(thickness=0.2, eps=eps_lossy)
    sim.add_layer(
        thickness=0.15,
        eps=1.0 + (EPS_TIO2 - 1.0) * geo.rectangle(Wx=0.6, Wy=0.3, Cx=0.5, Cy=0.5),
    )
    sim.solve_global_smatrix()
    sim.source_planewave(amplitude=list(pol), direction="forward", notation="ps")
    return sim, geo


def ps_power(sim, port, direction="forward"):
    orders = torch.stack(
        torch.meshgrid(sim.order_x, sim.order_y, indexing="ij"), dim=-1
    ).reshape(-1, 2)
    return sum(
        (
            sim.S_parameters(
                orders=orders, direction=direction, port=port, polarization=p
            ).abs()
            ** 2
        ).sum()
        for p in ("pp", "sp")
    )


def test_energy_balance_matches_s_parameters(device, prec):
    dtype, rdtype, tol = prec
    sim, _ = build(dtype, rdtype, device)
    flux = sim.flux_interfaces()
    A = sim.layer_absorption()
    R, T = ps_power(sim, "r"), ps_power(sim, "t")

    assert A.shape == (sim.layer_N,)
    assert A[0] > 1e-2 and A[1] > 1e-2
    assert abs(1 - flux[0] - R) < 10 * tol
    assert abs(flux[-1] - T) < 10 * tol
    assert abs(R + T + A.sum() - 1) < 10 * tol


def test_matches_grid_poynting_flux(device, prec):
    dtype, rdtype, tol = prec
    sim, geo = build(dtype, rdtype, device)
    flux = sim.flux_interfaces(normalize=False)
    grid = [sim.poynting_flux(i, x_cells=geo.x, y_cells=geo.y) for i in range(3)]
    grid.append(sim.poynting_flux(2, x_cells=geo.x, y_cells=geo.y, z_prop=0.15))
    grid = torch.stack(grid).to(flux.dtype)

    torch.testing.assert_close(flux, grid, rtol=10 * tol, atol=10 * tol)


def test_lossless_layers_do_not_absorb(device, prec):
    dtype, rdtype, tol = prec
    sim, _ = build(dtype, rdtype, device, eps_lossy=4.0 + 0j, eps_disk=EPS_TIO2)
    A = sim.layer_absorption()

    assert A.abs().max() < 10 * tol


def test_backward_source_energy_balance(device, prec):
    dtype, rdtype, tol = prec
    sim, _ = build(dtype, rdtype, device)
    sim.source_planewave(amplitude=[1.0, 0.0], direction="backward", notation="ps")
    flux = sim.flux_interfaces()
    A = sim.layer_absorption()
    R, T = ps_power(sim, "r", "backward"), ps_power(sim, "t", "backward")

    assert (A[:2] > 1e-2).all()
    assert abs(1 + flux[-1] - R) < 10 * tol
    assert abs(-flux[0] - T) < 10 * tol


def test_gradient_matches_finite_difference(device, prec):
    dtype, rdtype, _ = prec
    R = torch.tensor(0.3, dtype=rdtype, device=device, requires_grad=True)
    sim, _ = build(dtype, rdtype, device, R=R)
    A = sim.layer_absorption()[0]
    (grad,) = torch.autograd.grad(A, R)
    assert torch.isfinite(grad)

    if dtype == torch.complex128:
        h = 1e-5
        with torch.no_grad():
            Ap = build(dtype, rdtype, device, R=0.3 + h)[0].layer_absorption()[0]
            Am = build(dtype, rdtype, device, R=0.3 - h)[0].layer_absorption()[0]
        torch.testing.assert_close(grad, (Ap - Am) / (2 * h), rtol=1e-3, atol=1e-6)


def test_layer_selection_matches_full_absorption(device, prec):
    dtype, rdtype, _ = prec
    sim, _ = build(dtype, rdtype, device)
    A = sim.layer_absorption()

    torch.testing.assert_close(sim.layer_absorption(layers=[1]), A[1:2])
    torch.testing.assert_close(sim.layer_absorption(layers=[2, 0]), A[[2, 0]])
    torch.testing.assert_close(
        sim.flux_interfaces(interfaces=[3, 0]), sim.flux_interfaces()[[3, 0]]
    )
