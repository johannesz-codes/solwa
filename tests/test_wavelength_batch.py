"""A 1-D ``freq`` must reproduce a loop over scalar frequencies, incl. gradients."""

import pytest
import torch

import solwa

L = 400.0
LAMS = [500.0, 620.0, 780.0]
POLS = ["xx", "yx", "pp", "ss"]


def _run(freq, eps_hi, dtype, factorization, device, radius):
    sim = solwa.rcwa(
        freq=freq, order=[2, 2], L=[L, L], dtype=dtype, device=device,
        factorization=factorization,
    )  # fmt: skip
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=eps_hi)
    sim.set_incident_angle(inc_ang=0.2, azi_ang=0.1)
    geo = solwa.geometry(
        Lx=L, Ly=L, nx=64, ny=64, edge_sharpness=500.0,
        dtype=dtype.to_real(), device=device,
    )  # fmt: skip
    mask = geo.circle(R=radius, Cx=L / 2, Cy=L / 2)
    sim.add_layer(thickness=150.0, eps=1.0 + (eps_hi[..., None, None] - 1.0) * mask)
    sim.add_layer(thickness=80.0, eps=eps_hi)
    sim.solve_global_smatrix()
    orders = [[0, 0], [1, 0], [-1, 1]]
    return torch.stack(
        [sim.S_parameters(orders=orders, polarization=pol) for pol in POLS],
        dim=-2,
    )


@pytest.mark.parametrize("factorization", ["li", "laurent"])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_wavelength_batch(dtype, factorization, device):
    rdt = dtype.to_real()
    freq = 1.0 / torch.tensor(LAMS, dtype=rdt, device=device)
    eps_hi = (2.0 + 1e5 / torch.tensor(LAMS, device=device) ** 2 + 0.02j).to(dtype)
    radius = torch.tensor(110.0, dtype=rdt, device=device, requires_grad=True)

    S = _run(freq, eps_hi, dtype, factorization, device, radius)
    (grad,) = torch.autograd.grad((S.abs() ** 2).sum(), radius)

    S_ref, grad_ref = [], 0.0
    for i in range(len(LAMS)):
        s = _run(freq[i], eps_hi[i], dtype, factorization, device, radius)
        grad_ref = grad_ref + torch.autograd.grad((s.abs() ** 2).sum(), radius)[0]
        S_ref.append(s.detach())

    tol = 1e-4 if dtype == torch.complex64 else 1e-10
    assert S.shape == (len(LAMS), len(POLS), 3)
    torch.testing.assert_close(S.detach(), torch.stack(S_ref), rtol=tol, atol=tol)
    torch.testing.assert_close(grad, grad_ref, rtol=10 * tol, atol=10 * tol)
