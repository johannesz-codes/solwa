"""Gradients vs finite differences in a symmetry-breaking direction (#91).

A centered square at normal incidence has exactly degenerate modes; the
gradient must still match finite differences, also in complex64/auto."""

import pytest
import torch

import solwa


def _loss(eps, device, dtype=torch.complex128, auto=False):
    sim = solwa.rcwa(
        freq=1 / 1.3, order=[5, 5], L=[1.0, 1.0], dtype=dtype, device=device,
        auto_precision=auto,
    )  # fmt: skip
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=2.25)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=0.15, eps=eps.to(dtype))
    sim.solve_global_smatrix()
    S = torch.stack(
        [
            sim.S_parameters(
                orders=[[0, 0], [1, 0], [0, 1]], port=port, polarization=pol
            )
            for port in ("t", "r")
            for pol in ("xx", "yx")
        ]
    )
    return (S.abs() ** 2).sum()


@pytest.mark.parametrize(
    "Wy, loss, dtype, auto, tol",
    [
        (0.25, 0.1, torch.complex128, False, 1e-4),  # rectangle: no degeneracy
        (0.4, 0.1, torch.complex128, False, 1e-4),  # square: degenerate modes
        (0.4, 0.0, torch.complex128, False, 1e-4),  # lossless: kz branch
        (0.4, 0.1, torch.complex128, True, 1e-3),
        (0.4, 0.1, torch.complex64, False, 1e-2),
    ],
)
def test_degenerate_grad(Wy, loss, dtype, auto, tol, device):
    geo = solwa.geometry(
        Lx=1.0, Ly=1.0, nx=64, ny=64, edge_sharpness=500.0,
        dtype=torch.float64, device=device,
    )  # fmt: skip
    eps0 = 1.0 + (11.0 + 1j * loss) * geo.rectangle(Wx=0.4, Wy=Wy, Cx=0.5, Cy=0.5)
    v = torch.randn(64, 64, dtype=torch.float64, device=device,
                    generator=torch.Generator(device).manual_seed(0))  # fmt: skip

    eps = eps0.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(_loss(eps, device, dtype, auto), eps)
    ad = (grad.conj() * v).real.sum().item()
    h = 1e-6
    fd = (_loss(eps0 + h * v, device) - _loss(eps0 - h * v, device)).item() / (2 * h)
    assert abs(ad - fd) < tol * abs(fd)


def test_thick_layer_grad_finite(device):
    """Propagation divided differences must not overflow for evanescent modes."""
    eps = torch.full((16, 16), 2.25, dtype=torch.complex64, device=device)
    eps[:8, :8] += 0.5
    eps.requires_grad_(True)
    sim = solwa.rcwa(
        freq=1 / 1.3, order=[3, 3], L=[1.0, 1.0], dtype=torch.complex64, device=device
    )
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=1.0)
    sim.set_incident_angle(inc_ang=0.1, azi_ang=0.2)
    sim.add_layer(thickness=10.0, eps=eps)
    sim.solve_global_smatrix()
    T = sim.S_parameters(orders=[[0, 0]], port="t", polarization="xx").abs() ** 2
    (grad,) = torch.autograd.grad(T.sum(), eps)
    assert torch.isfinite(grad).all()
