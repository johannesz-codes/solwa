"""Gradients vs finite differences in a symmetry-breaking direction (#91)."""

import pytest
import torch

import solwa


def _loss(eps, device):
    sim = solwa.rcwa(
        freq=1 / 1.3, order=[5, 5], L=[1.0, 1.0], dtype=torch.complex128, device=device
    )
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=2.25)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=0.15, eps=eps)
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
    "Wy",
    [
        0.25,  # centered rectangle (C2v): no degenerate modes
        pytest.param(0.4, marks=pytest.mark.xfail(strict=True, reason="#91")),
    ],
)
def test_degenerate_grad(Wy, device):
    geo = solwa.geometry(
        Lx=1.0, Ly=1.0, nx=64, ny=64, edge_sharpness=500.0,
        dtype=torch.float64, device=device,
    )  # fmt: skip
    eps0 = 1.0 + (11.0 + 0.1j) * geo.rectangle(Wx=0.4, Wy=Wy, Cx=0.5, Cy=0.5)
    v = torch.randn(64, 64, dtype=torch.float64, device=device,
                    generator=torch.Generator(device).manual_seed(0))  # fmt: skip

    eps = eps0.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(_loss(eps, device), eps)
    ad = (grad.conj() * v).real.sum().item()
    h = 1e-6
    fd = (_loss(eps0 + h * v, device) - _loss(eps0 - h * v, device)).item() / (2 * h)
    assert abs(ad - fd) < 1e-4 * abs(fd)
