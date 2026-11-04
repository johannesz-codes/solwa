"""
Test for ``rcwa(auto_precision=True)``: complex64 compute with a complex128
eigendecomposition, while results keep the user-facing dtype.
"""

import pytest
import torch

import solwa

L = 450.0
REAL = {torch.complex64: torch.float32, torch.complex128: torch.float64}


def _transmission(dtype, auto, device):
    """Return (sim, t, T, dT/dR) for a patterned pillar plus a uniform spacer layer."""
    radius = torch.tensor(120.0, dtype=REAL[dtype], device=device, requires_grad=True)
    sim = solwa.rcwa(
        freq=torch.tensor(1 / 532.0, dtype=REAL[dtype], device=device),
        order=[2, 2],
        L=[L, L],
        dtype=dtype,
        device=device,
        auto_precision=auto,
    )
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=2.25)
    sim.set_incident_angle(inc_ang=0.2, azi_ang=0.0)
    geo = solwa.geometry(
        Lx=L,
        Ly=L,
        nx=128,
        ny=128,
        edge_sharpness=1000.0,
        dtype=REAL[dtype],
        device=device,
    )
    sim.add_layer(
        thickness=300.0, eps=1.0 + 5.0 * geo.circle(R=radius, Cx=L / 2, Cy=L / 2)
    )
    # Uniform layer given as a tensor: exactly degenerate modes.
    spacer = torch.full((128, 128), 2.1, dtype=REAL[dtype], device=device)
    sim.add_layer(thickness=80.0, eps=spacer)
    sim.solve_global_smatrix()
    t = sim.S_parameters(orders=[[0, 0], [1, 0]], polarization="xx")
    T = (t.abs() ** 2).sum()
    (grad,) = torch.autograd.grad(T, radius)
    return sim, t, T.item(), grad


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_auto_precision(dtype, device):
    _, _, T_ref, grad_ref = _transmission(torch.complex128, False, device)
    sim, t, T, grad = _transmission(dtype, True, device)

    # Internals run in complex64; results and gradients keep the input dtype.
    assert sim.S[0].dtype == torch.complex64
    assert t.dtype == dtype and grad.dtype == REAL[dtype]
    sim.source_planewave(amplitude=[1.0, 0.0])
    axis = torch.linspace(0.0, L, 6, dtype=REAL[dtype], device=device)
    E, _ = sim.field_xy(0, axis, axis)
    assert E[0].dtype == dtype
    assert sim.poynting_xy(0, axis, axis)[2].dtype == REAL[dtype]

    # Accuracy close to full complex128.
    assert abs(T - T_ref) < 1e-4
    assert abs(grad.item() - grad_ref.item()) / abs(grad_ref.item()) < 1e-3
