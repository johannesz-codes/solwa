"""
Tests for ``rcwa(auto_precision=True)``.

With auto precision the eigendecomposition runs in complex128 and everything
else in complex64, while public results come back in the user-facing dtype.
"""

import pytest
import torch

import solwa

LAMBDA = 532.0
L = 450.0
REAL = {torch.complex64: torch.float32, torch.complex128: torch.float64}


def _sim(dtype, auto, radius, device):
    sim = solwa.rcwa(
        freq=1 / LAMBDA,
        order=[4, 4],
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
    pillar = geo.circle(R=radius, Cx=L / 2, Cy=L / 2)
    sim.add_layer(thickness=300.0, eps=1.0 + 5.0 * pillar)
    sim.add_layer(thickness=80.0, eps=2.1)
    sim.solve_global_smatrix()
    return sim


def _transmission(dtype, auto, device):
    radius = torch.tensor(120.0, dtype=REAL[dtype], device=device, requires_grad=True)
    sim = _sim(dtype, auto, radius, device)
    t = sim.S_parameters(orders=[[0, 0], [1, 0]], polarization="xx")
    T = (t.abs() ** 2).sum()
    (grad,) = torch.autograd.grad(T, radius)
    return sim, t, T.detach(), grad


@pytest.fixture(scope="module")
def reference(device):
    _, _, T, grad = _transmission(torch.complex128, False, device)
    return T, grad


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_compute_precision(dtype, device):
    sim = _sim(dtype, True, 120.0, device)
    assert sim.S[0].dtype == torch.complex64
    assert sim.kz_norm[-1].dtype == torch.complex64


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_outputs_keep_input_dtype(dtype, device):
    sim, t, _, grad = _transmission(dtype, True, device)
    assert t.dtype == dtype
    assert grad.dtype == REAL[dtype]

    inc, azi = sim.diffraction_angle([[1, 0]])
    assert inc.dtype == REAL[dtype] and azi.dtype == REAL[dtype]

    sim.source_planewave(amplitude=[1.0, 0.0])
    axis = torch.linspace(0.0, L, 6, dtype=REAL[dtype], device=device)
    E, H = sim.field_xy(0, axis, axis)
    assert all(c.dtype == dtype for c in (*E, *H))
    Sx, Sy, Sz = sim.poynting_xy(0, axis, axis)
    assert Sz.dtype == REAL[dtype]
    E, H = sim.field_xz(
        axis, torch.linspace(-50.0, 400.0, 5, dtype=REAL[dtype], device=device), 0.0
    )
    assert E[0].dtype == dtype


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_matches_complex128(dtype, reference, device):
    T_ref, grad_ref = reference
    _, _, T, grad = _transmission(dtype, True, device)
    assert abs(T.item() - T_ref.item()) < 1e-4
    assert abs(grad.item() - grad_ref.item()) / abs(grad_ref.item()) < 1e-3


def test_disabled_by_default(device):
    sim = _sim(torch.complex128, False, 120.0, device)
    assert not sim.auto_precision
    assert sim.S[0].dtype == torch.complex128
