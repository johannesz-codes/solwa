"""``rcwa(eig_reuse=True)`` must agree with the regular eigendecomposition."""

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

import solwa
from solwa.eig_reuse import refine, reuse_eig

CDT = torch.complex128
L = 500.0
SMOOTH = [520.0 + k for k in range(8)]
POLS = ["xx", "yx", "ss", "pp"]


def _eps(lam):
    return 12.0 + 0.1j + 2e-3 * (600.0 - lam) * (1 + 0.1j)


def _fill(width, device):
    x = L * (torch.arange(64, dtype=torch.float64, device=device) + 0.5) / 64
    edge = torch.sigmoid((width / 2 - (x - L / 2).abs()) / (L / 64 / 2))
    return (edge[:, None] * edge[None, :]).to(CDT)


def _solve(
    lams, width, device, reuse, order=3, layers=1, factorization="li", stable=True
):
    lam = torch.as_tensor(lams, dtype=torch.float64, device=device)
    sim = solwa.rcwa(
        freq=1.0 / lam,
        order=[order, order],
        L=[L, L],
        dtype=CDT,
        device=device,
        factorization=factorization,
        stable_eig_grad=stable,
        eig_reuse=reuse,
    )
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=2.25)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    eps = _eps(lam)[..., None, None] if lam.dim() else _eps(lam)
    fill = _fill(width, device)
    for k in range(layers):
        sim.add_layer(thickness=120.0 + 30.0 * k, eps=2.25 + (eps - 2.25) * fill)
    sim.solve_global_smatrix()
    return torch.stack([sim.S_parameters(orders=[0, 0], polarization=p) for p in POLS])


def _setwise(a, b):
    a, b = a.detach().cpu().numpy(), b.detach().cpu().numpy()
    row, col = linear_sum_assignment(np.abs(a[:, None] - b[None, :]))
    return np.abs(a[row] - b[col]).max()


def _family(device, B=6, n=40):
    gen = torch.Generator(device=device).manual_seed(3)
    A0 = torch.randn(n, n, dtype=CDT, device=device, generator=gen)
    A1 = torch.randn(n, n, dtype=CDT, device=device, generator=gen)
    t = torch.arange(B, dtype=torch.float64, device=device) * 1e-3
    return A0 + t[:, None, None] * A1


def _count_eig(monkeypatch):
    calls = []
    eig = torch.linalg.eig

    def counted(x, *a, **k):
        calls.append(1)
        return eig(x, *a, **k)

    monkeypatch.setattr(torch.linalg, "eig", counted)
    return calls


def test_reuse_eig_matches_eig_on_non_normal_family(device):
    A = _family(device)
    mu, V = reuse_eig(A)
    for k in range(A.shape[0]):
        assert _setwise(mu[k], torch.linalg.eigvals(A[k])) < 1e-10
    res = A @ V - V * mu[:, None, :]
    assert res.norm(dim=-2).max() / A.norm(dim=(-2, -1)).max() < 1e-12


def test_refine_rejects_a_wrong_inverse(device):
    A = _family(device, B=1)
    V = torch.linalg.eig(A)[1]
    *_, ok = refine(A, V.clone(), torch.zeros_like(V), max_iter=2)
    assert not bool(ok[0])


@pytest.mark.parametrize("factorization", ["li", "laurent"])
def test_matches_regular_solve(factorization, device):
    ref = _solve(SMOOTH, 300.0, device, False, factorization=factorization)
    got = _solve(SMOOTH, 300.0, device, True, factorization=factorization)
    assert (got - ref).abs().max() < 1e-10


def test_two_patterned_layers(device):
    ref = _solve(SMOOTH, 300.0, device, False, layers=2)
    got = _solve(SMOOTH, 300.0, device, True, layers=2)
    assert (got - ref).abs().max() < 1e-10


def test_refinement_is_actually_used(device, monkeypatch):
    calls = _count_eig(monkeypatch)
    _solve(SMOOTH, 300.0, device, True)
    assert len(calls) < len(SMOOTH) / 2


def test_large_steps_stay_correct(device):
    lams = [450.0, 800.0, 470.0, 900.0, 520.0]
    ref = _solve(lams, 300.0, device, False)
    got = _solve(lams, 300.0, device, True)
    assert (got - ref).abs().max() < 1e-10


def test_scalar_frequency_is_unaffected(device):
    ref = _solve(520.0, 300.0, device, False)
    got = _solve(520.0, 300.0, device, True)
    assert torch.equal(got, ref)


def test_batch_of_one(device):
    ref = _solve([520.0], 300.0, device, False)
    got = _solve([520.0], 300.0, device, True)
    assert (got - ref).abs().max() < 1e-11


def test_requires_complex128_eig():
    with pytest.raises(ValueError, match="complex128"):
        solwa.rcwa(freq=1 / 500.0, order=[1, 1], L=[L, L], eig_reuse=True)
    solwa.rcwa(
        freq=1 / 500.0, order=[1, 1], L=[L, L], eig_reuse=True, auto_precision=True
    )


def _objective(width, reuse, device, stable=True):
    S = _solve(SMOOTH, width, device, reuse, stable=stable)
    return (S[:2].abs() ** 2).sum()


def _width(value, device, grad=False):
    return torch.tensor(value, dtype=torch.float64, device=device, requires_grad=grad)


@pytest.mark.parametrize("stable", [True, False])
def test_gradient_matches_regular_backward(stable, device):
    grads = []
    for reuse in (False, True):
        width = _width(300.0, device, grad=True)
        _objective(width, reuse, device, stable=stable).backward()
        grads.append(width.grad.item())
    assert np.isfinite(grads).all()
    assert abs(grads[1] - grads[0]) <= 1e-8 * abs(grads[0])


def test_gradient_matches_finite_difference(device):
    width = _width(300.0, device, grad=True)
    _objective(width, True, device).backward()
    h = 1e-3
    up = _objective(_width(300.0 + h, device), True, device).item()
    dn = _objective(_width(300.0 - h, device), True, device).item()
    assert abs(width.grad.item() - (up - dn) / (2 * h)) <= 1e-6 * abs(width.grad.item())


def test_gradient_with_respect_to_matrix(device):
    A = _family(device).requires_grad_()
    mu, _ = reuse_eig(A)
    mu.real.sum().backward()
    mu2 = torch.linalg.eigvals(A)
    grad_ref = torch.autograd.grad(mu2.real.sum(), A)[0]
    assert (A.grad - grad_ref).abs().max() <= 1e-9 * grad_ref.abs().max()
