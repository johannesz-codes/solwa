"""Verify patterned-material P/Q assembly against the original dense formula."""

import pytest
import torch

import solwa


def dense_pq(sim):
    """Use block-stacked diagonal matrices independently of scaled assembly."""
    x, y = torch.diag(sim.Kx_norm_dn), torch.diag(sim.Ky_norm_dn)
    eps, mu = sim.eps_conv[-1], sim.mu_conv[-1]
    zero = torch.zeros_like(eps)
    p = torch.vstack((torch.hstack((zero, mu)), torch.hstack((-mu, zero))))
    p = p + torch.vstack((x, y)) @ torch.linalg.inv(eps) @ torch.hstack((y, -x))
    q = torch.vstack((torch.hstack((zero, -eps)), torch.hstack((eps, zero))))
    q = q + torch.vstack((x, y)) @ torch.linalg.inv(mu) @ torch.hstack((-y, x))
    return p, q


@pytest.mark.parametrize("lossy", [False, True])
def test_patterned_pq_and_material_gradients_match_dense_formula(lossy):
    device = torch.device("cpu")
    contrasts = torch.tensor([2.7, .6], dtype=torch.float64, requires_grad=True)
    axis = (torch.arange(32, dtype=torch.float64) + .5) / 32
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    eps_mask = ((x > .18) & (x < .69) & (y > .26) & (y < .81)).to(torch.float64)
    mu_mask = ((x > .31) & (x < .86) & (y > .12) & (y < .58)).to(torch.float64)
    eps = 1. + contrasts[0] * (1. + (.07j if lossy else 0.)) * eps_mask
    mu = 1.2 + contrasts[1] * (1. + (.11j if lossy else 0.)) * mu_mask
    sim = solwa.rcwa(freq=1 / 532., order=[1, 2], L=[300., 340.],
                     dtype=torch.complex128, device=device)
    sim.add_input_layer(eps=1.)
    sim.set_incident_angle(inc_ang=.19, azi_ang=.37)
    sim.add_layer(thickness=127., eps=eps, mu=mu)

    actual = (sim.P[-1], sim.Q[-1])
    expected = dense_pq(sim)
    for matrix, reference in zip(actual, expected):
        torch.testing.assert_close(matrix, reference, atol=1e-12, rtol=1e-12)

    # Nonuniform real and imaginary weights avoid relying on a norm or trace
    # that could hide a sign mistake or transposition. This graph stops at P/Q
    # and does not differentiate through eigenvectors or layer scattering.
    weights = torch.arange(actual[0].numel(), dtype=torch.float64).reshape_as(actual[0])
    weights = torch.sin(weights * .37) + 1j * torch.cos(weights * .23)

    def objective(matrices):
        return sum((weight * matrix).real.sum() for weight, matrix in
                   zip((weights, weights.T * (.3 + .8j)), matrices))

    actual_gradient = torch.autograd.grad(objective(actual), contrasts, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(objective(expected), contrasts)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-11, rtol=1e-11)
