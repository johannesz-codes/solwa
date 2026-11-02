"""Li's Fourier factorization (``factorization="li"``) against the Laurent rule.

Checks that the inverse rule only changes what it should (TM / normal E
components), keeps the x/y bookkeeping right, conserves energy, converges to
the right value for a metallic TM grating and stays differentiable.
"""

import pytest
import torch

import solwa

EPS_METAL = (0.22 + 6.71j) ** 2
EPS_DIEL = 2.25**2 + 0j
EPS_SUB = 2.25

DTYPES = {
    "fp32": (torch.complex64, 1e-4),
    "fp64": (torch.complex128, 1e-9),
}


def lamellar(eps, dtype, device, nx=512, fill=0.5):
    """x-periodic, y-invariant binary grating as [nx, 1] grid."""
    x = (torch.arange(nx, device=device) + 0.5) / nx
    one = torch.tensor(1.0 + 0j, dtype=dtype, device=device)
    return torch.where(
        x < fill, torch.tensor(eps, dtype=dtype, device=device), one
    ).reshape(nx, 1)


def rectangle(eps, dtype, device, n=64, wx=0.6, wy=0.3):
    x = (torch.arange(n, device=device) + 0.5) / n
    X, Y = torch.meshgrid(x, x, indexing="ij")
    mask = (X < wx) & (Y < wy)
    one = torch.tensor(1.0 + 0j, dtype=dtype, device=device)
    return torch.where(mask, torch.tensor(eps, dtype=dtype, device=device), one)


def make_sim(
    eps_grid, order, factorization, dtype, device, *, freq=1.0, inc=0.0, azi=0.0, h=0.5
):
    sim = solwa.rcwa(
        freq=freq,
        order=order,
        L=[1.0, 1.0],
        dtype=dtype,
        device=device,
        factorization=factorization,
    )
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=EPS_SUB)
    sim.set_incident_angle(inc_ang=inc, azi_ang=azi)
    sim.add_layer(thickness=h, eps=eps_grid)
    sim.solve_global_smatrix()
    return sim


def r0(sim, pol):
    return (
        abs(sim.S_parameters(orders=[[0, 0]], port="r", polarization=pol)[0]).item()
        ** 2
    )


@pytest.fixture(params=list(DTYPES))
def prec(request):
    return DTYPES[request.param]


def test_invalid_factorization_rejected():
    with pytest.raises(ValueError):
        solwa.rcwa(freq=1.0, order=[1, 1], L=[1.0, 1.0], factorization="foo")


def test_homogeneous_grid_gives_scaled_identity(device, prec):
    dtype, tol = prec
    sim = solwa.rcwa(freq=1.0, order=[3, 2], L=[1.0, 1.0], dtype=dtype, device=device)
    grid = torch.full((32, 16), 2.5 + 0.1j, dtype=dtype, device=device)
    eye = torch.eye(sim.order_N, dtype=dtype, device=device)
    for conv in sim._material_conv_li(grid):
        assert torch.allclose(conv, (2.5 + 0.1j) * eye, atol=tol * 10)


def test_te_unchanged(device, prec):
    """Ey is tangential to x-interfaces: Li and Laurent must coincide."""
    dtype, tol = prec
    g = lamellar(EPS_METAL, dtype, device)
    res = [
        r0(make_sim(g, [10, 0], f, dtype, device, inc=0.3), "yy")
        for f in ("li", "laurent")
    ]
    assert res[0] == pytest.approx(res[1], rel=tol * 10, abs=tol)


def test_xy_bookkeeping_rotation(device, prec):
    """Transposing the structure and swapping the polarization gives the same R0."""
    dtype, tol = prec
    g = rectangle(EPS_METAL, dtype, device)
    a = r0(make_sim(g, [3, 3], "li", dtype, device, freq=1 / 1.3, h=0.25), "xx")
    b = r0(
        make_sim(g.T.contiguous(), [3, 3], "li", dtype, device, freq=1 / 1.3, h=0.25),
        "yy",
    )
    assert a == pytest.approx(b, rel=tol * 100, abs=tol * 10)


def test_energy_conservation_lossless(device, prec):
    dtype, tol = prec
    M = 8
    g = rectangle(EPS_DIEL, dtype, device)
    sim = make_sim(
        g, [M, M], "li", dtype, device, freq=1 / 1.3, inc=0.2, azi=0.4, h=0.25
    )
    orders = [[i, j] for i in range(-M, M + 1) for j in range(-M, M + 1)]
    total = 0.0
    for port in ("r", "t"):
        for pol in ("pp", "sp"):  # ps basis: |S|^2 is additive power
            total += (
                (
                    sim.S_parameters(orders=orders, port=port, polarization=pol).abs()
                    ** 2
                )
                .sum()
                .item()
            )
    assert total == pytest.approx(1.0, abs=tol * 10)


def test_metal_tm_converges_to_reference(device, prec):
    """Metallic lamellar grating, TM: Li at M=20 is close to the converged value,
    Laurent at the same order is not (reference: Li, M=300, fp64, nx=4096)."""
    dtype, _ = prec
    ref = 0.39743
    g = lamellar(EPS_METAL, dtype, device)
    li = r0(make_sim(g, [20, 0], "li", dtype, device, inc=0.3), "xx")
    laurent = r0(make_sim(g, [20, 0], "laurent", dtype, device, inc=0.3), "xx")
    assert abs(li - ref) < 2e-3
    assert abs(laurent - ref) > 5e-3


def test_gradient_matches_finite_difference(device):
    dtype = torch.complex128
    x = (torch.arange(128, device=device) + 0.5) / 128
    mask = (x < 0.5).to(torch.float64).reshape(-1, 1)

    def f(e):
        g = (1.0 + (e - 1.0) * mask).to(dtype)
        sim = make_sim(g, [6, 0], "li", dtype, device, inc=0.3)
        return (
            sim.S_parameters(orders=[[0, 0]], port="r", polarization="xx")[0].abs() ** 2
        )

    e = torch.tensor(4.0, dtype=torch.float64, device=device, requires_grad=True)
    f(e).backward()
    h = 1e-6
    fd = (
        f(torch.tensor(4.0 + h, dtype=torch.float64, device=device))
        - f(torch.tensor(4.0 - h, dtype=torch.float64, device=device))
    ) / (2 * h)
    assert e.grad.item() == pytest.approx(fd.item(), rel=1e-5)
