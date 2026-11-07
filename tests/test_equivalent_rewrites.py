"""CPU double-precision checks for equivalent RCWA matrix rewrites."""

import cmath
import math

import pytest
import torch

from solwa import rcwa


def simulation(
    cls=rcwa,
    *,
    order=1,
    contrast=3.0,
    thickness=120.0,
    homogeneous=False,
    wavelength=532.0,
    **settings,
):
    sim = cls(
        freq=1 / wavelength,
        order=[order, order],
        L=[300.0, 340.0],
        dtype=torch.complex128,
        device=torch.device("cpu"),
        **settings,
    )
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=1.0)
    sim.set_incident_angle(
        inc_ang=0.0 if homogeneous else 0.13, azi_ang=0.0 if homogeneous else 0.27
    )
    axis = (torch.arange(32, dtype=torch.float64) + 0.5) / 32
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    mask = ((x > 0.2) & (x < 0.63) & (y > 0.25) & (y < 0.77)).double()
    for layer in range(1 if homogeneous else 2):
        eps = (
            2.25
            if homogeneous
            else 1.0 + contrast * torch.roll(mask, (3 * layer, 5 * layer), (0, 1))
        )
        sim.add_layer(thickness=thickness, eps=eps)
    sim.solve_global_smatrix()
    return sim


class DirectCouplingReference(rcwa):
    """Solve the full tangential-field matching equations without block reduction."""

    def _solve_layer_smatrix(self):
        E, k = self.E_eigvec[-1], self.kz_norm[-1]
        identity = torch.eye(E.shape[0], dtype=E.dtype, device=E.device)
        if self.avoid_Pinv_instability and self.max_Pinv_instability == 0.0:
            H = self.Q[-1] @ E @ torch.diag(1 / k)
        else:
            H = torch.linalg.solve(self.P[-1], E @ torch.diag(k))
        self.H_eigvec.append(H)
        phase = torch.diag(torch.exp(1j * self.omega * k * self.thickness[-1]))
        admittance = torch.linalg.solve(self.Vf, H)
        A, B = E + admittance, (E - admittance) @ phase
        matching = torch.cat((torch.cat((A, B), 1), torch.cat((B, A), 1)), 0)
        coefficients = torch.linalg.solve(
            matching, 2 * torch.eye(2 * E.shape[0], dtype=E.dtype)
        )
        Cf, Cb = coefficients.chunk(2, dim=1)
        self.Cf.append(Cf)
        self.Cb.append(Cb)
        ftop, fbottom = Cf.chunk(2, dim=0)
        btop, bbottom = Cb.chunk(2, dim=0)
        self.layer_S11.append(E @ phase @ ftop + E @ fbottom)
        self.layer_S21.append(E @ ftop + E @ phase @ fbottom - identity)
        self.layer_S12.append(E @ phase @ btop + E @ bbottom - identity)
        self.layer_S22.append(E @ btop + E @ phase @ bbottom)


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"contrast": 3.0 + 0.15j},
        {"offload_device": "cpu"},
        {"stable_eig_grad": False},
        {"avoid_Pinv_instability": True},
        {"avoid_Pinv_instability": True, "max_Pinv_instability": 0.0},
    ],
)
def test_full_matching_system_and_fields(settings):
    actual = simulation(**settings)
    reference = simulation(DirectCouplingReference, **settings)
    for name in (
        "H_eigvec",
        "Cf",
        "Cb",
        "layer_S11",
        "layer_S21",
        "layer_S12",
        "layer_S22",
        "S",
    ):
        for got, expected in zip(getattr(actual, name), getattr(reference, name)):
            torch.testing.assert_close(got, expected, atol=2e-10, rtol=2e-10)
    axis = torch.linspace(0, 290, 3, dtype=torch.float64)
    for direction in ("forward", "backward"):
        actual.source_planewave(amplitude=[1.0, 0.3j], direction=direction)
        reference.source_planewave(amplitude=[1.0, 0.3j], direction=direction)
        for layer in range(2):
            for got_group, expected_group in zip(
                actual.field_xy(layer, axis, axis, z_prop=40.0),
                reference.field_xy(layer, axis, axis, z_prop=40.0),
            ):
                for got, expected in zip(got_group, expected_group):
                    torch.testing.assert_close(got, expected, atol=2e-9, rtol=2e-9)
    # Equal scattering blocks must remain independently writable.
    for left, right in (
        (actual.layer_S11, actual.layer_S22),
        (actual.layer_S21, actual.layer_S12),
    ):
        for a, b in zip(left, right):
            assert a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()


def transmission(sim):
    return (
        sim.S_parameters(orders=[[0, 0]], port="transmission", polarization="xx")
        .abs()
        .square()
        .sum()
    )


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("wavelength", [532.0, 610.0])
def test_fabry_perot_and_energy(order, wavelength):
    sim = simulation(
        order=order, homogeneous=True, thickness=200.0, wavelength=wavelength
    )
    n = 1.5
    phase = cmath.exp(2j * math.pi * n * 200.0 / wavelength)
    expected = (
        abs(4 * n / (1 + n) ** 2 * phase / (1 - ((n - 1) / (n + 1)) ** 2 * phase**2))
        ** 2
    )
    torch.testing.assert_close(
        transmission(sim),
        torch.tensor(expected, dtype=torch.float64),
        atol=1e-10,
        rtol=1e-10,
    )
    energy = sum(
        sim.S_parameters(orders=[[0, 0]], port=port, polarization=pol)
        .abs()
        .square()
        .sum()
        for port in ("transmission", "reflection")
        for pol in ("xx", "yx")
    )
    torch.testing.assert_close(
        energy, torch.tensor(1.0, dtype=torch.float64), atol=1e-10, rtol=1e-10
    )


@pytest.mark.parametrize("order", [1, 2])
def test_patterned_gradients(order):
    gradients = []
    for cls in (rcwa, DirectCouplingReference):
        contrast = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        thickness = torch.tensor(120.0, dtype=torch.float64, requires_grad=True)
        loss = transmission(
            simulation(cls, order=order, contrast=contrast, thickness=thickness)
        )
        gradients.append(torch.stack(torch.autograd.grad(loss, (contrast, thickness))))
    torch.testing.assert_close(gradients[0], gradients[1], atol=1e-9, rtol=1e-7)
    finite = []
    with torch.no_grad():
        for name, center, step in [("contrast", 3.0, 1e-4), ("thickness", 120.0, 1e-3)]:
            plus = transmission(simulation(order=order, **{name: center + step}))
            minus = transmission(simulation(order=order, **{name: center - step}))
            finite.append((plus - minus) / (2 * step))
    torch.testing.assert_close(gradients[0], torch.stack(finite), atol=1e-8, rtol=2e-5)
