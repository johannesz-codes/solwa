"""
TMM validation for layer_absorption correctness.

Validates that ``rcwa.layer_absorption()`` matches the analytical
Transfer-Matrix Method (TMM) for homogeneous absorbing layers, ensuring
that the Poynting-flux normalization uses incident power (not net flux).

Background
----------
A common implementation error when computing absorption from the Poynting
flux is to use the "naive" formula:

    A_naive = 1 - P_out / P_in

where P_in is the net Poynting flux entering the layer.  Because P_in
already has the reflected power subtracted, this formula yields:

    A_naive = A / (1 - R)

which systematically *overstates* the true absorptance by a factor of
1 / (1 - R).  For a typical ~4 % Fresnel reflectance this corresponds
to a ~4 % relative error — large enough to corrupt optimization targets.

The correct formula (implemented in ``layer_absorption``) is:

    A = (P_top - P_bottom) / P_incident

where P_incident = 0.5 * Lx * Ly for a unit-amplitude plane wave in
vacuum at normal incidence.

These tests compare ``layer_absorption`` against exact TMM results for
single absorbing slabs at several wavelengths, incidence conditions, and
material parameters.  Any regression toward the naive formula will cause
an immediate test failure.
"""

import cmath
from math import pi

import pytest
import torch

import solwa


# ---------------------------------------------------------------------------
# TMM reference implementation (Fabry-Perot for a single slab)
# ---------------------------------------------------------------------------


def tmm_single_slab(n0, n1, n2, thickness, wavelength):
    """Exact TMM T, R, A for an absorbing slab at normal incidence.

    Parameters
    ----------
    n0 : complex
        Refractive index of input medium.
    n1 : complex
        Refractive index of the slab.
    n2 : complex
        Refractive index of output medium.
    thickness : float
        Slab thickness (same length unit as wavelength).
    wavelength : float
        Free-space wavelength.

    Returns
    -------
    T, R, A : float
        Power transmittance, reflectance, and absorptance.
    """
    t01 = 2 * n0 / (n0 + n1)
    t12 = 2 * n1 / (n1 + n2)
    r10 = (n1 - n0) / (n1 + n0)
    r12 = (n1 - n2) / (n1 + n2)
    r01 = (n0 - n1) / (n0 + n1)
    t10 = 2 * n1 / (n1 + n0)

    phi = 2 * pi * n1 * thickness / wavelength

    denom = 1 - r10 * r12 * cmath.exp(2j * phi)
    t_total = t01 * t12 * cmath.exp(1j * phi) / denom
    r_total = r01 + t01 * r12 * t10 * cmath.exp(2j * phi) / denom

    # Power coefficients (account for different media on each side)
    T = abs(t_total) ** 2 * (n2.real / n0.real)
    R = abs(r_total) ** 2
    A = 1.0 - T - R
    return T, R, A


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

# Each entry: (wavelength, n_slab, thickness, L, order, description)
TEST_CASES = [
    # Moderate absorption in visible
    (532.0, 1.5 + 0.1j, 200.0, [300.0, 300.0], [3, 3], "glass-like with loss"),
    # High absorption (metallic-like)
    (600.0, 0.5 + 3.0j, 50.0, [400.0, 400.0], [3, 3], "metallic slab"),
    # Weak absorption, thicker layer
    (800.0, 2.0 + 0.01j, 500.0, [500.0, 500.0], [3, 3], "weak loss thick slab"),
    # Very lossy dielectric
    (450.0, 3.0 + 1.0j, 100.0, [300.0, 300.0], [3, 3], "high-index lossy"),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLayerAbsorptionMatchesTMM:
    """layer_absorption() must match TMM to within 1e-3 for homogeneous slabs."""

    @pytest.fixture(params=TEST_CASES, ids=[tc[5] for tc in TEST_CASES])
    def case(self, request):
        wavelength, n_slab, thickness, L, order, desc = request.param
        eps_slab = n_slab ** 2

        sim = solwa.rcwa(
            freq=1.0 / wavelength,
            order=order,
            L=L,
            dtype=torch.complex128,
            device=torch.device("cpu"),
        )
        sim.add_input_layer(eps=1.0)
        sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
        sim.add_layer(thickness=thickness, eps=complex(eps_slab))
        sim.solve_global_smatrix()
        sim.source_planewave(amplitude=[1.0, 0.0], direction="forward")

        T_tmm, R_tmm, A_tmm = tmm_single_slab(
            n0=1.0, n1=n_slab, n2=1.0,
            thickness=thickness, wavelength=wavelength,
        )

        return sim, L, T_tmm, R_tmm, A_tmm

    def test_layer_absorption_matches_tmm(self, case):
        """layer_absorption() agrees with TMM absorptance."""
        sim, L, T_tmm, R_tmm, A_tmm = case

        x = torch.linspace(0, L[0], 50, dtype=torch.float64)
        y = torch.linspace(0, L[1], 50, dtype=torch.float64)

        A_computed = sim.layer_absorption(0, x, y).real.item()
        assert abs(A_computed - A_tmm) < 1e-3, (
            f"layer_absorption={A_computed:.6f} vs TMM A={A_tmm:.6f} "
            f"(difference={abs(A_computed - A_tmm):.2e})"
        )

    def test_layer_absorption_not_naive(self, case):
        """layer_absorption() must NOT match the naive (incorrect) formula.

        The naive formula gives A/(1-R) which is always larger than A when
        R > 0.  This test ensures that the method returns the true A, not
        the inflated value.
        """
        sim, L, T_tmm, R_tmm, A_tmm = case

        x = torch.linspace(0, L[0], 50, dtype=torch.float64)
        y = torch.linspace(0, L[1], 50, dtype=torch.float64)

        A_computed = sim.layer_absorption(0, x, y).real.item()
        A_naive = A_tmm / (1.0 - R_tmm)  # the wrong answer

        # The computed value must be closer to A_tmm than to A_naive
        err_correct = abs(A_computed - A_tmm)
        err_naive = abs(A_computed - A_naive)
        assert err_correct < err_naive, (
            f"layer_absorption={A_computed:.6f} is closer to naive "
            f"A/(1-R)={A_naive:.6f} than to correct A={A_tmm:.6f}"
        )


class TestLayerAbsorptionEnergyBalance:
    """Sum of layer absorptions + T + R = 1 (energy conservation)."""

    def test_single_layer_energy_balance(self):
        """R + A + T = 1 for a single absorbing slab."""
        wavelength = 532.0
        n_slab = 1.5 + 0.1j
        thickness = 200.0
        L = [300.0, 300.0]

        sim = solwa.rcwa(
            freq=1.0 / wavelength, order=[3, 3], L=L,
            dtype=torch.complex128, device=torch.device("cpu"),
        )
        sim.add_input_layer(eps=1.0)
        sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
        sim.add_layer(thickness=thickness, eps=complex(n_slab ** 2))
        sim.solve_global_smatrix()
        sim.source_planewave(amplitude=[1.0, 0.0], direction="forward")

        # S-parameter T and R
        t = sim.S_parameters(
            orders=[[0, 0]], direction="forward", port="transmission",
            polarization="xx", ref_order=[0, 0],
        )
        r = sim.S_parameters(
            orders=[[0, 0]], direction="forward", port="reflection",
            polarization="xx", ref_order=[0, 0],
        )
        T = abs(t.item()) ** 2
        R = abs(r.item()) ** 2

        x = torch.linspace(0, L[0], 50, dtype=torch.float64)
        y = torch.linspace(0, L[1], 50, dtype=torch.float64)
        A = sim.layer_absorption(0, x, y).real.item()

        total = T + R + A
        assert abs(total - 1.0) < 1e-3, (
            f"Energy balance: T={T:.6f} + R={R:.6f} + A={A:.6f} = {total:.6f}"
        )

    def test_multilayer_energy_balance(self):
        """R + sum(A_i) + T = 1 for a two-layer absorbing structure."""
        wavelength = 532.0
        L = [300.0, 300.0]

        sim = solwa.rcwa(
            freq=1.0 / wavelength, order=[3, 3], L=L,
            dtype=torch.complex128, device=torch.device("cpu"),
        )
        sim.add_input_layer(eps=1.0)
        sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
        sim.add_layer(thickness=150.0, eps=complex((1.5 + 0.1j) ** 2))
        sim.add_layer(thickness=100.0, eps=complex((2.0 + 0.05j) ** 2))
        sim.solve_global_smatrix()
        sim.source_planewave(amplitude=[1.0, 0.0], direction="forward")

        t = sim.S_parameters(
            orders=[[0, 0]], direction="forward", port="transmission",
            polarization="xx", ref_order=[0, 0],
        )
        r = sim.S_parameters(
            orders=[[0, 0]], direction="forward", port="reflection",
            polarization="xx", ref_order=[0, 0],
        )
        T = abs(t.item()) ** 2
        R = abs(r.item()) ** 2

        x = torch.linspace(0, L[0], 50, dtype=torch.float64)
        y = torch.linspace(0, L[1], 50, dtype=torch.float64)

        A_total = sum(
            sim.layer_absorption(i, x, y).real.item()
            for i in range(sim.layer_N)
        )

        total = T + R + A_total
        assert abs(total - 1.0) < 1e-3, (
            f"Energy balance: T={T:.6f} + R={R:.6f} + A_total={A_total:.6f} = {total:.6f}"
        )
