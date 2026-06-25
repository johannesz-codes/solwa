"""
Minimal reproducer: absorption undershoot by ~2% when using Poynting flux.

Root cause
----------
When layer absorption is computed as ``1 - P_out / P_in`` (where P_in is the
net Poynting flux *entering* the layer), the result is **not** the true
absorption but ``A / (1 - R)`` — systematically higher than the correct value
by a factor of ``1 / (1 - R)``.  For typical reflectances of a few percent
this looks like a ~2 % discrepancy relative to the transfer-matrix method (TMM).

The correct formula is::

    A = (P_in - P_out) / P_incident

where ``P_incident`` is the power carried by the *incident wave alone* (not
the net flux, which already has reflection subtracted).  For a unit-amplitude
plane wave at normal incidence in vacuum the incident power over one unit cell
is ``P_incident = 0.5 * Lx * Ly``.

Equivalently, the S-parameter approach ``A = 1 - T - R`` is always correct and
agrees with TMM to machine precision for homogeneous layers.

Structure under test
--------------------
Absorbing slab (n = 1.5 + 0.1j, thickness = 200 nm) in air at normal incidence,
λ = 532 nm.  TMM provides the exact analytical reference.
"""

import cmath
from math import pi

import pytest
import torch

import solwa


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

LAMBDA = 532.0  # nm
N_SLAB = 1.5 + 0.1j
EPS_SLAB = N_SLAB**2
THICKNESS = 200.0  # nm
L = [300.0, 300.0]
ORDER = [3, 3]

# TMM reference (Fabry-Perot formula for absorbing slab at normal incidence)
_n0 = 1.0
_n1 = N_SLAB
_n2 = 1.0
_t01 = 2 * _n0 / (_n0 + _n1)
_t12 = 2 * _n1 / (_n1 + _n2)
_r10 = (_n1 - _n0) / (_n1 + _n0)
_r12 = (_n1 - _n2) / (_n1 + _n2)
_r01 = (_n0 - _n1) / (_n0 + _n1)
_t10 = 2 * _n1 / (_n1 + _n0)
_phi = 2 * pi * _n1 * THICKNESS / LAMBDA

T_TMM = abs(_t01 * _t12 * cmath.exp(1j * _phi) / (1 - _r10 * _r12 * cmath.exp(2j * _phi))) ** 2
R_TMM = abs(_r01 + _t01 * _r12 * _t10 * cmath.exp(2j * _phi) / (1 - _r10 * _r12 * cmath.exp(2j * _phi))) ** 2
A_TMM = 1.0 - T_TMM - R_TMM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=[torch.complex64, torch.complex128], ids=["fp32", "fp64"])
def sim(request):
    """RCWA simulation of a homogeneous absorbing slab."""
    dtype = request.param
    s = solwa.rcwa(freq=1 / LAMBDA, order=ORDER, L=L, dtype=dtype, device=torch.device("cpu"))
    s.add_input_layer(eps=1.0)
    s.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    s.add_layer(thickness=THICKNESS, eps=complex(EPS_SLAB))
    s.solve_global_smatrix()
    s.source_planewave(amplitude=[1.0, 0.0], direction="forward")
    return s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAbsorptionSparamMatchesTMM:
    """S-parameter absorption matches TMM to floating-point precision."""

    def test_transmittance(self, sim):
        t = sim.S_parameters(
            orders=[[0, 0]], direction="forward", port="transmission",
            polarization="xx", ref_order=[0, 0],
        )
        assert abs(abs(t.item()) ** 2 - T_TMM) < 1e-4

    def test_reflectance(self, sim):
        r = sim.S_parameters(
            orders=[[0, 0]], direction="forward", port="reflection",
            polarization="xx", ref_order=[0, 0],
        )
        assert abs(abs(r.item()) ** 2 - R_TMM) < 1e-4

    def test_absorption(self, sim):
        t = sim.S_parameters(
            orders=[[0, 0]], direction="forward", port="transmission",
            polarization="xx", ref_order=[0, 0],
        )
        r = sim.S_parameters(
            orders=[[0, 0]], direction="forward", port="reflection",
            polarization="xx", ref_order=[0, 0],
        )
        A_rcwa = 1.0 - abs(t.item()) ** 2 - abs(r.item()) ** 2
        assert abs(A_rcwa - A_TMM) < 1e-4


class TestPoyntingFluxNormalization:
    """Demonstrates the ~2 % error from incorrect Poynting flux normalization.

    The naive formula ``A_naive = 1 - P_out / P_in`` overshoots the true
    absorption by the factor ``1 / (1 - R)`` because ``P_in`` already has the
    reflected power subtracted.

    The correct formula normalizes by the *incident* power instead:
    ``A_correct = (P_in - P_out) / P_incident``.
    """

    def test_naive_normalization_overshoots(self, sim):
        """Naive Poynting ratio gives ~2 % excess (≈ R/(1-R) relative error)."""
        dtype = sim._dtype
        nx, ny = 50, 50
        x = torch.linspace(0, L[0], nx, dtype=dtype)
        y = torch.linspace(0, L[1], ny, dtype=dtype)

        P_in = sim.poynting_flux(0, x, y, z_prop=0.0).real.item()
        P_out = sim.poynting_flux(0, x, y, z_prop=THICKNESS).real.item()

        A_naive = 1.0 - P_out / P_in  # WRONG — this equals A/(1-R)
        expected_naive = A_TMM / (1.0 - R_TMM)

        # The naive result matches A/(1-R), NOT A
        assert abs(A_naive - expected_naive) < 1e-3
        # And is systematically higher than the true absorption
        assert A_naive > A_TMM

    def test_correct_normalization_matches_tmm(self, sim):
        """Poynting flux normalized by incident power matches TMM exactly."""
        dtype = sim._dtype
        nx, ny = 50, 50
        x = torch.linspace(0, L[0], nx, dtype=dtype)
        y = torch.linspace(0, L[1], ny, dtype=dtype)

        P_in = sim.poynting_flux(0, x, y, z_prop=0.0).real.item()
        P_out = sim.poynting_flux(0, x, y, z_prop=THICKNESS).real.item()

        # Incident power for unit-amplitude x-polarized plane wave in vacuum
        # Sz_inc = 0.5 * Re(Ex * Hy*) = 0.5 * 1 * 1 = 0.5 per unit area
        P_incident = 0.5 * L[0] * L[1]

        A_correct = (P_in - P_out) / P_incident
        assert abs(A_correct - A_TMM) < 1e-3

    def test_relative_error_equals_reflectance(self, sim):
        """The relative overestimate from naive normalization equals R/(1-R)."""
        dtype = sim._dtype
        nx, ny = 50, 50
        x = torch.linspace(0, L[0], nx, dtype=dtype)
        y = torch.linspace(0, L[1], ny, dtype=dtype)

        P_in = sim.poynting_flux(0, x, y, z_prop=0.0).real.item()
        P_out = sim.poynting_flux(0, x, y, z_prop=THICKNESS).real.item()

        A_naive = 1.0 - P_out / P_in
        relative_error = (A_naive - A_TMM) / A_TMM

        # Error should be approximately R/(1-R)
        expected_relative_error = R_TMM / (1.0 - R_TMM)
        assert abs(relative_error - expected_relative_error) < 1e-3


class TestStructuredLayerConsistency:
    """For structured absorbing layers, S-params and Poynting flux agree."""

    @pytest.fixture
    def sim_grating(self):
        """Binary absorbing grating (sub-wavelength period, only 0th order)."""
        dtype = torch.complex128
        grating_L = [400.0, 400.0]
        s = solwa.rcwa(
            freq=1 / LAMBDA, order=[15, 0], L=grating_L,
            dtype=dtype, device=torch.device("cpu"),
        )
        s.add_input_layer(eps=1.0)
        s.set_incident_angle(inc_ang=0.0, azi_ang=0.0)

        nx = 400
        eps_manual = torch.ones(nx, 4, dtype=dtype)
        eps_manual[: nx // 2, :] = 2.25 + 0.3j  # absorbing half
        s.add_layer(thickness=200.0, eps=eps_manual)
        s.solve_global_smatrix()
        s.source_planewave(amplitude=[1.0, 0.0], direction="forward")
        return s, grating_L

    def test_sparam_equals_poynting_correct(self, sim_grating):
        """S-parameter A agrees with correctly normalized Poynting flux."""
        sim, grating_L = sim_grating
        dtype = sim._dtype

        # S-parameter absorption (0th order only for sub-wavelength period)
        t = sim.S_parameters(
            orders=[[0, 0]], direction="forward", port="transmission",
            polarization="xx", ref_order=[0, 0],
        )
        r = sim.S_parameters(
            orders=[[0, 0]], direction="forward", port="reflection",
            polarization="xx", ref_order=[0, 0],
        )
        A_spar = 1.0 - abs(t.item()) ** 2 - abs(r.item()) ** 2

        # Poynting flux with correct normalization
        nx, ny = 100, 10
        x = torch.linspace(0, grating_L[0], nx, dtype=dtype)
        y = torch.linspace(0, grating_L[1], ny, dtype=dtype)

        P_in = sim.poynting_flux(0, x, y, z_prop=0.0).real.item()
        P_out = sim.poynting_flux(0, x, y, z_prop=200.0).real.item()
        P_incident = 0.5 * grating_L[0] * grating_L[1]
        A_poynting = (P_in - P_out) / P_incident

        assert abs(A_spar - A_poynting) < 1e-3, (
            f"S-param A={A_spar:.6f} vs Poynting A={A_poynting:.6f}"
        )
