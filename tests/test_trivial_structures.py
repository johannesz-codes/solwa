"""
RCWA tests using trivial photonic structures.

Each test validates a well-known analytical or symmetry property:
  - Homogeneous glass slab  : energy conservation and Fabry-Perot formula
  - Single slit             : energy conservation
  - Double slit             : destructive interference at m=±1, mirror symmetry,
                              and energy conservation
  - Binary lamellar grating : grating equation and energy conservation

Simulation notes (from the example notebooks)
----------------------------------------------
* λ = 532 nm is used throughout so that λ/period is never an integer ratio
  and no diffraction order falls exactly on kz = 0, which would cause NaN.
* Only ``add_input_layer`` is called (no ``add_output_layer``); the output
  half-space defaults to free space (eps = 1), matching the example convention.
* The new ``solwa.geometry`` instance API is used (not the legacy ``rcwa_geo``
  class) to avoid shared state between tests.
"""

import cmath
from math import asin, pi

import pytest
import torch

import solwa

# ---------------------------------------------------------------------------
# Shared simulation constants
# ---------------------------------------------------------------------------

LAMBDA = 532.0  # free-space wavelength [nm]
DEVICE = torch.device("cpu")
DTYPE = torch.complex64
GEO_DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _make_sim(freq, order, L):
    """Return an RCWA object with air input layer at normal incidence."""
    sim = solwa.rcwa(freq=freq, order=order, L=L, dtype=DTYPE, device=DEVICE)
    sim.add_input_layer(eps=1.0)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    return sim


def _T(sim, m, n=0):
    """Power transmittance for diffraction order (m, n)."""
    t = sim.S_parameters(
        orders=[[m, n]],
        direction="forward",
        port="transmission",
        polarization="xx",
        ref_order=[0, 0],
    )
    return abs(t.item()) ** 2


def _R(sim, m, n=0):
    """Power reflectance for diffraction order (m, n)."""
    r = sim.S_parameters(
        orders=[[m, n]],
        direction="forward",
        port="reflection",
        polarization="xx",
        ref_order=[0, 0],
    )
    return abs(r.item()) ** 2


def _energy_balance(sim, max_order):
    """Return T + R summed over all orders in [-max_order, max_order]."""
    T_sum = sum(_T(sim, m) for m in range(-max_order, max_order + 1))
    R_sum = sum(_R(sim, m) for m in range(-max_order, max_order + 1))
    return T_sum + R_sum


# ---------------------------------------------------------------------------
# 1. Homogeneous glass slab
# ---------------------------------------------------------------------------


class TestHomogeneousSlab:
    """
    RCWA of a lossless glass slab (n = 1.5) in air.

    Because the structure is homogeneous, RCWA reduces to the transfer-matrix
    method and the results must match the Fabry-Perot formula exactly (up to
    floating-point rounding).
    """

    N_GLASS = 1.5
    EPS_GLASS = N_GLASS**2  # 2.25, real → lossless
    THICKNESS = 200.0  # nm
    L = [300.0, 300.0]
    ORDER = [3, 3]

    @pytest.fixture
    def sim(self):
        s = _make_sim(freq=1 / LAMBDA, order=self.ORDER, L=self.L)
        s.add_layer(thickness=self.THICKNESS, eps=float(self.EPS_GLASS))
        s.solve_global_smatrix()
        return s

    def test_energy_conservation(self, sim):
        """T + R = 1 for a lossless dielectric slab (only 0th order propagates)."""
        total = _energy_balance(sim, max_order=0)
        assert abs(total - 1.0) < 1e-4

    def test_fabry_perot_agreement(self, sim):
        """Zeroth-order transmittance matches the Fabry-Perot formula."""
        n1 = self.N_GLASS
        d = self.THICKNESS
        # Amplitude product t01 * t12 and reflectance product r10 * r12.
        # At normal incidence, n0 = n2 = 1 (air):
        #   t01 = 2/(1+n1),  t12 = 2*n1/(n1+1)
        #   r10 = (n1-1)/(n1+1),  r12 = (n1-1)/(n1+1)
        t_product = (2.0 / (1 + n1)) * (2 * n1 / (n1 + 1))
        r_product = ((n1 - 1) / (n1 + 1)) ** 2
        phi = 2 * pi * n1 * d / LAMBDA  # single-pass phase in the slab
        T_FP = (
            abs(t_product * cmath.exp(1j * phi) / (1 - r_product * cmath.exp(2j * phi)))
            ** 2
        )
        T_rcwa = _T(sim, 0)
        assert abs(T_rcwa - T_FP) < 1e-3


# ---------------------------------------------------------------------------
# 2. Single slit
# ---------------------------------------------------------------------------


class TestSingleSlit:
    """
    1-D grating: one narrow transparent slit (20 % fill) per unit cell
    surrounded by a lossless dielectric background (eps = 4).
    """

    LX = 1000.0  # nm, grating period
    LY = 1000.0  # nm, orthogonal period (irrelevant for 1-D structure)
    SLIT_WIDTH = 200.0  # nm transparent opening
    EPS_BG = 4.0  # lossless dielectric background
    THICKNESS = 300.0  # nm
    ORDER = [5, 0]

    @pytest.fixture
    def sim(self):
        s = _make_sim(freq=1 / LAMBDA, order=self.ORDER, L=[self.LX, self.LY])
        geo = solwa.geometry(
            Lx=self.LX,
            Ly=self.LY,
            nx=100,
            ny=100,
            edge_sharpness=1000.0,
            dtype=GEO_DTYPE,
            device=DEVICE,
        )
        # Transparent slit in a high-index background
        opening = geo.rectangle(
            Wx=self.SLIT_WIDTH, Wy=self.LY, Cx=self.LX / 2, Cy=self.LY / 2
        )
        layer_eps = opening * 1.0 + (1.0 - opening) * self.EPS_BG
        s.add_layer(thickness=self.THICKNESS, eps=layer_eps)
        s.solve_global_smatrix()
        return s

    def test_energy_conservation(self, sim):
        """T + R = 1 summed over all propagating diffraction orders."""
        total = _energy_balance(sim, max_order=self.ORDER[0])
        assert abs(total - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# 3. Double slit
# ---------------------------------------------------------------------------


class TestDoubleSlit:
    """
    Unit cell with two identical, mirror-symmetric transparent slits.

    Period Lx = 2000 nm contains one slit at Lx/4 and one at 3*Lx/4.
    This yields two well-known properties at normal incidence:

    1. Destructive interference at m = ±1:
       The structure factor at m=1 is ∝ cos(π/2) = 0, so |T[±1]| = 0.

    2. Mirror symmetry |T[+m]| = |T[-m]| for all m.
    """

    LX = 2000.0  # nm
    LY = 1000.0  # nm
    SLIT_WIDTH = 200.0  # nm
    EPS_BG = 4.0
    THICKNESS = 300.0  # nm
    ORDER = [7, 0]

    @pytest.fixture
    def sim(self):
        s = _make_sim(freq=1 / LAMBDA, order=self.ORDER, L=[self.LX, self.LY])
        geo = solwa.geometry(
            Lx=self.LX,
            Ly=self.LY,
            nx=200,
            ny=100,
            edge_sharpness=1000.0,
            dtype=GEO_DTYPE,
            device=DEVICE,
        )
        slit_A = geo.rectangle(
            Wx=self.SLIT_WIDTH, Wy=self.LY, Cx=self.LX / 4, Cy=self.LY / 2
        )
        slit_B = geo.rectangle(
            Wx=self.SLIT_WIDTH, Wy=self.LY, Cx=3 * self.LX / 4, Cy=self.LY / 2
        )
        double_slit = geo.union(slit_A, slit_B)
        layer_eps = double_slit * 1.0 + (1.0 - double_slit) * self.EPS_BG
        s.add_layer(thickness=self.THICKNESS, eps=layer_eps)
        s.solve_global_smatrix()
        return s

    def test_first_order_destructive_interference(self, sim):
        """
        The ±1st orders vanish: the slit pair spacing equals half the period,
        making the structure-factor zero at m = ±1.
        """
        assert _T(sim, +1) < 1e-3
        assert _T(sim, -1) < 1e-3

    def test_mirror_symmetry(self, sim):
        """
        Mirror symmetry of the unit cell forces |T[+m]| = |T[-m]| for all m
        at normal incidence.
        """
        for m in (1, 2, 3):
            assert abs(_T(sim, +m) - _T(sim, -m)) < 1e-5

    def test_energy_conservation(self, sim):
        """T + R = 1 for the lossless double-slit structure."""
        total = _energy_balance(sim, max_order=self.ORDER[0])
        assert abs(total - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# 4. Binary lamellar grating
# ---------------------------------------------------------------------------


class TestBinaryGrating:
    """
    50 % fill-factor lamellar grating: alternating strips of air (eps = 1)
    and dielectric (eps = 4) with period 1000 nm.
    """

    PERIOD = 1000.0  # nm
    LY = 1000.0  # nm
    EPS_HIGH = 4.0
    THICKNESS = 300.0  # nm
    ORDER = [3, 1]

    @pytest.fixture
    def sim(self):
        s = _make_sim(
            freq=1 / LAMBDA, order=self.ORDER, L=[self.PERIOD, self.LY]
        )
        geo = solwa.geometry(
            Lx=self.PERIOD,
            Ly=self.LY,
            nx=100,
            ny=100,
            edge_sharpness=1000.0,
            dtype=GEO_DTYPE,
            device=DEVICE,
        )
        # Low-index (air) stripe on the left half, high-index on the right
        stripe = geo.rectangle(
            Wx=self.PERIOD / 2,
            Wy=self.LY,
            Cx=self.PERIOD / 2,
            Cy=self.LY / 2,
        )
        layer_eps = stripe * 1.0 + (1.0 - stripe) * self.EPS_HIGH
        s.add_layer(thickness=self.THICKNESS, eps=layer_eps)
        s.solve_global_smatrix()
        return s

    def test_grating_equation(self, sim):
        """
        The first transmitted diffraction order satisfies sin θ = λ / d,
        which is independent of material parameters.
        """
        angle_deg, _ = sim.diffraction_angle([[1, 0]], layer="output", unit="degree")
        expected_deg = asin(LAMBDA / self.PERIOD) * 180 / pi
        assert abs(angle_deg.item() - expected_deg) < 1e-3

    def test_energy_conservation(self, sim):
        """T + R = 1 for the lossless binary grating."""
        total = _energy_balance(sim, max_order=self.ORDER[0])
        assert abs(total - 1.0) < 1e-4
