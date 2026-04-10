"""Batch-wavelength tests for known optical structures.

For each structure from ``test_trivial_structures.py`` we run the identical
simulation with a *batch* of three wavelengths in a single forward pass and
verify:

1. Each element of the batch matches the corresponding scalar per-wavelength
   result (numerical regression).
2. The same optical properties hold for every element of the batch:
   energy conservation, Fabry-Perot formula, destructive interference, and
   the grating equation.

Wavelengths [500, 532, 600] nm are used throughout.  These are chosen to
avoid kz = 0 singularities (no diffraction order falls exactly on the light
cone edge for any cell geometry used below).
"""

import cmath
from math import asin, pi

import pytest
import torch

import solwa

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

WAVELENGTHS = [532.0, 600.0, 650.0]  # nm — chosen to avoid kz=0 singularities
FREQS = torch.tensor([1.0 / w for w in WAVELENGTHS], dtype=torch.complex64)

DTYPE = torch.complex64
DEVICE = torch.device("cpu")
GEO_DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sim(freq, order, L):
    """Return an RCWA object with air input layer at normal incidence."""
    sim = solwa.rcwa(freq=freq, order=order, L=L, dtype=DTYPE, device=DEVICE)
    sim.add_input_layer(eps=1.0)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    return sim


def _T_batch(sim, b, m, n=0):
    """Power transmittance for batch element *b*, diffraction order (m, n)."""
    t = sim.S_parameters(
        orders=[[m, n]],
        direction="forward",
        port="transmission",
        polarization="xx",
        ref_order=[0, 0],
    )
    return abs(t[b, 0].item()) ** 2


def _R_batch(sim, b, m, n=0):
    """Power reflectance for batch element *b*, diffraction order (m, n)."""
    r = sim.S_parameters(
        orders=[[m, n]],
        direction="forward",
        port="reflection",
        polarization="xx",
        ref_order=[0, 0],
    )
    return abs(r[b, 0].item()) ** 2


def _energy_balance_batch(sim, b, max_order):
    """Return T + R summed over all orders for batch element *b*."""
    T_sum = sum(_T_batch(sim, b, m) for m in range(-max_order, max_order + 1))
    R_sum = sum(_R_batch(sim, b, m) for m in range(-max_order, max_order + 1))
    return T_sum + R_sum


def _T_scalar(sim, m, n=0):
    """Power transmittance for a scalar sim, diffraction order (m, n)."""
    t = sim.S_parameters(
        orders=[[m, n]],
        direction="forward",
        port="transmission",
        polarization="xx",
        ref_order=[0, 0],
    )
    return abs(t.item()) ** 2


# ---------------------------------------------------------------------------
# 1. Homogeneous glass slab
# ---------------------------------------------------------------------------


class TestBatchHomogeneousSlab:
    """Batch version of TestHomogeneousSlab.

    A lossless glass slab (n = 1.5) in air, run with three wavelengths in a
    single forward pass.
    """

    N_GLASS = 1.5
    EPS_GLASS = N_GLASS**2  # 2.25
    THICKNESS = 200.0  # nm
    L = [300.0, 300.0]
    ORDER = [3, 3]

    @pytest.fixture(scope="class")
    def batch_sim(self):
        s = _make_sim(freq=FREQS, order=self.ORDER, L=self.L)
        s.add_layer(thickness=self.THICKNESS, eps=float(self.EPS_GLASS))
        s.solve_global_smatrix()
        return s

    @pytest.fixture(scope="class")
    def scalar_sims(self):
        sims = []
        for lam in WAVELENGTHS:
            s = _make_sim(freq=1.0 / lam, order=self.ORDER, L=self.L)
            s.add_layer(thickness=self.THICKNESS, eps=float(self.EPS_GLASS))
            s.solve_global_smatrix()
            sims.append(s)
        return sims

    def test_batch_matches_scalar_transmission(self, batch_sim, scalar_sims):
        """Zeroth-order transmission matches scalar per-wavelength result."""
        for b, s in enumerate(scalar_sims):
            T_batch = _T_batch(batch_sim, b, 0)
            T_scalar = _T_scalar(s, 0)
            assert abs(T_batch - T_scalar) < 1e-5, (
                f"λ={WAVELENGTHS[b]} nm: batch T={T_batch:.6f}, scalar T={T_scalar:.6f}"
            )

    def test_energy_conservation_per_wavelength(self, batch_sim):
        """T + R = 1 for each wavelength in the batch (lossless slab).

        Ground-truth: energy conservation for a lossless medium.

        Reference: Born & Wolf §1.5.
        """
        for b in range(len(WAVELENGTHS)):
            total = _energy_balance_batch(batch_sim, b, max_order=0)
            assert abs(total - 1.0) < 1e-4, (
                f"λ={WAVELENGTHS[b]} nm: T+R={total:.6f}"
            )

    def test_fabry_perot_per_wavelength(self, batch_sim):
        """Zeroth-order transmittance matches the Fabry-Perot formula for each wavelength.

        For a lossless slab (n1, thickness d) in air at normal incidence:

            T_FP = |t01 * t12 * exp(iφ) / (1 - r10*r12*exp(2iφ))|²

        Reference: Born & Wolf, §1.6.
        """
        n1 = self.N_GLASS
        d = self.THICKNESS
        t_product = (2.0 / (1 + n1)) * (2 * n1 / (n1 + 1))
        r_product = ((n1 - 1) / (n1 + 1)) ** 2
        for b, lam in enumerate(WAVELENGTHS):
            phi = 2 * pi * n1 * d / lam
            T_FP = (
                abs(
                    t_product
                    * cmath.exp(1j * phi)
                    / (1 - r_product * cmath.exp(2j * phi))
                )
                ** 2
            )
            T_rcwa = _T_batch(batch_sim, b, 0)
            assert abs(T_rcwa - T_FP) < 1e-3, (
                f"λ={lam} nm: RCWA={T_rcwa:.6f}, Fabry-Perot={T_FP:.6f}"
            )


# ---------------------------------------------------------------------------
# 2. Single slit
# ---------------------------------------------------------------------------


class TestBatchSingleSlit:
    """Batch version of TestSingleSlit.

    One narrow transparent slit (20 % fill) per unit cell surrounded by a
    lossless dielectric background (eps = 4).
    """

    LX = 1000.0  # nm
    LY = 1000.0  # nm
    SLIT_WIDTH = 200.0  # nm
    EPS_BG = 4.0
    THICKNESS = 300.0  # nm
    ORDER = [5, 0]

    @pytest.fixture(scope="class")
    def batch_sim(self):
        s = _make_sim(freq=FREQS, order=self.ORDER, L=[self.LX, self.LY])
        geo = solwa.geometry(
            Lx=self.LX, Ly=self.LY, nx=100, ny=100,
            edge_sharpness=1000.0, dtype=GEO_DTYPE, device=DEVICE,
        )
        opening = geo.rectangle(
            Wx=self.SLIT_WIDTH, Wy=self.LY, Cx=self.LX / 2, Cy=self.LY / 2
        )
        layer_eps = opening * 1.0 + (1.0 - opening) * self.EPS_BG
        s.add_layer(thickness=self.THICKNESS, eps=layer_eps)
        s.solve_global_smatrix()
        return s

    @pytest.fixture(scope="class")
    def scalar_sims(self):
        geo = solwa.geometry(
            Lx=self.LX, Ly=self.LY, nx=100, ny=100,
            edge_sharpness=1000.0, dtype=GEO_DTYPE, device=DEVICE,
        )
        opening = geo.rectangle(
            Wx=self.SLIT_WIDTH, Wy=self.LY, Cx=self.LX / 2, Cy=self.LY / 2
        )
        layer_eps = opening * 1.0 + (1.0 - opening) * self.EPS_BG
        sims = []
        for lam in WAVELENGTHS:
            s = _make_sim(freq=1.0 / lam, order=self.ORDER, L=[self.LX, self.LY])
            s.add_layer(thickness=self.THICKNESS, eps=layer_eps)
            s.solve_global_smatrix()
            sims.append(s)
        return sims

    def test_batch_matches_scalar(self, batch_sim, scalar_sims):
        """Batch results match scalar per-wavelength simulations."""
        for b, s in enumerate(scalar_sims):
            for m in (0, 1):
                T_batch = _T_batch(batch_sim, b, m)
                T_scalar = _T_scalar(s, m)
                assert abs(T_batch - T_scalar) < 1e-5, (
                    f"λ={WAVELENGTHS[b]} nm, m={m}: "
                    f"batch T={T_batch:.6f}, scalar T={T_scalar:.6f}"
                )

    def test_energy_conservation_per_wavelength(self, batch_sim):
        """T + R = 1 summed over propagating orders for each wavelength.

        Reference: Petit (ed.), "Electromagnetic Theory of Gratings" (1980), Ch. 1.
        """
        for b in range(len(WAVELENGTHS)):
            total = _energy_balance_batch(batch_sim, b, max_order=self.ORDER[0])
            assert abs(total - 1.0) < 1e-4, (
                f"λ={WAVELENGTHS[b]} nm: T+R={total:.6f}"
            )


# ---------------------------------------------------------------------------
# 3. Double slit
# ---------------------------------------------------------------------------


class TestBatchDoubleSlit:
    """Batch version of TestDoubleSlit.

    Two identical mirror-symmetric transparent slits per unit cell.
    """

    LX = 2000.0  # nm
    LY = 1000.0  # nm
    SLIT_WIDTH = 200.0  # nm
    EPS_BG = 4.0
    THICKNESS = 300.0  # nm
    ORDER = [7, 0]

    @pytest.fixture(scope="class")
    def batch_sim(self):
        s = _make_sim(freq=FREQS, order=self.ORDER, L=[self.LX, self.LY])
        geo = solwa.geometry(
            Lx=self.LX, Ly=self.LY, nx=200, ny=100,
            edge_sharpness=1000.0, dtype=GEO_DTYPE, device=DEVICE,
        )
        slit_A = geo.rectangle(
            Wx=self.SLIT_WIDTH, Wy=self.LY, Cx=self.LX / 4, Cy=self.LY / 2
        )
        slit_B = geo.rectangle(
            Wx=self.SLIT_WIDTH, Wy=self.LY, Cx=3 * self.LX / 4, Cy=self.LY / 2
        )
        layer_eps = geo.union(slit_A, slit_B) * 1.0 + (1.0 - geo.union(slit_A, slit_B)) * self.EPS_BG
        s.add_layer(thickness=self.THICKNESS, eps=layer_eps)
        s.solve_global_smatrix()
        return s

    @pytest.fixture(scope="class")
    def scalar_sims(self):
        geo = solwa.geometry(
            Lx=self.LX, Ly=self.LY, nx=200, ny=100,
            edge_sharpness=1000.0, dtype=GEO_DTYPE, device=DEVICE,
        )
        slit_A = geo.rectangle(
            Wx=self.SLIT_WIDTH, Wy=self.LY, Cx=self.LX / 4, Cy=self.LY / 2
        )
        slit_B = geo.rectangle(
            Wx=self.SLIT_WIDTH, Wy=self.LY, Cx=3 * self.LX / 4, Cy=self.LY / 2
        )
        double_slit = geo.union(slit_A, slit_B)
        layer_eps = double_slit * 1.0 + (1.0 - double_slit) * self.EPS_BG
        sims = []
        for lam in WAVELENGTHS:
            s = _make_sim(freq=1.0 / lam, order=self.ORDER, L=[self.LX, self.LY])
            s.add_layer(thickness=self.THICKNESS, eps=layer_eps)
            s.solve_global_smatrix()
            sims.append(s)
        return sims

    def test_batch_matches_scalar(self, batch_sim, scalar_sims):
        """Batch results match scalar per-wavelength simulations."""
        for b, s in enumerate(scalar_sims):
            for m in (0, 2):
                T_batch = _T_batch(batch_sim, b, m)
                T_scalar = _T_scalar(s, m)
                assert abs(T_batch - T_scalar) < 1e-5, (
                    f"λ={WAVELENGTHS[b]} nm, m={m}: "
                    f"batch T={T_batch:.6f}, scalar T={T_scalar:.6f}"
                )

    def test_first_order_destructive_interference_per_wavelength(self, batch_sim):
        """The ±1st orders vanish due to the double-slit structure factor, for every wavelength.

        The structure factor at m=±1 is zero because the two slits are placed at
        x = Λ/4 and x = 3Λ/4, giving exp(2πi·m·x1/Λ) + exp(2πi·m·x2/Λ) = 0
        at m = ±1.  This is purely geometric and wavelength-independent.

        Reference: Hecht, "Optics", §10.2.
        """
        for b in range(len(WAVELENGTHS)):
            assert _T_batch(batch_sim, b, +1) < 1e-3, (
                f"λ={WAVELENGTHS[b]} nm: T[+1]={_T_batch(batch_sim, b, +1):.6f}"
            )
            assert _T_batch(batch_sim, b, -1) < 1e-3, (
                f"λ={WAVELENGTHS[b]} nm: T[-1]={_T_batch(batch_sim, b, -1):.6f}"
            )

    def test_mirror_symmetry_per_wavelength(self, batch_sim):
        """Mirror symmetry |T[+m]| = |T[-m]| holds for every wavelength in the batch.

        Reference: Moharam & Gaylord (1981), J. Opt. Soc. Am. 71, §III.
        """
        for b in range(len(WAVELENGTHS)):
            for m in (1, 2, 3):
                diff = abs(_T_batch(batch_sim, b, +m) - _T_batch(batch_sim, b, -m))
                assert diff < 1e-5, (
                    f"λ={WAVELENGTHS[b]} nm, m={m}: "
                    f"T[+m]={_T_batch(batch_sim, b, +m):.6f}, "
                    f"T[-m]={_T_batch(batch_sim, b, -m):.6f}"
                )

    def test_energy_conservation_per_wavelength(self, batch_sim):
        """T + R = 1 for each wavelength in the batch."""
        for b in range(len(WAVELENGTHS)):
            total = _energy_balance_batch(batch_sim, b, max_order=self.ORDER[0])
            assert abs(total - 1.0) < 1e-4, (
                f"λ={WAVELENGTHS[b]} nm: T+R={total:.6f}"
            )


# ---------------------------------------------------------------------------
# 4. Binary lamellar grating
# ---------------------------------------------------------------------------


class TestBatchBinaryGrating:
    """Batch version of TestBinaryGrating.

    50 % fill-factor lamellar grating: alternating strips of air (eps = 1)
    and dielectric (eps = 4) with period 1000 nm.
    """

    PERIOD = 1000.0  # nm
    LY = 1000.0  # nm
    EPS_HIGH = 4.0
    THICKNESS = 300.0  # nm
    ORDER = [3, 1]

    @pytest.fixture(scope="class")
    def batch_sim(self):
        s = _make_sim(freq=FREQS, order=self.ORDER, L=[self.PERIOD, self.LY])
        geo = solwa.geometry(
            Lx=self.PERIOD, Ly=self.LY, nx=100, ny=100,
            edge_sharpness=1000.0, dtype=GEO_DTYPE, device=DEVICE,
        )
        stripe = geo.rectangle(
            Wx=self.PERIOD / 2, Wy=self.LY,
            Cx=self.PERIOD / 2, Cy=self.LY / 2,
        )
        layer_eps = stripe * 1.0 + (1.0 - stripe) * self.EPS_HIGH
        s.add_layer(thickness=self.THICKNESS, eps=layer_eps)
        s.solve_global_smatrix()
        return s

    @pytest.fixture(scope="class")
    def scalar_sims(self):
        geo = solwa.geometry(
            Lx=self.PERIOD, Ly=self.LY, nx=100, ny=100,
            edge_sharpness=1000.0, dtype=GEO_DTYPE, device=DEVICE,
        )
        stripe = geo.rectangle(
            Wx=self.PERIOD / 2, Wy=self.LY,
            Cx=self.PERIOD / 2, Cy=self.LY / 2,
        )
        layer_eps = stripe * 1.0 + (1.0 - stripe) * self.EPS_HIGH
        sims = []
        for lam in WAVELENGTHS:
            s = _make_sim(freq=1.0 / lam, order=self.ORDER, L=[self.PERIOD, self.LY])
            s.add_layer(thickness=self.THICKNESS, eps=layer_eps)
            s.solve_global_smatrix()
            sims.append(s)
        return sims

    def test_batch_matches_scalar(self, batch_sim, scalar_sims):
        """Batch results match scalar per-wavelength simulations."""
        for b, s in enumerate(scalar_sims):
            for m in (0, 1):
                T_batch = _T_batch(batch_sim, b, m)
                T_scalar = _T_scalar(s, m)
                assert abs(T_batch - T_scalar) < 1e-5, (
                    f"λ={WAVELENGTHS[b]} nm, m={m}: "
                    f"batch T={T_batch:.6f}, scalar T={T_scalar:.6f}"
                )

    def test_grating_equation_per_wavelength(self, batch_sim):
        """The first diffraction order satisfies sin(θ_1) = λ/d for each wavelength.

        Ground-truth: grating equation at normal incidence, θ_m = arcsin(m·λ/d).
        Each batch element corresponds to a different wavelength, so the expected
        angle is wavelength-dependent.

        Reference: Born & Wolf, "Principles of Optics", §8.6.
        """
        inc_angle, _ = batch_sim.diffraction_angle(
            [[1, 0]], layer="output", unit="degree"
        )
        # inc_angle shape: [B, 1] for batch freq
        for b, lam in enumerate(WAVELENGTHS):
            expected_deg = asin(lam / self.PERIOD) * 180 / pi
            assert abs(inc_angle[b, 0].item() - expected_deg) < 1e-3, (
                f"λ={lam} nm: RCWA θ={inc_angle[b, 0].item():.4f}°, "
                f"expected={expected_deg:.4f}°"
            )

    def test_energy_conservation_per_wavelength(self, batch_sim):
        """T + R = 1 for each wavelength in the batch."""
        for b in range(len(WAVELENGTHS)):
            total = _energy_balance_batch(batch_sim, b, max_order=self.ORDER[0])
            assert abs(total - 1.0) < 1e-4, (
                f"λ={WAVELENGTHS[b]} nm: T+R={total:.6f}"
            )
