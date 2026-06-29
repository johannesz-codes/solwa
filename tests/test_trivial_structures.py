"""
RCWA tests using trivial photonic structures.

Each test validates a well-known analytical or symmetry property:
  - Homogeneous glass slab  : energy conservation and Fabry-Perot formula
  - Single slit             : energy conservation
  - Double slit             : destructive interference at m=±1, mirror symmetry,
                              and energy conservation
  - Binary lamellar grating : grating equation and energy conservation
  - Absorbing slab          : near-field absorption (Poynting vector) vs TMM

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
        """T + R = 1 for a lossless dielectric slab (only 0th order propagates).

        Ground-truth formula: energy conservation (optical theorem) for a
        lossless (Im ε = 0) medium.  The cell period is smaller than λ, so
        only the zeroth diffraction order is propagating; the total power
        balance therefore reduces to

            T[0] + R[0] = 1.

        Reference: any EM textbook, e.g. Born & Wolf §1.5.
        """
        total = _energy_balance(sim, max_order=0)
        assert abs(total - 1.0) < 1e-4

    def test_fabry_perot_agreement(self, sim):
        """Zeroth-order transmittance matches the Fabry-Perot formula.

        For a lossless slab (refractive index n1, thickness d) surrounded by
        air (n0 = n2 = 1) at normal incidence, the total transmitted amplitude
        is given by the Fabry-Perot etalon formula (summing all round-trip
        reflections as a geometric series):

            t_total = t01 * t12 * exp(i*φ) / (1 - r10 * r12 * exp(2i*φ))

        where the Fresnel amplitude coefficients at normal incidence are:

            t01 = 2*n0 / (n0 + n1)  →  2 / (1 + n1)       (air → slab)
            t12 = 2*n1 / (n1 + n2)  →  2*n1 / (n1 + 1)    (slab → air)
            r10 = (n1 - n0) / (n1 + n0)  →  (n1 - 1) / (n1 + 1)
            r12 = (n1 - n2) / (n1 + n2)  →  (n1 - 1) / (n1 + 1)

        and φ = 2π n1 d / λ is the single-pass optical phase inside the slab.

        Since n0 = n2 the power transmittance is T = |t_total|².

        References: Born & Wolf, "Principles of Optics", §1.6 (Fabry-Perot
        etalon); Hecht, "Optics", §9.6 (multiple-beam interference).
        """
        n1 = self.N_GLASS
        d = self.THICKNESS
        # Fresnel amplitude coefficients (air n0=1, slab n1, air n2=1):
        #   t01 = 2/(1+n1),  t12 = 2*n1/(n1+1)
        #   r10 = (n1-1)/(n1+1),  r12 = (n1-1)/(n1+1)
        t_product = (2.0 / (1 + n1)) * (2 * n1 / (n1 + 1))  # t01 * t12
        r_product = ((n1 - 1) / (n1 + 1)) ** 2  # r10 * r12
        phi = 2 * pi * n1 * d / LAMBDA  # single-pass optical phase
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
        """T + R = 1 summed over all propagating diffraction orders.

        Ground-truth formula: energy conservation for a lossless medium.
        For a periodic structure illuminated at normal incidence the total
        scattered power must equal the incident power:

            Σ_m T[m] + Σ_m R[m] = 1

        where the sum runs over all diffraction orders m for which the
        in-plane wave-vector |k_x + m·G| ≤ k0 (propagating orders only).
        Evanescent orders carry no net time-averaged power.

        Reference: Petit (ed.), "Electromagnetic Theory of Gratings" (1980),
        Ch. 1; also standard result in coupled-wave theory.
        """
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
        """The ±1st orders vanish due to destructive interference.

        Ground-truth formula: double-slit structure factor.
        For two identical slits at positions x1 = Λ/4 and x2 = 3Λ/4
        the Fourier (structure) factor for grating order m is:

            S(m) = exp(2πi·m·x1/Λ) + exp(2πi·m·x2/Λ)
                 = exp(iπm/2) + exp(i3πm/2)
                 = exp(iπm/2) [1 + exp(iπm)]

        At m = ±1:  1 + exp(±iπ) = 1 − 1 = 0  →  S(±1) = 0.

        Because the scattered field amplitude is proportional to S(m), the
        transmitted (and reflected) power in those orders is zero.

        Reference: Hecht, "Optics", §10.2 (multiple-slit diffraction).
        """
        assert _T(sim, +1) < 1e-3
        assert _T(sim, -1) < 1e-3

    def test_mirror_symmetry(self, sim):
        """Mirror symmetry forces |T[+m]| = |T[-m]| for all m at normal incidence.

        Ground-truth formula: symmetry argument on the Fourier coefficients.
        A permittivity profile ε(x) that is even about x = Λ/2 (i.e.,
        ε(Λ/2 + δ) = ε(Λ/2 − δ)) has real Fourier coefficients ε̂_m = ε̂_{-m}*.
        At normal incidence (k_inc = 0), the ±m grating orders are excited
        symmetrically, so the S-matrix elements satisfy

            |T[+m]| = |T[−m]|   and   |R[+m]| = |R[−m]|.

        Reference: standard symmetry argument; see Moharam & Gaylord (1981),
        J. Opt. Soc. Am. 71, §III.
        """
        for m in (1, 2, 3):
            assert abs(_T(sim, +m) - _T(sim, -m)) < 1e-5

    def test_energy_conservation(self, sim):
        """T + R = 1 for the lossless double-slit structure.

        Same ground-truth formula as TestSingleSlit.test_energy_conservation:
        energy conservation summed over all propagating diffraction orders,

            Σ_m T[m] + Σ_m R[m] = 1,

        valid for any lossless (Im ε = 0) periodic structure.
        """
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
        s = _make_sim(freq=1 / LAMBDA, order=self.ORDER, L=[self.PERIOD, self.LY])
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
        """The first transmitted diffraction order satisfies the grating equation.

        Ground-truth formula: the grating equation at normal incidence (θ_inc = 0):

            sin θ_m = m · λ / d

        where d is the grating period and m is the diffraction order index.
        At m = 1 and with λ = 532 nm, d = 1000 nm:

            θ_1 = arcsin(532 / 1000) ≈ 32.12°.

        This kinematic relation depends only on the geometry (λ, d) and is
        completely independent of the grating material or fill factor.

        Reference: Born & Wolf, "Principles of Optics", §8.6 (grating equation);
        Hecht, "Optics", §10.2.
        """
        angle_deg, _ = sim.diffraction_angle([[1, 0]], layer="output", unit="degree")
        expected_deg = asin(LAMBDA / self.PERIOD) * 180 / pi
        assert abs(angle_deg.item() - expected_deg) < 1e-3

    def test_energy_conservation(self, sim):
        """T + R = 1 for the lossless binary grating.

        Same ground-truth formula as the other energy-conservation tests:

            Σ_m T[m] + Σ_m R[m] = 1,

        where the sum runs over all propagating diffraction orders.  With
        d = 1000 nm and λ = 532 nm, orders |m| ≤ 1 are propagating
        (sin θ_1 = 532/1000 < 1); higher orders are evanescent.
        """
        total = _energy_balance(sim, max_order=self.ORDER[0])
        assert abs(total - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# 5. Near-field absorption: Poynting vector vs. Transfer Matrix Method
# ---------------------------------------------------------------------------


class TestAbsorbingSlabPoynting:
    """
    Cross-validate the near-field Poynting-vector absorption against the
    transfer-matrix method (TMM) for a homogeneous absorbing slab.

    The structure is a single lossy dielectric layer (complex ε) in air.
    Because the layer is homogeneous, RCWA reduces to the TMM exactly, so
    both approaches must give the same absorbed power fraction.

    Two independent methods are compared:

    1. **Poynting-vector method** (near-field):
       Absorption = (Sz_flux at layer top − Sz_flux at layer bottom) / P_inc,
       where the fluxes are obtained by integrating the time-averaged
       Poynting vector Sz = ½ Re(E × H*)_z over the unit cell.

    2. **Transfer-Matrix Method** (TMM, analytic reference):
       Absorption = 1 − R_TMM − T_TMM,
       where R_TMM and T_TMM follow from the Fabry-Perot scattering formula
       for a slab with complex refractive index (see test docstring below).
    """

    EPS_ABS = complex(4.0, 1.2)   # lossy dielectric: Re(ε)=4, Im(ε)=1.2
    THICKNESS = 200.0              # nm
    L = [300.0, 300.0]             # nm, unit-cell period (< λ → 0th order only)
    ORDER = [3, 3]
    NX, NY = 64, 64                # integration grid points

    @pytest.fixture
    def sim(self):
        s = _make_sim(freq=1 / LAMBDA, order=self.ORDER, L=self.L)
        s.add_layer(thickness=self.THICKNESS, eps=self.EPS_ABS)
        s.solve_global_smatrix()
        s.source_planewave(amplitude=[1.0, 0.0], direction="forward")
        return s

    @staticmethod
    def _tmm_absorption(eps1, d, lam):
        """
        Analytic absorption fraction for a homogeneous slab (TMM).

        For a slab with complex permittivity ε₁ (n₁ = √ε₁) and thickness d,
        surrounded by air (n₀ = n₂ = 1), at normal incidence, the
        Fabry-Perot scattering formula gives:

            δ  = 2π n₁ d / λ                   (single-pass optical phase)

            r₀₁ = (1 − n₁)/(1 + n₁)            (air→slab Fresnel reflection)
            r₁₀ = −r₀₁                          (slab→air, reciprocity)
            t₀₁ = 2/(1 + n₁)                   (air→slab transmission)
            t₁₀ = 2 n₁/(n₁ + 1)               (slab→air transmission)
            r₁₂ = (n₁ − 1)/(n₁ + 1)            (slab→output Fresnel reflection)
            t₁₂ = 2 n₁/(n₁ + 1)               (slab→output transmission)

            r_total = r₀₁ + t₀₁ t₁₀ r₁₂ exp(2iδ) / (1 − r₁₀ r₁₂ exp(2iδ))
            t_total = t₀₁ t₁₂ exp(iδ)         / (1 − r₁₀ r₁₂ exp(2iδ))

        Since n₀ = n₂ = 1 (real), the power coefficients are
            R = |r_total|²,   T = |t_total|²,   A = 1 − R − T.

        References: Born & Wolf, "Principles of Optics", §1.6; Hecht,
        "Optics", §9.6.
        """
        n1 = cmath.sqrt(eps1)
        delta = 2 * pi * n1 * d / lam
        r01 = (1 - n1) / (1 + n1)
        r10 = -r01
        t01 = 2 / (1 + n1)
        t10 = 2 * n1 / (n1 + 1)
        r12 = (n1 - 1) / (n1 + 1)
        t12 = 2 * n1 / (n1 + 1)
        denom = 1 - r10 * r12 * cmath.exp(2j * delta)
        r_tot = r01 + t01 * t10 * r12 * cmath.exp(2j * delta) / denom
        t_tot = t01 * t12 * cmath.exp(1j * delta) / denom
        R = abs(r_tot) ** 2
        T = abs(t_tot) ** 2  # n0 = n2 = 1, so no refractive-index correction
        return 1.0 - R - T

    def test_near_field_absorption_matches_tmm(self, sim):
        """Near-field (Poynting) absorption agrees with the TMM to < 0.1 %.

        Ground-truth formula: transfer-matrix method for a homogeneous
        absorbing slab (see ``_tmm_absorption`` for the full derivation).

        The Poynting-vector absorption is computed as the drop in z-directed
        Poynting flux across the layer, normalised by the incident power:

            A_Poynting = (∫∫ Sz(z=0) dx dy − ∫∫ Sz(z=d) dx dy) / P_inc

        where P_inc = ½ Lx Ly for a unit-amplitude x-polarised plane wave
        at normal incidence in vacuum.

        Because the layer is homogeneous, RCWA is equivalent to the TMM and
        both values must coincide to within floating-point rounding
        (complex64 ≈ 7 significant digits).

        References: Hecht, "Optics", §13.2 (energy absorption in dielectrics);
        Born & Wolf, §1.5 (Poynting vector and energy flow).
        """
        Lx, Ly = self.L
        x_axis = torch.linspace(0.0, Lx, self.NX, dtype=torch.float32, device=DEVICE)
        y_axis = torch.linspace(0.0, Ly, self.NY, dtype=torch.float32, device=DEVICE)
        P_inc = 0.5 * Lx * Ly

        flux_entrance = sim.poynting_flux(0, x_axis, y_axis, z_prop=0.0)
        flux_exit = sim.poynting_flux(0, x_axis, y_axis, z_prop=self.THICKNESS)
        A_poynting = ((flux_entrance - flux_exit) / P_inc).item()

        A_tmm = self._tmm_absorption(self.EPS_ABS, self.THICKNESS, LAMBDA)

        assert abs(A_poynting - A_tmm) < 1e-4
