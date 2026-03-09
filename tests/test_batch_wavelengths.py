"""Tests for batch-wavelength (batch-frequency) processing in solwa.rcwa.

These tests verify that passing a 1-D batch tensor as ``freq`` produces results
that are numerically consistent with running the simulation once per wavelength
using a scalar ``freq``.
"""

import pytest
import torch
import solwa


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DTYPE = torch.complex64
DEVICE = torch.device("cpu")


def _build_scalar_sim(freq_val, order, L):
    """Return a fully-solved scalar rcwa for a single frequency."""
    sim = solwa.rcwa(freq=freq_val, order=order, L=L, dtype=DTYPE, device=DEVICE)
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=1.0)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=100.0, eps=2.25)
    sim.solve_global_smatrix()
    return sim


def _build_batch_sim(freqs, order, L):
    """Return a fully-solved batch rcwa for a 1-D tensor of frequencies."""
    sim = solwa.rcwa(freq=freqs, order=order, L=L, dtype=DTYPE, device=DEVICE)
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=1.0)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=100.0, eps=2.25)
    sim.solve_global_smatrix()
    return sim


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBatchInit:
    """rcwa can be constructed with a batch frequency tensor."""

    def test_batch_freq_stored_correctly(self):
        freqs = torch.tensor([1 / 500, 1 / 600], dtype=DTYPE, device=DEVICE)
        sim = solwa.rcwa(freq=freqs, order=[1, 1], L=[300, 300], dtype=DTYPE, device=DEVICE)
        assert sim.freq.shape == (2,)
        assert sim.omega.shape == (2,)

    def test_scalar_freq_unchanged(self):
        sim = solwa.rcwa(freq=1 / 500, order=[1, 1], L=[300, 300], dtype=DTYPE, device=DEVICE)
        assert sim.freq.dim() == 0
        assert sim.omega.dim() == 0


class TestBatchKVectors:
    """Batch k-vectors have the expected shapes."""

    def test_kx_norm_dn_shape(self):
        B = 3
        freqs = torch.tensor([1 / 500, 1 / 600, 1 / 700], dtype=DTYPE, device=DEVICE)
        sim = solwa.rcwa(freq=freqs, order=[2, 2], L=[300, 300], dtype=DTYPE, device=DEVICE)
        sim.add_input_layer(eps=1.0)
        sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
        N = (2 * 2 + 1) ** 2  # 25 modes for order=[2,2]
        assert sim.Kx_norm_dn.shape == (B, N)
        assert sim.Ky_norm_dn.shape == (B, N)
        assert sim.Kx_norm.shape == (B, N, N)
        assert sim.Ky_norm.shape == (B, N, N)

    def test_scalar_kx_norm_dn_shape(self):
        sim = solwa.rcwa(freq=1 / 500, order=[2, 2], L=[300, 300], dtype=DTYPE, device=DEVICE)
        sim.add_input_layer(eps=1.0)
        sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
        N = 25
        assert sim.Kx_norm_dn.shape == (N,)
        assert sim.Kx_norm.shape == (N, N)


class TestBatchSParameters:
    """Batch S-parameter results match scalar results element-wise."""

    wavelengths = [400.0, 500.0, 600.0]
    order = [2, 2]
    L = [300, 300]

    @pytest.fixture(scope="class")
    def batch_sim(self):
        freqs = torch.tensor(
            [1 / w for w in self.wavelengths], dtype=DTYPE, device=DEVICE
        )
        return _build_batch_sim(freqs, self.order, self.L)

    @pytest.fixture(scope="class")
    def scalar_sims(self):
        return [
            _build_scalar_sim(1.0 / w, self.order, self.L) for w in self.wavelengths
        ]

    def test_batch_output_shape(self, batch_sim):
        T = batch_sim.S_parameters(
            orders=[0, 0],
            direction="forward",
            port="transmission",
            polarization="xx",
        )
        assert T.shape == (len(self.wavelengths), 1)

    def test_batch_matches_scalar_xx(self, batch_sim, scalar_sims):
        T_batch = batch_sim.S_parameters(
            orders=[0, 0],
            direction="forward",
            port="transmission",
            polarization="xx",
        )
        for i, s in enumerate(scalar_sims):
            T_scalar = s.S_parameters(
                orders=[0, 0],
                direction="forward",
                port="transmission",
                polarization="xx",
            )
            assert torch.allclose(T_batch[i], T_scalar, rtol=1e-4, atol=1e-6), (
                f"Mismatch at wavelength {self.wavelengths[i]} nm: "
                f"batch={T_batch[i]}, scalar={T_scalar}"
            )

    def test_batch_matches_scalar_reflection(self, batch_sim, scalar_sims):
        R_batch = batch_sim.S_parameters(
            orders=[0, 0],
            direction="forward",
            port="reflection",
            polarization="xx",
        )
        for i, s in enumerate(scalar_sims):
            R_scalar = s.S_parameters(
                orders=[0, 0],
                direction="forward",
                port="reflection",
                polarization="xx",
            )
            assert torch.allclose(R_batch[i], R_scalar, rtol=1e-4, atol=1e-6)

    def test_multiple_orders(self, batch_sim, scalar_sims):
        orders = [[0, 0], [1, 0], [-1, 0]]
        T_batch = batch_sim.S_parameters(
            orders=orders,
            direction="forward",
            port="transmission",
            polarization="xx",
        )
        assert T_batch.shape == (len(self.wavelengths), len(orders))
        for i, s in enumerate(scalar_sims):
            T_scalar = s.S_parameters(
                orders=orders,
                direction="forward",
                port="transmission",
                polarization="xx",
            )
            assert torch.allclose(T_batch[i], T_scalar, rtol=1e-4, atol=1e-6)


class TestBatchSParametersNoInputLayer:
    """Batch processing works without explicit input/output layers."""

    def test_no_input_layer(self):
        B = 2
        freqs = torch.tensor([1 / 500, 1 / 600], dtype=DTYPE, device=DEVICE)
        sim = solwa.rcwa(freq=freqs, order=[1, 1], L=[300, 300], dtype=DTYPE, device=DEVICE)
        sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
        sim.add_layer(thickness=50.0, eps=3.0)
        sim.solve_global_smatrix()
        T = sim.S_parameters(
            orders=[0, 0],
            direction="forward",
            port="transmission",
            polarization="xx",
            power_norm=False,
        )
        assert T.shape == (B, 1)


class TestBatchMaterialInterpolation:
    """Material.apply supports batch wavelengths."""

    @pytest.fixture
    def material(self, tmp_path):
        """Create a simple material data file for testing."""
        nk_file = tmp_path / "test_material.txt"
        nk_file.write_text(
            "400 1.5 0.0\n500 1.6 0.0\n600 1.7 0.0\n700 1.8 0.0\n"
        )
        return solwa.materials.Material(str(nk_file))

    def test_scalar_wavelength(self, material):
        wl = torch.tensor(500.0)
        nk = material.apply(wl)
        assert nk.shape == torch.Size([])

    def test_batch_wavelengths(self, material):
        wl = torch.tensor([400.0, 500.0, 600.0])
        nk = material.apply(wl)
        assert nk.shape == (3,)

    def test_batch_matches_scalar(self, material):
        wavelengths = [400.0, 500.0, 600.0, 700.0]
        batch = material.apply(torch.tensor(wavelengths))
        for i, w in enumerate(wavelengths):
            scalar = material.apply(torch.tensor(w))
            assert torch.allclose(batch[i], scalar, atol=1e-5), (
                f"Mismatch at wavelength {w}: batch={batch[i]}, scalar={scalar}"
            )

    def test_boundary_clamping_batch(self, material):
        """Values outside data range should be clamped to boundary values."""
        wl = torch.tensor([200.0, 500.0, 900.0])
        nk = material.apply(wl)
        assert nk.shape == (3,)
        # Below range → same as 400 nm value
        assert torch.allclose(nk[0], material.apply(torch.tensor(400.0)))
        # Above range → same as 700 nm value
        assert torch.allclose(nk[2], material.apply(torch.tensor(700.0)))
