"""
Tests for material grid size validation in _material_conv.

Ensures that a clear ValueError is raised when an inhomogeneous material grid
is too small for the selected RCWA order, instead of producing out-of-bounds
indexing errors (which surface as CUDA device-side asserts on GPU).
"""

import pytest
import torch

import solwa


DEVICE = torch.device("cpu")
DTYPE = torch.complex64


def _make_sim(order):
    """Create a minimal RCWA simulation with the given order."""
    freq = 1.0 / 532.0  # arbitrary frequency
    L = [1000.0, 1000.0]
    sim = solwa.rcwa(freq=freq, order=order, L=L, dtype=DTYPE, device=DEVICE)
    sim.add_input_layer(eps=1.0)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    return sim


class TestMaterialGridValidation:
    """Validation of inhomogeneous material grid size."""

    def test_too_small_grid_raises_valueerror(self):
        """A grid too small for the order should raise ValueError."""
        sim = _make_sim(order=[5, 5])
        # order=5 means orders from -5 to 5, requiring grid > 10 in each dim.
        # A 10x10 grid is exactly at the boundary (needs > 10), so it should fail.
        small_material = torch.ones(10, 10, dtype=DTYPE, device=DEVICE) * 2.25
        with pytest.raises(ValueError, match="too small"):
            sim._material_conv(small_material)

    def test_too_small_grid_y_raises_valueerror(self):
        """A grid too small only in y should still raise ValueError."""
        sim = _make_sim(order=[5, 5])
        # x dimension is fine (>10), y dimension is too small (10)
        material = torch.ones(11, 10, dtype=DTYPE, device=DEVICE) * 2.25
        with pytest.raises(ValueError, match="too small"):
            sim._material_conv(material)

    def test_too_small_grid_x_raises_valueerror(self):
        """A grid too small only in x should still raise ValueError."""
        sim = _make_sim(order=[5, 5])
        material = torch.ones(10, 11, dtype=DTYPE, device=DEVICE) * 2.25
        with pytest.raises(ValueError, match="too small"):
            sim._material_conv(material)

    def test_sufficient_grid_does_not_raise(self):
        """A grid large enough should not raise."""
        sim = _make_sim(order=[5, 5])
        # order=5 needs grid > 10, so 11x11 is sufficient
        material = torch.ones(11, 11, dtype=DTYPE, device=DEVICE) * 2.25
        result = sim._material_conv(material)
        assert result is not None
        assert result.shape[0] == result.shape[1]

    def test_large_order_small_grid_raises(self):
        """Reproduces the issue: order=32 with a 64x63 grid."""
        sim = _make_sim(order=[32, 32])
        # order=32 means differences up to 64; needs grid > 64 in each dim.
        # 64x63 is too small (64 is not > 64, 63 is not > 64).
        material = torch.ones(64, 63, dtype=DTYPE, device=DEVICE) * 2.25
        with pytest.raises(ValueError, match="too small"):
            sim._material_conv(material)

    def test_error_message_contains_useful_info(self):
        """Error message should mention shapes and orders."""
        sim = _make_sim(order=[3, 4])
        material = torch.ones(5, 5, dtype=DTYPE, device=DEVICE) * 2.25
        with pytest.raises(ValueError, match=r"order_x=\[-3, \.\.\., 3\]"):
            sim._material_conv(material)
