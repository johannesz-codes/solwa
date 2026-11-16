import pytest
import torch

import solwa


def test_material_grid_too_small_raises():
    """Grid must have > 2*order samples per axis; batched (wavelength) eps is checked on its last two dims."""
    freq = torch.tensor([1 / 500.0, 1 / 600.0])
    sim = solwa.rcwa(
        freq=freq, order=[5, 5], L=[1000.0, 1000.0], dtype=torch.complex128
    )
    sim.add_input_layer(eps=1.0)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    for shape in ([2, 10, 11], [2, 11, 10]):
        with pytest.raises(ValueError, match="too small"):
            sim.add_layer(
                thickness=100.0, eps=torch.full(shape, 2.25, dtype=torch.complex128)
            )
    sim.add_layer(
        thickness=100.0, eps=torch.full([2, 11, 11], 2.25, dtype=torch.complex128)
    )
