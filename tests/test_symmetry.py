"""Correctness and validation tests for mirror-parity solver reduction."""

import pytest
import torch

import solwa


def _symmetric_material(axis, size=20, dtype=torch.float32):
    material = torch.ones((size, size), dtype=dtype)
    mirror_start = size // 3
    mirror_stop = size - mirror_start
    if axis == "x":
        material[size // 4 : 3 * size // 4, mirror_start:mirror_stop] = 2.4
    else:
        material[mirror_start:mirror_stop, size // 4 : 3 * size // 4] = 2.4
    return material


def _simulation(axis=None, *, thickness=110.0, material=None):
    sim = solwa.rcwa(
        freq=1 / 520,
        order=[1, 1],
        L=[410, 430],
        dtype=torch.complex64,
        device=torch.device("cpu"),
        symmetry_axis=axis,
    )
    sim.add_input_layer(eps=1.2)
    sim.add_output_layer(eps=1.5)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(
        thickness=thickness,
        eps=_symmetric_material(axis or "x") if material is None else material,
    )
    sim.add_layer(thickness=45.0, eps=1.8)
    sim.solve_global_smatrix()
    return sim


@pytest.mark.parametrize("axis", ["x", "y"])
def test_symmetry_blocks_match_full_solver(axis):
    material = _symmetric_material(axis)
    full = _simulation(material=material)
    reduced = _simulation(axis, material=material)

    assert reduced._symmetry_block_sizes == (reduced.order_N, reduced.order_N)
    for full_matrix, reduced_matrix in zip(full.S, reduced.S):
        torch.testing.assert_close(reduced_matrix, full_matrix, rtol=3e-5, atol=3e-5)


def test_symmetry_preserves_thickness_gradient():
    material = _symmetric_material("x")
    full_thickness = torch.tensor(105.0, requires_grad=True)
    reduced_thickness = full_thickness.detach().clone().requires_grad_(True)
    full = _simulation(thickness=full_thickness, material=material)
    reduced = _simulation(
        "x", thickness=reduced_thickness, material=material
    )

    center = full.order_N // 2
    full_loss = torch.abs(full.S[0][center, center]) ** 2
    reduced_loss = torch.abs(reduced.S[0][center, center]) ** 2
    full_loss.backward()
    reduced_loss.backward()

    torch.testing.assert_close(reduced_loss, full_loss, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(
        reduced_thickness.grad, full_thickness.grad, rtol=2e-4, atol=2e-6
    )


def test_lazy_full_basis_reconstruction_preserves_fields():
    material = _symmetric_material("x")
    full = _simulation(material=material)
    reduced = _simulation("x", material=material)
    full.source_planewave()
    reduced.source_planewave()
    x_axis = torch.linspace(0.0, 410.0, 4)
    y_axis = torch.linspace(0.0, 430.0, 5)

    full_fields = full.field_xy(0, x_axis, y_axis, z_prop=20.0)
    reduced_fields = reduced.field_xy(0, x_axis, y_axis, z_prop=20.0)

    for full_group, reduced_group in zip(full_fields, reduced_fields):
        for full_component, reduced_component in zip(full_group, reduced_group):
            torch.testing.assert_close(
                reduced_component, full_component, rtol=4e-5, atol=1e-5
            )


def test_rejects_material_that_breaks_requested_symmetry():
    material = torch.ones((16, 16))
    material[2:5, 3:8] = 2.0
    sim = solwa.rcwa(1 / 500, [1, 1], [400, 400], symmetry_axis="x", device="cpu")
    sim.set_incident_angle(0.0, 0.0)

    with pytest.raises(ValueError, match="not mirror symmetric"):
        sim.add_layer(100.0, material)


def test_rejects_incidence_normal_to_mirror_axis():
    sim = solwa.rcwa(1 / 500, [1, 1], [400, 400], symmetry_axis="y", device="cpu")

    with pytest.raises(ValueError, match="requires kx0=0"):
        sim.set_incident_angle(0.2, 0.0)


def test_rejects_inconsistent_pattern_sampling():
    sim = solwa.rcwa(1 / 500, [1, 1], [400, 400], symmetry_axis="x", device="cpu")
    sim.set_incident_angle(0.0, 0.0)
    sim.add_layer(100.0, _symmetric_material("x", 16))

    with pytest.raises(ValueError, match="same sampling count"):
        sim.add_layer(100.0, _symmetric_material("x", 18))
