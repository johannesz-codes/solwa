"""Li factorization against nannos, an independent RCWA code.

For a 1D lamellar grating Li's rule reduces to the inverse rule for the
component normal to the lamellae, which is what nannos' "tangent" formulation
does. Its per-order efficiencies are stored in ``reference_nannos_li.json``
(nannos pulls in ~1 GB of plotting dependencies, too much for CI). Regenerate
after changing the setup below with

    python tests/test_reference_nannos.py   # needs nannos==2.6.5

Both orientations are checked (lamellae normal to x and to y), which covers
both branches of the 2D rule, at planar and conical incidence, p and s.
"""

import json
import math
import pathlib

import numpy as np
import pytest
import torch

import solwa

REF = pathlib.Path(__file__).with_name("reference_nannos_li.json")
M, NX, H = 10, 512, 0.5
EPS_METAL, EPS_OUT, THETA = (0.22 + 6.71j) ** 2, 2.25, 0.3
ORDERS = range(-3, 4)
CASES = [(phi, pol) for phi in (0.0, 30.0) for pol in "ps"]


def _grid():
    x = (np.arange(NX) + 0.5) / NX
    return np.where(x < 0.5, EPS_METAL, 1.0 + 0j)


@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("phi, pol", CASES)
def test_li_matches_nannos(phi, pol, rotated, device):
    ref = json.loads(REF.read_text())[f"{phi}{pol}"]
    eps = torch.as_tensor(_grid(), device=device)
    sim = solwa.rcwa(
        freq=1.0, order=[0, M] if rotated else [M, 0], L=[1.0, 1.0],
        dtype=torch.complex128, device=device, factorization="li",
    )  # fmt: skip
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=EPS_OUT)
    azi = math.radians(phi) + (math.pi / 2 if rotated else 0.0)
    sim.set_incident_angle(inc_ang=THETA, azi_ang=azi)
    sim.add_layer(thickness=H, eps=eps.reshape((1, NX) if rotated else (NX, 1)))
    sim.solve_global_smatrix()
    orders = [[0, m] if rotated else [m, 0] for m in ORDERS]
    for port in ("r", "t"):
        eff = sum(
            sim.S_parameters(orders, port=port, polarization=out + pol).abs() ** 2
            for out in "ps"
        )
        np.testing.assert_allclose(eff.cpu().numpy(), ref[port], rtol=0, atol=1e-12)


if __name__ == "__main__":
    import nannos as nn

    out = {}
    for phi, pol in CASES:
        lat = nn.Lattice(1.0, discretization=NX)
        eps = lat.ones() * (1.0 + 0j)
        eps[:] = _grid()[:, None] if eps.ndim == 2 else _grid()
        layers = [
            lat.Layer("in", epsilon=1.0),
            lat.Layer("grating", thickness=H, epsilon=eps),
            lat.Layer("out", epsilon=EPS_OUT),
        ]
        wave = nn.PlaneWave(
            wavelength=1.0, angles=(math.degrees(THETA), phi, 0 if pol == "p" else 90)
        )
        sim = nn.Simulation(layers, wave, nh=2 * M + 1, formulation="tangent")
        R, T = sim.diffraction_efficiencies(orders=True)
        idx = [list(np.asarray(sim.harmonics)[0]).index(m) for m in ORDERS]
        out[f"{phi}{pol}"] = {
            "r": np.real(R)[idx].tolist(),
            "t": np.real(T)[idx].tolist(),
        }
    REF.write_text(json.dumps(out, indent=1) + "\n")
