"""Cross-check against grcwa, an independent RCWA code (numpy port of S4's method).

grcwa uses the Laurent rule on the same pixel grid and rectangular truncation,
so ``factorization="laurent"`` must reproduce it to rounding error: diffraction
efficiencies per order and the fields inside a patterned layer.
"""

import math

import numpy as np
import pytest
import grcwa
import torch

import solwa

M, NX = 4, 32
FREQ, THETA, PHI = 1 / 0.75, math.radians(20), math.radians(30)
EPS_IN, EPS_OUT = 1.0, 2.25


def _layers():
    x = (np.arange(NX) + 0.5) / NX
    X, Y = np.meshgrid(x, x, indexing="ij")
    rect = (np.abs(X - 0.4) < 0.25) & (np.abs(Y - 0.55) < 0.15)
    ellipse = (X - 0.5) ** 2 / 0.09 + (Y - 0.45) ** 2 / 0.04 < 1
    return [
        (0.3, np.where(rect, 6 + 0.3j, 1.0 + 0j)),
        (0.2, 2.0),
        (0.25, np.where(ellipse, 2.0 + 0j, 3.5 + 0.01j)),
    ]


def _grcwa(pol):
    obj = grcwa.obj((2 * M + 1) ** 2, [1.0, 0], [0, 1.0], FREQ, THETA, PHI, verbose=0)
    obj.Add_LayerUniform(0.1, EPS_IN)
    for h, eps in _layers():
        if np.ndim(eps):
            obj.Add_LayerGrid(h, NX, NX)
        else:
            obj.Add_LayerUniform(h, eps)
    obj.Add_LayerUniform(0.1, EPS_OUT)
    obj.Init_Setup(Gmethod=1)  # rectangular truncation |m|, |n| <= M
    obj.GridLayer_geteps(
        np.concatenate([e.ravel() for _, e in _layers() if np.ndim(e)])
    )
    obj.MakeExcitationPlanewave(
        *((1, 0, 0, 0) if pol == "p" else (0, 0, 1, 0)), order=0
    )
    return obj


def _solwa(device):
    sim = solwa.rcwa(
        freq=FREQ, order=[M, M], L=[1.0, 1.0], dtype=torch.complex128,
        device=device, factorization="laurent",
    )  # fmt: skip
    sim.add_input_layer(eps=EPS_IN)
    sim.add_output_layer(eps=EPS_OUT)
    sim.set_incident_angle(inc_ang=THETA, azi_ang=PHI)
    for h, eps in _layers():
        sim.add_layer(thickness=h, eps=torch.as_tensor(eps, device=device))
    sim.solve_global_smatrix()
    return sim


@pytest.mark.parametrize("pol", ["p", "s"])
def test_efficiencies_match_grcwa(pol, device):
    obj = _grcwa(pol)
    R, T = (np.real(a) for a in obj.RT_Solve(normalize=1, byorder=1))
    sim = _solwa(device)
    orders = obj.G.tolist()
    for port, ref in (("r", R), ("t", T)):
        eff = sum(
            sim.S_parameters(orders, port=port, polarization=out + pol).abs() ** 2
            for out in "ps"
        )
        np.testing.assert_allclose(eff.cpu().numpy(), ref, rtol=0, atol=1e-12)


def test_fields_match_grcwa(device):
    z = 0.1  # inside the first (patterned) layer
    E_ref, H_ref = _grcwa("p").Solve_FieldOnGrid(1, z)
    sim = _solwa(device)
    sim.source_planewave(amplitude=[1.0, 0.0], notation="ps")
    x = torch.arange(NX, dtype=torch.float64, device=device) / NX
    E, H = sim.field_xy(0, x, x, z)
    ours = np.stack([f.cpu().numpy() for f in (*E, *H)])
    ref = np.stack([np.asarray(f).reshape(NX, NX) for f in (*E_ref, *H_ref)])
    # grcwa omits the Bloch factor exp(i k_par . r) and uses another phase origin
    k = 2 * math.pi * FREQ * math.sin(THETA)
    X, Y = np.meshgrid(x.cpu().numpy(), x.cpu().numpy(), indexing="ij")
    ref = ref * np.exp(1j * k * (math.cos(PHI) * X + math.sin(PHI) * Y))
    i = np.unravel_index(np.abs(ref).argmax(), ref.shape)
    ref = ref * (ours[i] / ref[i])
    assert np.abs(ours - ref).max() < 1e-10 * np.abs(ref).max()
