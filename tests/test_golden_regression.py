"""Golden regression test: complex128 results must not drift from stored references.

The references in ``golden_regression.npz`` were computed with this file's
``compute`` on main. Refactors of the solver must reproduce them to ~1e-10.
Intentional numerical changes regenerate them with

    python tests/test_golden_regression.py [--device cpu|cuda]

and say so in the PR.
"""

import math
import pathlib

import numpy as np
import torch

import solwa

REF = pathlib.Path(__file__).with_name("golden_regression.npz")
RTOL = 1e-10
NX = 48


def _setup(name, device):
    """Two asymmetric reference structures, both with non-degenerate modes."""
    g = torch.Generator().manual_seed(0)
    p = torch.rand(NX, NX, generator=g, dtype=torch.float64).to(device)
    geo = solwa.geometry(
        Lx=1.0, Ly=0.9, nx=NX, ny=NX, edge_sharpness=500.0,
        dtype=torch.float64, device=device,
    )  # fmt: skip
    if name == "oblique_li":
        kw = dict(factorization="li")
        ang = (math.radians(20), math.radians(30))
        p = p.requires_grad_(True)
        layers = [
            (0.3, 1.0 + (5.0 + 0.5j) * geo.rectangle(Wx=0.5, Wy=0.3, Cx=0.4, Cy=0.55)),
            (0.2, 2.0),
            (0.25, 1.0 + (3.0 + 0.1j) * p),
        ]
    else:  # metal ellipse, Laurent rule, avoid_Pinv_instability, near-normal
        kw = dict(factorization="laurent", avoid_Pinv_instability=True)
        ang = (math.radians(5), math.radians(10))
        layers = [
            (0.15, 1.0 + ((0.2 + 3.0j) ** 2 - 1.0) * geo.ellipse(Rx=0.3, Ry=0.2, Cx=0.45, Cy=0.5)),
            (0.1, 2.25 + 0.05j),
        ]  # fmt: skip
    sim = solwa.rcwa(
        freq=1 / 0.75, order=[4, 3], L=[1.0, 0.9],
        dtype=torch.complex128, device=device, **kw,
    )  # fmt: skip
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=2.25)
    sim.set_incident_angle(inc_ang=ang[0], azi_ang=ang[1])
    for h, eps in layers:
        sim.add_layer(thickness=h, eps=eps)
    sim.solve_global_smatrix()
    return sim, p


def compute(device):
    out = {}
    orders = [[0, 0], [1, 0], [-1, 0], [0, 1], [1, -1]]
    for name in ("oblique_li", "metal_laurent"):
        sim, p = _setup(name, device)
        for d in ("f", "b"):
            for port in ("t", "r"):
                for pol in ("xx", "yx", "xy", "yy", "pp", "sp", "ps", "ss"):
                    S = sim.S_parameters(
                        orders, direction=d, port=port, polarization=pol
                    )
                    out[f"{name}/S_{d}{port}_{pol}"] = S

        sim.source_planewave(amplitude=[0.6, 0.8j], direction="forward")
        x = torch.linspace(0.0, 1.0, 7, dtype=torch.float64, device=device)
        y = torch.linspace(0.0, 0.9, 6, dtype=torch.float64, device=device)
        z = torch.linspace(
            -0.1, sum(sim.thickness) + 0.1, 9, dtype=torch.float64, device=device
        )
        fields = {
            "xz": sim.field_xz(x, z, 0.3),
            "yz": sim.field_yz(y, z, 0.6),
            "xy0": sim.field_xy(0, x, y, 0.1),
            "xyin": sim.field_xy(-1, x, y, -0.05),
        }  # fmt: skip
        for key, (E, H) in fields.items():
            for c, f in zip(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"), (*E, *H)):
                out[f"{name}/{key}_{c}"] = f

        if p.requires_grad:
            T = sum(
                (sim.S_parameters(orders, port="t", polarization=pol).abs() ** 2).sum()
                for pol in ("pp", "sp")
            )
            out[f"{name}/T"] = T
            (out[f"{name}/dT_dp"],) = torch.autograd.grad(T, p)

    return {k: v.detach().cpu().numpy() for k, v in out.items()}


def test_golden(device):
    ref = np.load(REF)
    res = compute(device)
    assert set(res) == set(ref.files)
    errs = {k: np.abs(res[k] - ref[k]).max() / np.abs(ref[k]).max() for k in ref.files}
    bad = {k: f"{e:.1e}" for k, e in errs.items() if not e <= RTOL}
    assert not bad, f"relative deviation > {RTOL}: {bad}"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    np.savez(REF, **compute(torch.device(ap.parse_args().device)))
