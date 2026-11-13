"""Benchmark ``rcwa(eig_reuse=True)`` against the regular eigendecomposition.

Full solve of a batch of neighbouring wavelengths for air | a-Si:H 200 nm | pillar grating
114.5 nm | Ag 200 nm | air at normal incidence (Li factorization, complex128).

    python benchmarks/eig_reuse.py --order 12 --batch 8 --grad
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import interp1d

import solwa

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CDT, RDT = torch.complex128, torch.float64
NK_FILE = Path(__file__).parents[1] / "example" / "Materials_data" / "aSiH.txt"
PERIOD, WIDTH, N_DIEL, NX = 522.1, 338.9, 1.5, 512
WINDOWS = {"short": 300.0, "edge": 440.0, "long": 700.0}

nk = np.loadtxt(NK_FILE)
N_I, K_I = (interp1d(nk[:, 0], nk[:, c], kind="cubic") for c in (1, 2))


def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()


def eps_si(lam):
    lam = lam.cpu().numpy()
    return torch.as_tensor((N_I(lam) + 1j * K_I(lam)) ** 2, dtype=CDT, device=DEV)


def eps_ag(lam):  # Drude model
    ev = 1239.84193 / lam
    return 3.7 - 9.0**2 / (ev * (ev + 0.02j)).to(CDT)


def solve(lam, order, width, reuse):
    x = PERIOD * (torch.arange(NX, dtype=RDT, device=DEV) + 0.5) / NX
    edge = torch.sigmoid((width / 2 - (x - PERIOD / 2).abs()) / (PERIOD / NX / 2))
    fill = (edge[:, None] * edge[None, :]).to(CDT)
    si = eps_si(lam)
    sim = solwa.rcwa(
        freq=1.0 / lam,
        order=[order, order],
        L=[PERIOD, PERIOD],
        dtype=CDT,
        device=DEV,
        eig_reuse=reuse,
    )
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=1.0)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=200.0, eps=si)
    sim.add_layer(
        thickness=114.5, eps=N_DIEL**2 + (si[..., None, None] - N_DIEL**2) * fill
    )
    sim.add_layer(thickness=200.0, eps=eps_ag(lam))
    sim.solve_global_smatrix()
    return torch.cat(
        [
            sim.S_parameters(orders=[0, 0], port=p, polarization="xx")
            for p in ("reflection", "transmission")
        ]
    )


def timed(lam, order, reuse, grad):
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    width = torch.tensor(WIDTH, dtype=RDT, device=DEV, requires_grad=grad)
    sync()
    t0 = time.perf_counter()
    S = solve(lam, order, width, reuse)
    if grad:
        (S.abs() ** 2).sum().backward()
    sync()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20 if DEV.type == "cuda" else 0.0
    return dt, S.detach(), peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=7)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad", action="store_true", help="include the backward pass")
    a = ap.parse_args()
    name = torch.cuda.get_device_name() if DEV.type == "cuda" else "cpu"
    print(
        f"{name}, torch {torch.__version__}, order {a.order}, batch {a.batch}, grad {a.grad}"
    )
    warm = 450.0 + torch.arange(a.batch, dtype=RDT, device=DEV)
    for reuse in (False, True):
        timed(warm, a.order, reuse, a.grad)
    for win, lo in WINDOWS.items():
        lam = lo + torch.arange(a.batch, dtype=RDT, device=DEV)
        t_e, S_e, p_e = timed(lam, a.order, False, a.grad)
        t_r, S_r, p_r = timed(lam, a.order, True, a.grad)
        print(
            f"{win:5s}: eig {t_e / a.batch * 1e3:8.1f} ms/lam ({p_e:.0f} MiB) | "
            f"reuse {t_r / a.batch * 1e3:8.1f} ms/lam ({p_r:.0f} MiB) | "
            f"{t_e / t_r:.2f}x | max|dS| {(S_e - S_r).abs().max():.1e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
