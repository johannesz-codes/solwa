#!/usr/bin/env python3
"""Minimal reproducer for cell-centered-grid trapz area loss in poynting_flux."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from solwa.geometry import geometry  # noqa: E402
from solwa.utils import poynting_flux  # noqa: E402


class ConstantPoyntingSimulation:
    """Small stand-in exposing the poynting_xy API used by poynting_flux."""

    def poynting_xy(self, layer, x_points, y_points, z_prop):
        del layer, z_prop
        shape = (x_points.numel(), y_points.numel())
        sx = torch.zeros(shape, dtype=x_points.dtype, device=x_points.device)
        sy = torch.zeros(shape, dtype=x_points.dtype, device=x_points.device)
        sz = torch.ones(shape, dtype=x_points.dtype, device=x_points.device)
        return sx, sy, sz


def cell_centered_grid(n: int):
    geo = geometry(
        Lx=1.0,
        Ly=1.0,
        nx=n,
        ny=n,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    geo.grid()
    return geo


def deterministic_reproducer():
    print("Deterministic constant-field reproducer")
    print("Unit cell: Lx = Ly = 1, Sz = 1 on solwa.geometry cell centers")
    print(
        "Correct midpoint/cell-average integral is exactly 1.0; "
        "current poynting_flux uses trapz over [dx/2, 1-dx/2]."
    )
    print()
    print(
        f"{'N':>6}  {'current_trapz_flux':>20}  {'correct_midpoint_flux':>23}  "
        f"{'current / correct':>18}  {'expected_ratio':>16}  {'percent_error':>14}"
    )
    print("-" * 110)

    sim = ConstantPoyntingSimulation()
    rows = []
    for n in (25, 50, 100, 200, 400):
        geo = cell_centered_grid(n)
        current = poynting_flux(sim, layer=0, x_points=geo.x, y_points=geo.y, z_prop=0.0)

        sz = torch.ones((geo.nx, geo.ny), dtype=geo.dtype, device=geo.device)
        dx = geo.Lx / geo.nx
        dy = geo.Ly / geo.ny
        correct = sz.sum() * dx * dy

        ratio = current / correct
        expected_ratio = ((n - 1) / n) ** 2
        percent_error = 100.0 * (ratio.item() - 1.0)
        rows.append((n, ratio.item()))

        print(
            f"{n:6d}  {current.item():20.12f}  {correct.item():23.12f}  "
            f"{ratio.item():18.12f}  {expected_ratio:16.12f}  {percent_error:13.6f}%"
        )

    return rows


def absorption_normalization_demo(n: int = 100):
    alpha = ((n - 1) / n) ** 2
    p_in_true = 1.0
    p_out_true = 0.7
    p_incident = 1.0

    p_in_buggy = alpha * p_in_true
    p_out_buggy = alpha * p_out_true

    a_ratio_true = 1.0 - p_out_true / p_in_true
    a_ratio_buggy = 1.0 - p_out_buggy / p_in_buggy

    a_abs_true = (p_in_true - p_out_true) / p_incident
    a_abs_buggy = (p_in_buggy - p_out_buggy) / p_incident

    print()
    print(f"Absorption-normalization demonstration for N = {n}")
    print(f"Near-field area factor alpha = ((N - 1) / N)^2 = {alpha:.12f}")
    print()
    print("Ratio-style single-layer absorption cancels the common factor:")
    print(f"  A_ratio_true  = 1 - {p_out_true:.6f} / {p_in_true:.6f} = {a_ratio_true:.12f}")
    print(
        f"  A_ratio_buggy = 1 - ({alpha:.6f} * {p_out_true:.6f}) "
        f"/ ({alpha:.6f} * {p_in_true:.6f}) = {a_ratio_buggy:.12f}"
    )
    print(f"  difference    = {a_ratio_buggy - a_ratio_true:+.12e}")
    print()
    print("Absolute layer absorption normalized to independent incident power keeps the error:")
    print(
        f"  A_abs_true    = ({p_in_true:.6f} - {p_out_true:.6f}) "
        f"/ {p_incident:.6f} = {a_abs_true:.12f}"
    )
    print(
        f"  A_abs_buggy   = ({p_in_buggy:.6f} - {p_out_buggy:.6f}) "
        f"/ {p_incident:.6f} = {a_abs_buggy:.12f}"
    )
    print(f"  buggy / true  = {a_abs_buggy / a_abs_true:.12f}")
    print(f"  percent error = {100.0 * (a_abs_buggy / a_abs_true - 1.0):.6f}%")


def main():
    deterministic_reproducer()
    absorption_normalization_demo(n=100)
    print()
    print(
        "Optional real RCWA smoke test: skipped. This diagnostic intentionally stays at "
        "the unit level so it is fast, deterministic, and isolates poynting_flux integration."
    )


if __name__ == "__main__":
    main()
