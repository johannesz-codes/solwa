"""Small CUDA benchmark for Solwa's mirror-parity reduction.

The script intentionally requires PyTorch 2.13 and CUDA so benchmark results
cannot silently come from a CPU or an older PyTorch installation.
"""

import argparse
import csv
import gc
import statistics
from pathlib import Path

import torch

import solwa


def make_material(samples, device):
    eps = torch.ones((samples, samples), dtype=torch.float32, device=device)
    mirror_start = samples // 3
    eps[
        samples // 4 : 3 * samples // 4,
        mirror_start : samples - mirror_start,
    ] = 2.5
    return eps


def run_once(order, eps, symmetry_axis):
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.inference_mode():
        sim = solwa.rcwa(
            freq=1 / 520,
            order=[order, order],
            L=[420, 420],
            dtype=torch.complex64,
            device=eps.device,
            symmetry_axis=symmetry_axis,
        )
        sim.set_incident_angle(0.0, 0.0)
        sim.add_layer(110.0, eps)
        sim.add_layer(55.0, eps=1.7)
        sim.solve_global_smatrix()
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    del sim
    gc.collect()
    torch.cuda.empty_cache()
    return elapsed_ms, peak_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders", type=int, nargs="+", default=[7, 11, 15])
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.__version__.startswith("2.13."):
        raise RuntimeError(f"benchmark requires PyTorch 2.13, found {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires a CUDA-capable PyTorch build")

    device = torch.device("cuda")
    eps = make_material(args.samples, device)
    rows = []
    for order in args.orders:
        for symmetry_axis in (None, "x"):
            for _ in range(args.warmup):
                run_once(order, eps, symmetry_axis)
            samples = [run_once(order, eps, symmetry_axis) for _ in range(args.repeats)]
            rows.append(
                {
                    "order": order,
                    "harmonics": (2 * order + 1) ** 2,
                    "mode": "full" if symmetry_axis is None else "symmetry-x",
                    "median_ms": statistics.median(value[0] for value in samples),
                    "min_ms": min(value[0] for value in samples),
                    "peak_allocated_mb": max(value[1] for value in samples),
                }
            )

    print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}; {torch.cuda.get_device_name(0)}")
    print("order  harmonics  mode          median_ms  min_ms  peak_MB  speedup")
    for order in args.orders:
        full = next(row for row in rows if row["order"] == order and row["mode"] == "full")
        reduced = next(
            row for row in rows if row["order"] == order and row["mode"] == "symmetry-x"
        )
        for row in (full, reduced):
            speedup = full["median_ms"] / row["median_ms"]
            print(
                f"{order:5d}  {row['harmonics']:9d}  {row['mode']:12s}  "
                f"{row['median_ms']:9.2f}  {row['min_ms']:6.2f}  "
                f"{row['peak_allocated_mb']:7.1f}  {speedup:7.2f}x"
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
