"""Measure CPU RCWA peak RSS in fresh Linux processes, separate from timing.

RSS includes the interpreter, PyTorch/MKL, native workspaces and allocator caches.
It is not a count of tensor bytes or a GPU VRAM measurement. Compare revisions
whose numerical equivalence has already passed benchmark_rcwa_revisions.py.
"""

import argparse
import gc
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys

import torch

from benchmark_rcwa_revisions import ROOT, git, load_solver, objective, simulate


def rss_kib():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    raise RuntimeError("Linux VmRSS unavailable")


def worker(revision, order, layers, threads, mode):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    solver, hashes = load_solver("_solwa_memory_worker", revision)
    gc.collect()
    before_rss = rss_kib()
    before_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with torch.set_grad_enabled(mode == "backward"):
        contrast = (
            torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
            if mode == "backward"
            else 3.0
        )
        sim = simulate(solver, order, layers, torch.device("cpu"), contrast=contrast)
        if mode == "backward":
            objective(sim).backward()
    after_rss = rss_kib()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Keep the solved simulation alive through both readings.
    assert all(torch.isfinite(block).all() for block in sim.S)
    if mode == "backward":
        assert torch.isfinite(contrast.grad)
    return dict(
        revision=revision,
        source_sha256=hashes,
        baseline_rss_kib=before_rss,
        baseline_peak_kib=before_peak,
        peak_rss_kib=peak,
        retained_rss_kib=after_rss,
        peak_growth_kib=peak - before_peak,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--orders", type=int, nargs="+", default=[5])
    parser.add_argument("--layers", type=int, nargs="+", default=[3])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--mode", choices=["forward", "backward"], default="forward")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("This benchmark uses Linux ru_maxrss units and /proc RSS")
    if min(args.threads, args.repeats, *args.layers) < 1 or min(args.orders) < 0:
        parser.error("Positive threads/repeats/layers and nonnegative orders required")
    if args.worker:
        print(
            json.dumps(
                worker(
                    args.current,
                    args.orders[0],
                    args.layers[0],
                    args.threads,
                    args.mode,
                )
            )
        )
        return
    if args.output is None:
        parser.error("--output is required")
    revisions = {
        name: git("rev-parse", "--verify", "--end-of-options", value + "^{commit}")
        .decode()
        .strip()
        for name, value in [("baseline", args.baseline), ("current", args.current)]
    }
    results = []
    for order in args.orders:
        for layers in args.layers:
            samples = {name: [] for name in revisions}
            for repeat in range(args.repeats):
                names = list(revisions)
                if repeat % 2:
                    names.reverse()
                for name in names:
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--baseline",
                        revisions["baseline"],
                        "--current",
                        revisions[name],
                        "--orders",
                        str(order),
                        "--layers",
                        str(layers),
                        "--threads",
                        str(args.threads),
                        "--mode",
                        args.mode,
                    ]
                    samples[name].append(
                        json.loads(
                            subprocess.check_output(command, cwd=ROOT, text=True)
                        )
                    )
            measurements = {}
            for name, values in samples.items():
                peaks = [row["peak_rss_kib"] / 1024 for row in values]
                measurements[name] = dict(
                    peak_median_mib=statistics.median(peaks),
                    peak_min_mib=min(peaks),
                    peak_max_mib=max(peaks),
                    peak_growth_median_mib=statistics.median(
                        row["peak_growth_kib"] / 1024 for row in values
                    ),
                    retained_median_mib=statistics.median(
                        row["retained_rss_kib"] / 1024 for row in values
                    ),
                    samples=values,
                )
            delta = (
                measurements["current"]["peak_median_mib"]
                - measurements["baseline"]["peak_median_mib"]
            )
            results.append(
                dict(
                    order=order,
                    layers=layers,
                    measurements=measurements,
                    peak_delta_mib=delta,
                    peak_delta_percent=100
                    * delta
                    / measurements["baseline"]["peak_median_mib"],
                )
            )
            print(
                f"order={order} layers={layers}: peak RSS change {delta:+.2f} MiB",
                flush=True,
            )
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        platform.processor(),
    )
    metadata = dict(
        cpu=cpu,
        revisions=revisions,
        torch=torch.__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        threads=args.threads,
        interop_threads=1,
        dtype="complex128",
        repeats=args.repeats,
        device="cpu",
        batch_size=1,
        grad_enabled=args.mode == "backward",
        mode=args.mode,
        gradient_objective=(
            "Txx(0,0), derivative with respect to shared permittivity contrast"
            if args.mode == "backward"
            else None
        ),
        measurement="Linux process ru_maxrss; 1 KiB=1024 bytes",
        geometry="64x64 pixels; 300x340 nm; wavelength 532 nm; rectangular eps=1/4; angles .13/.27 rad; thickness 120+35i nm; shifts 3i/5i",
        caveat="Fresh process per sample. Includes Python, PyTorch, MKL, native workspaces and allocator caches. Peak growth is above the import high-water mark, not tensor-live memory. No GPU VRAM measurement.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dict(metadata=metadata, results=results), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
