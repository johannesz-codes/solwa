"""Validate and time checkout RCWA against a Git revision using PyTorch only.

Run from any directory with Python benchmarks/benchmark_rcwa_revisions.py.
Timings include construction, Fourier transforms, eigensolves and scattering.
The baseline revision must be trusted: its numerical Python modules are executed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import types

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = "ab725611e7a510042c831474e38a0be63e780aa2"


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def load_solver(name, revision=None):
    package = types.ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
    hashes = {}
    for stem in ("torch_eig", "rcwa"):
        relative = f"src/solwa/{stem}.py"
        source = (
            git("show", f"{revision}:{relative}")
            if revision
            else (ROOT / relative).read_bytes()
        )
        hashes[relative] = hashlib.sha256(source).hexdigest()
        module = types.ModuleType(f"{name}.{stem}")
        module.__package__ = name
        module.__file__ = f"{revision or 'checkout'}:{relative}"
        sys.modules[module.__name__] = module
        exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module.rcwa, hashes


def simulate(cls, order, layers, device, contrast=3.0, thickness=None):
    sim = cls(
        freq=1 / 532.0,
        order=[order, order],
        L=[300.0, 340.0],
        dtype=torch.complex128,
        device=device,
    )
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=1.0)
    sim.set_incident_angle(inc_ang=0.13, azi_ang=0.27)
    axis = (torch.arange(64, dtype=torch.float64, device=device) + 0.5) / 64
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    mask = ((x > 0.2) & (x < 0.63) & (y > 0.25) & (y < 0.77)).to(torch.float64)
    for layer in range(layers):
        pattern = torch.roll(mask, shifts=(3 * layer, 5 * layer), dims=(0, 1))
        sim.add_layer(
            thickness=120.0 + 35.0 * layer if thickness is None else thickness,
            eps=1.0 + contrast * pattern,
        )
    sim.solve_global_smatrix()
    return sim


def physical_values(sim):
    values = [block.reshape(-1) for block in sim.S]
    axis = torch.linspace(0, 290, 4, dtype=torch.float64, device=sim._device)
    for direction in ("forward", "backward"):
        sim.source_planewave(amplitude=[1.0, 0.3j], direction=direction)
        for layer in range(sim.layer_N):
            fields = sim.field_xy(layer, axis, axis, z_prop=40.0)
            values.extend(
                component.reshape(-1) for group in fields for component in group
            )
    return torch.cat(values)


def objective(sim):
    return (
        sim.S_parameters(
            orders=[[0, 0]], direction="forward", port="transmission", polarization="xx"
        )
        .abs()
        .square()
        .sum()
    )


def validate(solvers, orders, layer_counts, device):
    forward = []
    with torch.no_grad():
        for order in orders:
            for layers in layer_counts:
                expected = physical_values(
                    simulate(solvers["baseline"], order, layers, device)
                )
                actual = physical_values(
                    simulate(solvers["current"], order, layers, device)
                )
                torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)
                forward.append(
                    dict(
                        order=order,
                        layers=layers,
                        max_abs_error=(actual - expected).abs().max().item(),
                    )
                )
    gradients = {}
    reference = None
    for name, cls in solvers.items():
        contrast = torch.tensor(
            3.0, dtype=torch.float64, device=device, requires_grad=True
        )
        thickness = torch.tensor(
            120.0, dtype=torch.float64, device=device, requires_grad=True
        )
        loss = objective(simulate(cls, 1, 2, device, contrast, thickness))
        gradient = torch.stack(torch.autograd.grad(loss, (contrast, thickness)))
        if reference is None:
            reference = gradient
        torch.testing.assert_close(gradient, reference, atol=1e-8, rtol=1e-6)
        finite = []
        with torch.no_grad():
            for index, (value, step) in enumerate(((3.0, 1e-4), (120.0, 1e-3))):
                values = []
                for sign in (1, -1):
                    parameters = [3.0, 120.0]
                    parameters[index] = value + sign * step
                    values.append(
                        objective(simulate(cls, 1, 2, device, *parameters)).item()
                    )
                finite.append((values[0] - values[1]) / (2 * step))
        fd = torch.tensor(finite, dtype=torch.float64, device=device)
        torch.testing.assert_close(gradient, fd, atol=1e-8, rtol=2e-5)
        gradients[name] = dict(
            order=1,
            layers=2,
            parameters=["contrast", "shared_thickness"],
            autograd=gradient.tolist(),
            finite_difference=finite,
        )
    return dict(scattering_and_fields=forward, gradients=gradients)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(solvers, orders, layer_counts, repeats, device):
    results = []
    with torch.no_grad():
        for order in orders:
            for layers in layer_counts:
                samples = {name: [] for name in solvers}
                for cls in solvers.values():
                    warmup = simulate(cls, order, layers, device)
                    synchronize(device)
                    del warmup
                names = list(solvers)
                for repeat in range(repeats):
                    offset = repeat % len(names)
                    for name in names[offset:] + names[:offset]:
                        synchronize(device)
                        start = time.perf_counter()
                        simulation = simulate(solvers[name], order, layers, device)
                        synchronize(device)
                        samples[name].append((time.perf_counter() - start) * 1000)
                        del simulation
                measurements = {
                    name: dict(
                        raw_ms=times,
                        median_ms=statistics.median(times),
                        min_ms=min(times),
                        max_ms=max(times),
                        stdev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
                    )
                    for name, times in samples.items()
                }
                speedup = (
                    measurements["baseline"]["median_ms"]
                    / measurements["current"]["median_ms"]
                )
                results.append(
                    dict(
                        order=order,
                        layers=layers,
                        measurements=measurements,
                        baseline_over_current=speedup,
                    )
                )
                print(
                    f"order={order} layers={layers}: baseline/current={speedup:.3f}x",
                    flush=True,
                )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--orders", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("rcwa-revisions.json"))
    args = parser.parse_args()
    if min(args.threads, args.repeats, *args.layers) < 1 or min(args.orders) < 0:
        parser.error(
            "threads, repeats and layers must be positive; orders must be nonnegative"
        )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    revision = (
        git("rev-parse", "--verify", "--end-of-options", args.baseline + "^{commit}")
        .decode()
        .strip()
    )
    baseline, baseline_hashes = load_solver("_solwa_revision_baseline", revision)
    current, current_hashes = load_solver("_solwa_revision_current")
    solvers = dict(baseline=baseline, current=current)
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    metadata = dict(
        baseline_commit=revision,
        checkout_head=git("rev-parse", "HEAD").decode().strip(),
        checkout_numerical_diff=bool(
            git(
                "diff", "HEAD", "--", "src/solwa/rcwa.py", "src/solwa/torch_eig.py"
            ).strip()
        ),
        baseline_sha256=baseline_hashes,
        checkout_sha256=current_hashes,
        python=sys.version,
        torch=torch.__version__,
        platform=platform.platform(),
        cpu=cpu,
        device=str(device),
        dtype="complex128",
        batch_size=1,
        gradients_timed=False,
        stable_eig_grad=True,
        avoid_Pinv_instability=False,
        offload_device=None,
        threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        cuda_version=torch.version.cuda,
        orders=args.orders,
        layers=args.layers,
        repeats=args.repeats,
        geometry=dict(
            grid=[64, 64],
            period_nm=[300.0, 340.0],
            wavelength_nm=532.0,
            contrast=3.0,
            angles_rad=[0.13, 0.27],
            thickness_nm="120 + 35 * layer",
            pixel_shift="(3 * layer, 5 * layer)",
        ),
    )
    if device.type == "cuda":
        metadata["gpu"] = torch.cuda.get_device_name(device)
    print(
        "Validating scattering, internal fields and gradients before timing...",
        flush=True,
    )
    validation = validate(solvers, args.orders, args.layers, device)
    results = benchmark(solvers, args.orders, args.layers, args.repeats, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(metadata=metadata, validation=validation, results=results), indent=2
        )
        + "\n"
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
