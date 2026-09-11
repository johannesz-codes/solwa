# RCWA optimization 3/7: Solve layer field transformations directly

Use solve(P, E*kz) in the unguarded path and solve(Vf, H). The explicit P inverse and existing diagnostics remain in the guarded path. Coefficient inverse slicing is retained; no transient solve(C, 2I) is added.

## Stack order

Merge in numerical order, 01 through 07. This branch is `codex/rcwa-perf-03-solve` and
its original base is `codex/rcwa-perf-02-diagonal` at `f005e2726731d26f225780082687f9bea97d6176`.
The repository allows squash merges only. After the predecessor is squash-merged,
rebase this branch onto the updated main, dropping the original base shown above,
and retarget its PR to main. Do not merge a successor into a predecessor branch.
The PR description contains the linked stack and exact commands.

## Validation

- `20 passed` with CPU float64/complex128 regression cases and the existing
  complex64 tests; default/guarded/Q-fallback matching, ordinary eigensolver,
  lossy layers, CPU offloading, both field directions, analytical slab energy
  and Fabry-Perot, and two-layer gradient finite differences.
- Independently loaded predecessor compared with this source at orders [3,3]
  and [5,5], one/three layers: all global complex S blocks and fields in every
  internal layer for forward/backward incidence; max absolute error 8.491e-14.
- Additional contrast/shared-thickness autograd comparison against predecessor
  and central finite differences before timing, at order [1,1] with two layers.
- Black check passed. Ruff E4/E7/E9/F checks passed. Full Ruff 0.16.6 reports
  18 existing findings, identical to the original main baseline;
  no new full-Ruff findings introduced. Unrelated baseline lint is unchanged.
- CUDA/GPU memory and GPU consistency have not been tested on this CPU host.
  CPU offloading coverage does not verify actual CPU/GPU transfers.

## Timing

Intel Core Ultra 7 258V, PyTorch 2.11.0+xpu/MKL, CPU, one intra-op and one
interop thread, complex128, batch 1, no gradients timed, no offloading.
64x64 pixels, 300x340 nm periods, 532 nm wavelength, epsilon 1/4 rectangular
patterns, oblique incidence (0.13/0.27 rad), one or three layers.
Each result is the median of seven interleaved fresh simulations after warmup.
No other benchmark was run concurrently. These are per-step CPU measurements,
not GPU predictions; small differences need caution.

| Order | Layers | Parent median (ms) | Current median (ms) | Parent/current |
|---|---:|---:|---:|---:|
| [3,3] | 1 | 26.625 | 26.097 | 1.020x |
| [3,3] | 3 | 75.201 | 73.854 | 1.018x |
| [5,5] | 1 | 294.934 | 289.503 | 1.019x |
| [5,5] | 3 | 849.137 | 834.836 | 1.017x |


[Raw timings, source hashes and checks](../../benchmarks/results/rcwa-perf-03-cpu.json)

The measurement precedes this commit: `checkout_head` names the parent, while
`checkout_numerical_diff=true` and `checkout_sha256` identify the tested source.

```bash
python benchmarks/benchmark_rcwa_revisions.py \
  --baseline aa3f27476ac94d93a318b6460133de3531b4a722 --threads 1 --orders 3 5 --layers 1 3 \
  --repeats 7 --output /tmp/rcwa-perf-03.json
```

## RAM impact

| CPU workload | Parent peak MiB (range) | Current peak MiB (range) | Delta MiB | Interpretation |
|---|---:|---:|---:|---|
| Forward only | 780.65 (779.71–780.88) | 780.80 (779.67–784.61) | +0.15 | ranges overlap; no resolved improvement/regression |
| Forward + backward | 938.20 (936.39–941.18) | 938.21 (937.84–938.37) | +0.01 | ranges overlap; no resolved improvement/regression |


Median and range of **three fresh Linux child processes per revision and mode**,
order [5,5], three layers, CPU complex128, one thread, same geometry/materials
as the timing experiment. Peak is process `ru_maxrss`, including Python,
PyTorch/MKL, native workspaces and allocator caches; the import high-water mark
alone is roughly 658 MiB on this host. It is not live-tensor memory or GPU VRAM.
Forward+backward differentiates zero-order Txx with respect to the shared
permittivity contrast; all other geometry and solver settings are identical.
No timing benchmark ran concurrently with these memory measurements.
Three samples show dispersion, not a statistical confidence interval.

**Tensor-level explanation:** The default path no longer materializes P^-1 and Vf^-1 solely for multiplication. Each is 16*m^2 bytes (0.894 MiB here), but factorization workspaces and retained autograd solve data still allocate memory. The guarded P inverse remains, so no RAM saving is promised for that path.

**GPU:** no CUDA/VRAM measurement is available. Tensor-shape reasoning suggests
which allocations change, but the CPU RSS deltas must not be advertised as
GPU-memory savings. Guarded mode, actual device offloading and other gradient
objectives require their own memory measurements.

[Forward raw data](../../benchmarks/results/rcwa-perf-03-ram-forward.json) · [Forward+backward raw data](../../benchmarks/results/rcwa-perf-03-ram-backward.json)

```bash
python benchmarks/benchmark_rcwa_memory.py \
  --baseline aa3f27476ac94d93a318b6460133de3531b4a722 --current 8d2776eb0c365e4dc46011899b1b6963545b4500 \
  --orders 5 --layers 3 --repeats 3 --mode forward --output /tmp/ram-forward.json
python benchmarks/benchmark_rcwa_memory.py \
  --baseline aa3f27476ac94d93a318b6460133de3531b4a722 --current 8d2776eb0c365e4dc46011899b1b6963545b4500 \
  --orders 5 --layers 3 --repeats 3 --mode backward --output /tmp/ram-backward.json
```

The RAM evidence commits preserve the measured source revisions in ancestry;
only benchmark tooling, raw data and documentation were added afterward.
