# RCWA optimization 7/7: Factor the internal layer scattering blocks

Factor E out of the internal scattering expressions, reducing eight dense products to two. Equal pairs use clone() to retain independently writable storage and autograd. This equality is local to the internal layer reference interfaces, not an arbitrary global stack.

## Stack order

Merge in numerical order, 01 through 07. This branch is `codex/rcwa-perf-07-factor` and
its original base is `codex/rcwa-perf-06-redheffer` at `dc1c1266e3baa2ba5f0bff4b6c0f1fa54bdcccfd`.
The repository allows squash merges only. After the predecessor is squash-merged,
rebase this branch onto the updated main, dropping the original base shown above,
and retarget its PR to main. Do not merge a successor into a predecessor branch.
The PR description contains the linked stack and exact commands.

## Validation

- `22 passed` with CPU float64/complex128 regression cases and the existing
  complex64 tests; default/guarded/Q-fallback matching, ordinary eigensolver,
  lossy layers, CPU offloading, both field directions, analytical slab energy
  and Fabry-Perot, and two-layer gradient finite differences.
- Independently loaded predecessor compared with this source at orders [3,3]
  and [5,5], one/three layers: all global complex S blocks and fields in every
  internal layer for forward/backward incidence; max absolute error 4.914e-15.
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
| [3,3] | 1 | 22.958 | 21.780 | 1.054x |
| [3,3] | 3 | 60.814 | 57.303 | 1.061x |
| [5,5] | 1 | 243.655 | 226.770 | 1.074x |
| [5,5] | 3 | 651.783 | 598.548 | 1.089x |


[Raw timings, source hashes and checks](../../benchmarks/results/rcwa-perf-07-cpu.json)

The measurement precedes this commit: `checkout_head` names the parent, while
`checkout_numerical_diff=true` and `checkout_sha256` identify the tested source.

```bash
python benchmarks/benchmark_rcwa_revisions.py \
  --baseline 0ff4b81e33837dbbade5dccddaaf3d612756e11e --threads 1 --orders 3 5 --layers 1 3 \
  --repeats 7 --output /tmp/rcwa-perf-07.json
```

## RAM impact

| CPU workload | Parent peak MiB (range) | Current peak MiB (range) | Delta MiB | Interpretation |
|---|---:|---:|---:|---|
| Forward only | 785.33 (784.45–788.01) | 781.83 (780.72–782.08) | -3.50 | lower peak in these samples |
| Forward + backward | 899.57 (899.32–900.38) | 895.96 (895.57–901.29) | -3.61 | ranges overlap; no resolved improvement/regression |


Median and range of **three fresh Linux child processes per revision and mode**,
order [5,5], three layers, CPU complex128, one thread, same geometry/materials
as the timing experiment. Peak is process `ru_maxrss`, including Python,
PyTorch/MKL, native workspaces and allocator caches; the import high-water mark
alone is roughly 658 MiB on this host. It is not live-tensor memory or GPU VRAM.
Forward+backward differentiates zero-order Txx with respect to the shared
permittivity contrast; all other geometry and solver settings are identical.
No timing benchmark ran concurrently with these memory measurements.
Three samples show dispersion, not a statistical confidence interval.

**Tensor-level explanation:** The four independently writable S outputs are retained using clones, so persistent S storage is not halved. Factoring E reduces product intermediates, while row-scaled coefficient sums also allocate m by m buffers. Any peak improvement is workload/allocator dependent.

**GPU:** no CUDA/VRAM measurement is available. Tensor-shape reasoning suggests
which allocations change, but the CPU RSS deltas must not be advertised as
GPU-memory savings. Guarded mode, actual device offloading and other gradient
objectives require their own memory measurements.

[Forward raw data](../../benchmarks/results/rcwa-perf-07-ram-forward.json) · [Forward+backward raw data](../../benchmarks/results/rcwa-perf-07-ram-backward.json)

```bash
python benchmarks/benchmark_rcwa_memory.py \
  --baseline 0ff4b81e33837dbbade5dccddaaf3d612756e11e --current ebc8c5a31573789f5e43f4d958c65a0a1ae5eca3 \
  --orders 5 --layers 3 --repeats 3 --mode forward --output /tmp/ram-forward.json
python benchmarks/benchmark_rcwa_memory.py \
  --baseline 0ff4b81e33837dbbade5dccddaaf3d612756e11e --current ebc8c5a31573789f5e43f4d958c65a0a1ae5eca3 \
  --orders 5 --layers 3 --repeats 3 --mode backward --output /tmp/ram-backward.json
```

The RAM evidence commits preserve the measured source revisions in ancestry;
only benchmark tooling, raw data and documentation were added afterward.
