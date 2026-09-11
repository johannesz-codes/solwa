# RCWA optimization 6/7: Reuse Redheffer connection products

Compute the four connection products once per Redheffer composition and reuse them for scattering and every layer coupling coefficient. Preserve multiplication order and existing coefficient-list iteration.

## Stack order

Merge in numerical order, 01 through 07. This branch is `codex/rcwa-perf-06-redheffer` and
its original base is `codex/rcwa-perf-05-pq` at `0a85b88930dca4c51739a3d35b7f8c46ffa2e149`.
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
  internal layer for forward/backward incidence; max absolute error 0.000e+00.
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
| [3,3] | 1 | 24.055 | 22.797 | 1.055x |
| [3,3] | 3 | 67.117 | 60.834 | 1.103x |
| [5,5] | 1 | 262.487 | 248.680 | 1.056x |
| [5,5] | 3 | 751.990 | 654.417 | 1.149x |


[Raw timings, source hashes and checks](../../benchmarks/results/rcwa-perf-06-cpu.json)

The measurement precedes this commit: `checkout_head` names the parent, while
`checkout_numerical_diff=true` and `checkout_sha256` identify the tested source.

```bash
python benchmarks/benchmark_rcwa_revisions.py \
  --baseline 4129f564168ab2d703a795a484e0df97fa293257 --threads 1 --orders 3 5 --layers 1 3 \
  --repeats 7 --output /tmp/rcwa-perf-06.json
```

## RAM impact

| CPU workload | Parent peak MiB (range) | Current peak MiB (range) | Delta MiB | Interpretation |
|---|---:|---:|---:|---|
| Forward only | 781.36 (781.24–783.42) | 785.68 (784.41–786.06) | +4.32 | higher peak in these samples |
| Forward + backward | 929.50 (929.46–931.49) | 899.80 (897.41–900.89) | -29.70 | lower peak in these samples |


Median and range of **three fresh Linux child processes per revision and mode**,
order [5,5], three layers, CPU complex128, one thread, same geometry/materials
as the timing experiment. Peak is process `ru_maxrss`, including Python,
PyTorch/MKL, native workspaces and allocator caches; the import high-water mark
alone is roughly 658 MiB on this host. It is not live-tensor memory or GPU VRAM.
Forward+backward differentiates zero-order Txx with respect to the shared
permittivity contrast; all other geometry and solver settings are identical.
No timing benchmark ran concurrently with these memory measurements.
Three samples show dispersion, not a statistical confidence interval.

**Tensor-level explanation:** Four m by m connection products are retained together (up to 3.574 MiB here) and reused across all coefficient updates. This trades extra live temporary storage for less recomputation; persistent C/S output sizes are unchanged. Autograd graph reuse can change the tradeoff in backward workloads.

**GPU:** no CUDA/VRAM measurement is available. Tensor-shape reasoning suggests
which allocations change, but the CPU RSS deltas must not be advertised as
GPU-memory savings. Guarded mode, actual device offloading and other gradient
objectives require their own memory measurements.

[Forward raw data](../../benchmarks/results/rcwa-perf-06-ram-forward.json) · [Forward+backward raw data](../../benchmarks/results/rcwa-perf-06-ram-backward.json)

```bash
python benchmarks/benchmark_rcwa_memory.py \
  --baseline 4129f564168ab2d703a795a484e0df97fa293257 --current 0ff4b81e33837dbbade5dccddaaf3d612756e11e \
  --orders 5 --layers 3 --repeats 3 --mode forward --output /tmp/ram-forward.json
python benchmarks/benchmark_rcwa_memory.py \
  --baseline 4129f564168ab2d703a795a484e0df97fa293257 --current 0ff4b81e33837dbbade5dccddaaf3d612756e11e \
  --orders 5 --layers 3 --repeats 3 --mode backward --output /tmp/ram-backward.json
```

The RAM evidence commits preserve the measured source revisions in ancestry;
only benchmark tooling, raw data and documentation were added afterward.
