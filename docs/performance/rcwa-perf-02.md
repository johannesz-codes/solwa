# RCWA optimization 2/7: Replace layer diagonal products with scaling

Use column scaling for E times kz/phase and select scaled inverse columns for the two identity/zero matching right sides. Keep the guarded Q fallback inverse to preserve its singular-matrix behavior.

## Stack order

Merge in numerical order, 01 through 07. This branch is `codex/rcwa-perf-02-diagonal` and
its original base is `codex/rcwa-perf-01-reuse` at `5a355d12bf7885b8c275e10d8baea07521b49660`.
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
  internal layer for forward/backward incidence; max absolute error 7.978e-15.
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
| [3,3] | 1 | 28.803 | 26.547 | 1.085x |
| [3,3] | 3 | 81.588 | 74.954 | 1.089x |
| [5,5] | 1 | 334.987 | 302.104 | 1.109x |
| [5,5] | 3 | 956.788 | 867.278 | 1.103x |


[Raw timings, source hashes and checks](../../benchmarks/results/rcwa-perf-02-cpu.json)

The measurement precedes this commit: `checkout_head` names the parent, while
`checkout_numerical_diff=true` and `checkout_sha256` identify the tested source.

```bash
python benchmarks/benchmark_rcwa_revisions.py \
  --baseline 7a4132a0415746b65f714509a4426a2676b2d8ac --threads 1 --orders 3 5 --layers 1 3 \
  --repeats 7 --output /tmp/rcwa-perf-02.json
```

## RAM impact

| CPU workload | Parent peak MiB (range) | Current peak MiB (range) | Delta MiB | Interpretation |
|---|---:|---:|---:|---|
| Forward only | 786.95 (785.89–787.91) | 781.03 (780.85–781.14) | -5.91 | lower peak in these samples |
| Forward + backward | 962.07 (960.93–963.87) | 937.59 (936.37–940.70) | -24.48 | lower peak in these samples |


Median and range of **three fresh Linux child processes per revision and mode**,
order [5,5], three layers, CPU complex128, one thread, same geometry/materials
as the timing experiment. Peak is process `ru_maxrss`, including Python,
PyTorch/MKL, native workspaces and allocator caches; the import high-water mark
alone is roughly 658 MiB on this host. It is not live-tensor memory or GPU VRAM.
Forward+backward differentiates zero-order Txx with respect to the shared
permittivity contrast; all other geometry and solver settings are identical.
No timing benchmark ran concurrently with these memory measurements.
Three samples show dispersion, not a statistical confidence interval.

**Tensor-level explanation:** Dense phase/kz matrices and identity/zero right-side products are avoided. An m by m complex128 buffer is 16*m^2 bytes (0.894 MiB for m=242); one 2m by m right side is 1.787 MiB. Persistent output sizes are unchanged. Views/scaling and allocator reuse determine the actual peak.

**GPU:** no CUDA/VRAM measurement is available. Tensor-shape reasoning suggests
which allocations change, but the CPU RSS deltas must not be advertised as
GPU-memory savings. Guarded mode, actual device offloading and other gradient
objectives require their own memory measurements.

[Forward raw data](../../benchmarks/results/rcwa-perf-02-ram-forward.json) · [Forward+backward raw data](../../benchmarks/results/rcwa-perf-02-ram-backward.json)

```bash
python benchmarks/benchmark_rcwa_memory.py \
  --baseline 7a4132a0415746b65f714509a4426a2676b2d8ac --current aa3f27476ac94d93a318b6460133de3531b4a722 \
  --orders 5 --layers 3 --repeats 3 --mode forward --output /tmp/ram-forward.json
python benchmarks/benchmark_rcwa_memory.py \
  --baseline 7a4132a0415746b65f714509a4426a2676b2d8ac --current aa3f27476ac94d93a318b6460133de3531b4a722 \
  --orders 5 --layers 3 --repeats 3 --mode backward --output /tmp/ram-backward.json
```

The RAM evidence commits preserve the measured source revisions in ancestry;
only benchmark tooling, raw data and documentation were added afterward.
