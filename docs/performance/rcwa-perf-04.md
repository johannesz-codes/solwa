# RCWA optimization 4/7: Reduce layer matching to two block inverses

For C=[[A,B],[B,A]], the orthogonal plus/minus basis yields diag(A+B,A-B) without assuming commutativity. The inverse factor 1/2 cancels the right-side factor 2. Form Cf=[U;V] and Cb=[V;U] from the two inverses.

## Stack order

Merge in numerical order, 01 through 07. This branch is `codex/rcwa-perf-04-block` and
its original base is `codex/rcwa-perf-03-solve` at `69f4032eebd2231c2f2debf81ed2b24eb881dce1`.
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
  internal layer for forward/backward incidence; max absolute error 1.799e-14.
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
| [3,3] | 1 | 26.053 | 24.402 | 1.068x |
| [3,3] | 3 | 73.146 | 68.255 | 1.072x |
| [5,5] | 1 | 288.966 | 264.427 | 1.093x |
| [5,5] | 3 | 815.036 | 743.236 | 1.097x |


[Raw timings, source hashes and checks](../../benchmarks/results/rcwa-perf-04-cpu.json)

The measurement precedes this commit: `checkout_head` names the parent, while
`checkout_numerical_diff=true` and `checkout_sha256` identify the tested source.

```bash
python benchmarks/benchmark_rcwa_revisions.py \
  --baseline 8d2776eb0c365e4dc46011899b1b6963545b4500 --threads 1 --orders 3 5 --layers 1 3 \
  --repeats 7 --output /tmp/rcwa-perf-04.json
```

## RAM impact

| CPU workload | Parent peak MiB (range) | Current peak MiB (range) | Delta MiB | Interpretation |
|---|---:|---:|---:|---|
| Forward only | 779.77 (779.73–780.02) | 771.53 (771.25–776.02) | -8.23 | lower peak in these samples |
| Forward + backward | 940.04 (937.82–940.50) | 930.28 (929.55–935.02) | -9.76 | lower peak in these samples |


Median and range of **three fresh Linux child processes per revision and mode**,
order [5,5], three layers, CPU complex128, one thread, same geometry/materials
as the timing experiment. Peak is process `ru_maxrss`, including Python,
PyTorch/MKL, native workspaces and allocator caches; the import high-water mark
alone is roughly 658 MiB on this host. It is not live-tensor memory or GPU VRAM.
Forward+backward differentiates zero-order Txx with respect to the shared
permittivity contrast; all other geometry and solver settings are identical.
No timing benchmark ran concurrently with these memory measurements.
Three samples show dispersion, not a statistical confidence interval.

**Tensor-level explanation:** The explicit 2m by 2m matching matrix and its inverse (3.574 MiB each here) are replaced by two m by m inverses (0.894 MiB each) plus m by m sums/differences. Cf/Cb and output dimensions are unchanged. This lowers principal matching temporaries, not every solver allocation.

**GPU:** no CUDA/VRAM measurement is available. Tensor-shape reasoning suggests
which allocations change, but the CPU RSS deltas must not be advertised as
GPU-memory savings. Guarded mode, actual device offloading and other gradient
objectives require their own memory measurements.

[Forward raw data](../../benchmarks/results/rcwa-perf-04-ram-forward.json) · [Forward+backward raw data](../../benchmarks/results/rcwa-perf-04-ram-backward.json)

```bash
python benchmarks/benchmark_rcwa_memory.py \
  --baseline 8d2776eb0c365e4dc46011899b1b6963545b4500 --current 5461143b3adf26cccf797ca0e862467339e6da11 \
  --orders 5 --layers 3 --repeats 3 --mode forward --output /tmp/ram-forward.json
python benchmarks/benchmark_rcwa_memory.py \
  --baseline 8d2776eb0c365e4dc46011899b1b6963545b4500 --current 5461143b3adf26cccf797ca0e862467339e6da11 \
  --orders 5 --layers 3 --repeats 3 --mode backward --output /tmp/ram-backward.json
```

The RAM evidence commits preserve the measured source revisions in ancestry;
only benchmark tooling, raw data and documentation were added afterward.
