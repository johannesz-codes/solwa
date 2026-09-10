# RCWA optimization 4/7: Reduce layer matching to two block inverses

For C=[[A,B],[B,A]], the orthogonal plus/minus basis yields diag(A+B,A-B) without assuming commutativity. The inverse factor 1/2 cancels the right-side factor 2. Form Cf=[U;V] and Cb=[V;U] from the two inverses.

## Stack order

Merge in numerical order, 01 through 07. This branch is `codex/rcwa-perf-04-block` and
its original base is `codex/rcwa-perf-03-solve` at `8d2776eb0c365e4dc46011899b1b6963545b4500`.
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
