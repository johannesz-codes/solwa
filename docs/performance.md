# Performance guide

solwa's speed, memory and accuracy depend mostly on a few options that are easy
to miss. This page says which one to use when. All numbers below were measured;
the hardware is given with each timing.

## Where the time goes

For a structured layer, solwa builds P and Q, computes the eigendecomposition of
P·Q (2N×2N, N = number of harmonics), inverts several matrices, and combines
layers with Redheffer star products. Profile of a 4-layer stack in complex64
(RTX 5070 Ti):

| step | order 10 (2N = 882) | order 15 (2N = 1922) |
|---|---|---|
| `eig` | 55 % | 39 % |
| inversions (layer S-matrix) | 20 % | 20 % |
| matmuls (star product + layer) | 17 % | 34 % |
| rest | 8 % | 7 % |

Cost grows as O(N³) per layer, and N grows as (2M+1)² with the Fourier order M
in 2D. Doubling M in 2D costs about 64× in time and 16× in memory per matrix.
So the most effective setting is the smallest order that converges, and the
factorization rule decides how small that is.

## Choosing the options

| goal | setting |
|---|---|
| TM polarization or 2D gratings converge faster | `factorization="li"` (default) |
| near-complex128 accuracy at complex64 memory | `auto_precision=True` |
| fastest, accuracy ~1e-4 is enough (low orders) | `dtype=torch.complex64` (default) |
| reference runs, convergence studies | `dtype=torch.complex128` or `auto_precision=True` |
| many layers do not fit in GPU memory (forward runs) | `offload_device="cpu"` |

### Factorization: `li` vs `laurent`

`factorization="li"` (default since v0.3.0rc1, PR #54) applies Li's inverse rule
for the field components normal to material boundaries. Run time per order is
practically the same as `laurent`, but the error at a given order is about 15 to
350 times lower for TM and 2D structures (complex128, error vs. a converged
high-order reference):

| case | order | error Laurent | error Li |
|---|---|---|---|
| 1D metal grating, TM, R₀ | 10 | 2.1e-1 | 6.5e-3 |
| 1D dielectric grating, TM, R₀ | 10 | 1.9e-3 | 1.1e-4 |
| 2D metal blocks, T₀ | 11 | 3.5e-2 | 7.9e-4 |
| 2D dielectric blocks, T₀ | 15 | 7.8e-3 | 2.2e-5 |

For TE in 1D both rules give identical results. Li's rule is optimal for
axis-aligned shapes; circles and rotated shapes benefit less. Use `laurent`
only to compare with older results.

### Precision: `dtype` and `auto_precision`

With complex64 the eigendecomposition limits accuracy. `auto_precision=True`
(PR #59) runs only `eig` in complex128 and everything else in complex64; inputs
and outputs stay in the `dtype` you passed. Order 10, max error of all
diffraction efficiencies vs. full complex128 (RTX 5070 Ti):

| structure | complex64 | `auto_precision` |
|---|---|---|
| TiO2-like pillar | 3.3e-4 | 1.6e-6 |
| 4-layer stack | 9.5e-5 | 2.7e-6 |

The complex64 error grows with the order (pillar: 2e-3 at order 15, 7.6e-2 at
order 30, while `auto_precision` stays near 1e-5), and from order 15 its
gradients are unusable for optimization (61 % error at order 15).

Peak memory equals complex64, i.e. half of complex128. Speed depends on the
GPU's FP64 throughput:

- **Consumer GPUs** (RTX 5070 Ti) have little FP64: `auto_precision` is about
  4× slower than complex64 and 3–5× faster than complex128 (orders 10–20).
- **H200**: complex128 `eig` is fast, so `auto_precision` runs at about the
  speed of complex128 with half the memory. `eig` is over 90 % of the time
  there.

Lower precisions do not pay off. `eig` and `inv` exist only in complex64/128,
and matmuls are at most about a third of the time. TF32 or fp16 matmuls gave
8–18 % speedup for about 10× higher error in the efficiencies and 1e-2 to 5e-2
relative gradient error; bf16 was worse. Don't use TF32
(`torch.backends.cuda.matmul.allow_tf32`) for optimization.

### GPU memory: offloading

`offload_device="cpu"` moves the tensors of each finished layer (P, Q,
convolution matrices, eigenvectors, layer S-matrices) to host memory after it is
added, and the global field coefficients after the global S-matrix is solved.
They come back to the GPU on demand. Pinned host buffers are used automatically
when offloading from CUDA to the CPU (PR #33); there is no option for it.

20 patterned layers, order 10, complex64, forward run (RTX 5070 Ti): peak VRAM
2640 → 1156 MiB (−56 %), time 3.55 → 3.62 s (+1.7 %).

Offloading barely helps with gradients (40 layers, order 7, with backward:
10576 → 10545 MiB), because autograd keeps its saved tensors on the GPU. These
tensors dominate memory for gradients (stack, order 13, complex64, RTX 5070 Ti:
2.0 GB forward, 5.2 GB with backward). Storing them in fp16/bf16 does not help,
since autograd keeps the originals anyway.

## Upcoming

- **Wavelength batching** (open PR #60): `freq` as a 1-D tensor solves all
  wavelengths in one batched S-matrix run. RTX 5070 Ti, complex64, 32
  wavelengths, 3 layers: 1.6× faster at order 5 and 1.4× at order 8, with memory
  growing linearly with the batch size (order 8: 257 → 6847 MiB). Fields
  (`field_*`) are not batched yet (#63).
- **Eigenvector reuse across wavelengths** (experimental, draft PR #61;
  benchmark script only, not in solwa yet): refines the eigendecomposition from
  the neighbouring wavelength with GEMMs only. On H200 the eig part is 3–8.5×
  faster than `torch.linalg.eig` (order 7–20) at 1.4–2.4× its memory, with
  R/T/A errors ≤ 7.7e-13 vs. complex128 (order 7). Gradients through it are
  untested.

## Pitfalls

- **p/s quantities at normal incidence** give NaN gradients with respect to the
  period, because the s/p basis is undefined for the (0,0) order. This affects
  `notation="ps"` sources and `S_parameters(polarization=...)` with `"pp"`,
  `"ps"` or `"sp"`. Use `notation="xy"` and `"xx"`, `"xy"`, `"yx"` or `"yy"` when
  you differentiate at normal incidence.
- **Rayleigh anomalies**: exactly at kz = 0 of a diffraction order in the
  incident or output medium (e.g. λ equal to the period at normal incidence),
  the S-matrix contains NaN and `S_parameters` returns 0 for it without an
  error, also for the propagating orders. Shift the wavelength or angle slightly.
- **Benchmark timings on GPU** need `torch.cuda.synchronize()` before stopping
  the clock, and the first call includes CUDA/cuSOLVER warm-up.
