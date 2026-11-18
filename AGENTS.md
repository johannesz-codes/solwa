
# AGENTS.md

SOLWA is a Python library for differentiable Rigorous Coupled-Wave
Analysis (RCWA), built on PyTorch and optimized for GPU acceleration.

Numerical correctness, gradient accuracy, GPU performance, and memory
efficiency are the primary engineering priorities.

## Code Review Rules

Review changes for actionable bugs and regressions (P0–P2).
Prioritize incorrect numerical results, incorrect gradients, crashes,
silent data corruption, significant performance or VRAM regressions,
and CI resource leaks.

For each finding, identify a concrete failure scenario and explain
how the changed code causes it. Focus on issues introduced or
materially worsened by the PR, not unrelated pre-existing defects.

Avoid style nits, speculative edge cases, unnecessary defensive
guards, and redundant tests. Do not request additional validation
without a realistic failure scenario.

### Numerical correctness

- Check behavior in both complex64 (FP32) and complex128 (FP64),
  including `auto_precision` (eig in complex128, remaining operations
  in complex64). Account for intentional dtype conversions at API
  boundaries.
- Flag unintended dtype promotion or downcasting when it affects
  correctness, precision, or performance.
- Examine divisions, inversions, square roots, and complex branch
  choices near singularities. Relevant cases include kz ≈ 0 at
  Rayleigh anomalies and lossless media.
- Do not recommend silently masking NaN/Inf or automatically
  perturbing physical input parameters to conceal singularities.
  Prefer mathematically justified handling or explicit diagnostics.
- For changes to S-matrices, Redheffer products, or field/flux
  calculations, consider energy conservation (R + T + A = 1).
  Check reciprocity where applicable, accounting for mode
  normalization and propagation directions.
- Distinguish genuine numerical errors from expected FP32 rounding
  and convergence limitations of a truncated Fourier basis.

### Autograd

- Check gradients with respect to differentiable inputs, including
  permittivity, geometry, and layer parameters.
- Watch for in-place modifications of tensors needed by autograd,
  unintended `.detach()`, `.item()`, `.numpy()`, and `torch.no_grad()`
  in differentiable computational paths.
- For changes affecting eigendecomposition, its backward pass, or
  layer scattering, compare directional derivatives against central
  finite differences at suitable smooth test points.
- Pay particular attention to degenerate or nearly degenerate
  eigenvalues. Known limitation: #91 (`Eig.backward` produces
  inaccurate symmetry-breaking gradients at degeneracies).
  Do not report this existing limitation as a new regression unless
  the PR introduces or worsens it.
- Do not assume complex128 autograd gradients are a valid reference
  near degeneracies. Prefer independent numerical verification.

### Devices, memory, and performance

- Tests and benchmarks must respect the explicitly requested
  `--device`. Do not silently select a test device based on
  `torch.cuda.is_available()`.
- Flag incorrect device placement, unnecessary host-device transfers,
  and GPU synchronization in performance-critical paths.
- Examine additional matrix inversions, eigendecompositions, solves,
  and matrix multiplications. Dense matrix operations can be costly
  at high Fourier orders.
- Watch for avoidable copies and tensors retained by autograd.
  Memory scales approximately as B * N² per layer for stored
  matrices; unnecessary saved intermediates can be expensive.
- Prefer `torch.linalg.solve` over explicit inversion when only
  the inverse applied to another tensor is required.
- For wavelength batching, check leading batch dimensions,
  broadcasting, and preservation of independent batch entries.
- Report performance concerns only when the additional computational
  or memory cost is meaningful and can be explained concretely.

### Public API

- Preserve compatibility of documented APIs in `solwa.rcwa`,
  `solwa.geometry`, `solwa.materials`, and `solwa.utils`.
- Flag unintended behavioral changes, removed arguments, or changed
  return shapes. Intentional breaking changes require explicit
  documentation and appropriate release notes.
- Prefer deprecation warnings when a migration path is practical.

### GPU CI (Nebius)

- Workflows under `.github/workflows/gpu-*.yml` and `ci/nebius/`
  manage paid cloud resources. Treat resource leaks and exposure
  of cloud credentials as high-priority defects.
- GPU CI must only run for authorized main pushes, eligible
  same-repository PRs labelled `gpu-ci`, or manual dispatches
  initiated from main.
- Verify cleanup on success, failure, cancellation, and timeout.
  Consider both the workflow cleanup jobs and the independent
  reaper in `gpu-cleanup.yml`.
- Do not allow untrusted PR code to execute with cloud credentials.
- Avoid changes that can accidentally trigger costly GPU jobs.

### Tests and project conventions

- Keep PRs focused and diffs minimal.
- Functions should receive their dependencies explicitly rather
  than relying on hidden global or implicit state.
- New numerical behavior should have focused regression coverage,
  using FP32 and FP64 where relevant.
- Do not request tests for trivial implementation details or
  behavior already adequately covered.
- Performance-sensitive changes should include representative
  runtime and peak VRAM comparisons when feasible.
- Favor simple, mathematically justified solutions over defensive
  complexity.
