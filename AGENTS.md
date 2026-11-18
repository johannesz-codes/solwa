# AGENTS.md

solwa is a differentiable RCWA library in PyTorch (`src/solwa/`), built mainly
for GPUs. Results must be numerically right, gradients must be right, and it
must stay fast and memory-lean.

## Code Review Rules

Report real bugs first (P0/P1): wrong numbers, wrong or NaN gradients, silent
data loss, crashes, large VRAM/time regressions, CI cost leaks. Skip style
nits and hypothetical edge cases that no realistic caller can reach. Do not ask
for defensive guards, extra validation or more tests unless a concrete input
breaks the code.

### Numerical correctness

- Code must be correct in both complex64 (FP32) and complex128 (FP64), and
  with `auto_precision` (eig in c128, rest in c64). Flag silent dtype
  promotion or downcasts, Python floats/`np` scalars that change dtype, and
  tolerances that only hold in FP64.
- Flag new `1/x`, `sqrt`, `log` or divisions where `x` can be 0 or on a branch
  cut, e.g. `kz = 0` at a Rayleigh anomaly or lossless half-spaces. NaN/Inf
  must not be silently masked to 0 or propagate unnoticed into results.
- Energy conservation (R + T + A = 1) and reciprocity are good sanity checks
  when a change touches S-matrices, Redheffer products or field/flux code.

### Autograd

- Gradients must stay correct, also for degenerate eigenvalues (C4-symmetric
  or homogeneous layers at normal incidence). Changes to the custom backward
  in `torch_eig.py` need a finite-difference check.
- Flag in-place ops on tensors autograd saves, `.detach()`/`.item()`/`.numpy()`
  inside the differentiable path, and `torch.no_grad` that cuts a gradient.

### Devices, memory and performance

- Tests and benchmarks take the device from the job (`--device`); they must
  never pick it silently via `torch.cuda.is_available()`.
- Flag tensors created on the wrong device or dtype, and hidden host-device
  syncs or copies in hot loops.
- Flag extra matrix inversions, eig calls or GEMMs on N×N matrices, extra
  copies of per-layer matrices, and FP64 tensors kept alive in the autograd
  graph. VRAM scales as B·N² per layer, so a single extra saved tensor matters.
  Prefer `solve` over `inv` when only a product is needed.
- Batching: wavelength batching adds a leading batch dim. Check that new code
  broadcasts over it and does not silently drop or mix batch entries.

### API

- Public API (`solwa.rcwa`, `solwa.geometry`, `solwa.materials`,
  `solwa.utils`) must stay backward compatible. Removals or renames need a
  `DeprecationWarning` and a release note.

### GPU CI (Nebius)

- Workflows under `.github/workflows/gpu-*.yml` and `ci/nebius/` create paid
  VMs. Every path, including failure, cancel and timeout, must delete the VM
  (`if: always()` cleanup). Flag anything that could leak a VM, run untrusted
  PR code with secrets, or start GPU CI on anything other than main pushes,
  PRs labelled `gpu-ci` or a manual dispatch from main.

### Project style

- PRs are lean with a minimal diff. Functions take the objects they work on
  explicitly (no hidden global or implicit state).
- One compact test per feature. Numerically sensitive code is tested in FP32
  and FP64. Trivial non-numerical changes need no test.
- Performance-relevant PRs should state timing and VRAM numbers.
