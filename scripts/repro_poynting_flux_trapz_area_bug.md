# Reproducer: `poynting_flux` Cell-Centered `trapz` Area Loss

`solwa.geometry.geometry.grid()` creates periodic cell-centered coordinates. For a unit cell with `N` samples in each direction, the coordinates are

```text
x_i = (i + 0.5) / N
y_i = (i + 0.5) / N
```

so the first and last samples are at `dx / 2` and `1 - dx / 2`, not at the cell boundaries.

`solwa.utils.poynting_flux` currently integrates `Sz` with nested `torch.trapz` calls. `trapz` integrates over the coordinate span it is given. On this midpoint grid, that span is `1 - dx` in `x` and `1 - dy` in `y`. For a constant field `Sz = 1`, the current result is therefore

```text
((N - 1) / N) * ((N - 1) / N)
```

instead of the correct midpoint/cell-average integral `mean(Sz) * Lx * Ly = 1`.

This is a pure grid-area factor. For fixed `nx`, `ny`, it is independent of wavelength, material, and field amplitude. That is why the error should appear exactly flat over wavelength when the same grid is used.

A single-layer ratio-style check can hide the problem. If both near-field fluxes are computed by the same `poynting_flux` call pattern,

```text
A = 1 - P_out / P_in
```

then both `P_out` and `P_in` carry the same multiplicative area factor, and the factor cancels.

Absolute layer absorption normalized to an independently computed incident power does not cancel it:

```text
A_abs = (P_in - P_out) / P_incident
```

Here only the near-field layer fluxes are scaled by the erroneous area factor. `P_incident` is independent, so `A_abs` is scaled too small by `((N - 1) / N)^2`. For `N = 100`, that factor is `0.9801`, or about `-1.99%`.
