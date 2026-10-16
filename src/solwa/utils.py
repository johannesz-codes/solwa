import warnings

import numpy as np
import torch


def _axis_tensors(values, sim, axis_name):
    """Convert coordinates for field evaluation and for quadrature.

    Returns ``(field_axis, quad_axis)``. ``field_axis`` has the real dtype that
    matches the simulation and is passed to ``sim.poynting_xy``. ``quad_axis``
    keeps the floating-point precision of the input, so quadrature weights
    and spacing checks are never degraded by a lower-precision simulation.
    Non-floating inputs (lists, integer arrays) are read as float64.

    Raises ``ValueError`` if casting to the simulation dtype moves any
    coordinate by more than 1 % of the smallest gap between coordinates,
    since the fields would then be sampled away from the quadrature nodes.
    """
    sim_dtype = getattr(sim, "_dtype", None)
    if sim_dtype == torch.complex64:
        dtype = torch.float32
    elif sim_dtype == torch.complex128:
        dtype = torch.float64
    else:
        dtype = None
    device = getattr(sim, "_device", None)

    if torch.is_tensor(values):
        quad_axis = values.to(device=device)
    else:
        quad_axis = torch.as_tensor(np.asarray(values), device=device)
    if not quad_axis.is_floating_point():
        quad_axis = quad_axis.to(torch.float64)

    field_axis = quad_axis if dtype is None else quad_axis.to(dtype)

    if field_axis.dtype != quad_axis.dtype and quad_axis.numel() >= 2:
        values = quad_axis.detach().reshape(-1)
        shift = (field_axis.detach().reshape(-1).to(values.dtype) - values).abs()
        gaps = (values[1:] - values[:-1]).abs()
        gaps = gaps[gaps > 0]
        if gaps.numel() > 0 and shift.max() > 0.01 * gaps.min():
            raise ValueError(
                f"{axis_name} cannot be represented in the simulation's "
                f"{field_axis.dtype} precision without moving the sample "
                "points. Use a complex128 simulation or coordinates closer "
                "to zero."
            )

    return field_axis, quad_axis


def _uniform_step(axis):
    """Infer the spacing of an equally spaced, increasing axis.

    The spacing is the mean step ``(axis[-1] - axis[0]) / (n - 1)``. Every
    coordinate must be finite and every individual step positive and within
    the rounding error expected for coordinates of this magnitude and dtype.
    Grids from :class:`solwa.geometry` stay below
    ``eps * max|x|``; the bound allows four times that. If the bound exceeds
    1 % of the step, the precision cannot tell uniform from uneven spacing
    and the axis is rejected.

    Returns
    -------
    tuple
        ``(step, None)`` on success, otherwise ``(None, reason)`` with
        ``reason`` one of ``"nonfinite"``, ``"uneven"`` or ``"precision"``.
    """
    if axis.ndim != 1 or axis.numel() < 2 or not axis.is_floating_point():
        return None, "uneven"

    values = axis.detach()
    if not torch.all(torch.isfinite(values)):
        return None, "nonfinite"

    step = (axis[-1] - axis[0]) / (axis.numel() - 1)
    spacing = values[1:] - values[:-1]
    if torch.any(spacing <= 0):
        return None, "uneven"

    tol = 4 * torch.finfo(values.dtype).eps * values.abs().max()
    if tol > 0.01 * step.detach():
        return None, "precision"
    if torch.any((spacing - step.detach()).abs() > tol):
        return None, "uneven"

    return step, None


def _cell_step(cells, axis_name):
    """Return the width of adjacent, equally sized cells from their centers."""
    if cells.ndim != 1:
        raise ValueError(
            f"{axis_name} must be a one-dimensional array of cell centers."
        )
    if cells.numel() < 2:
        raise ValueError(
            f"{axis_name} must contain at least two adjacent cell centers so the "
            "cell width can be inferred."
        )

    step, reason = _uniform_step(cells)
    if reason == "nonfinite":
        raise ValueError(f"{axis_name} must contain only finite cell centers.")
    if reason == "precision":
        raise ValueError(
            f"The precision of {axis_name} is too low to verify equal cell "
            "spacing at this coordinate magnitude. Use float64 coordinates or "
            "shift the axis closer to zero."
        )
    if reason is not None:
        raise ValueError(
            f"{axis_name} must contain strictly increasing, equally spaced centers "
            "of adjacent cells."
        )

    return step


def _period(sim, index):
    """Return the simulation period along one axis, or None if unknown."""
    L = getattr(sim, "L", None)
    if L is None:
        return None
    return float(L[index])


def _looks_cell_centered(axis, period):
    """Return True if a point axis is the cell-center grid of a full period.

    A half-step phase alone cannot tell cell centers from a translated
    endpoint grid such as ``linspace(-L / 2, L / 2, n)``. So this only
    matches an axis of ``n`` equally spaced points with ``n * step`` equal to
    the period and the first point at ``step / 2``, which is exactly what
    :class:`solwa.geometry` produces for the simulation's unit cell.
    """
    if period is None:
        return False
    step, reason = _uniform_step(axis)
    if reason is not None:
        return False

    values = axis.detach()
    step = step.detach()
    # Tolerances cover the rounding of geometry grids (about eps * period
    # for the total width and eps * step for the first center).
    eps = torch.finfo(values.dtype).eps
    covers_period = torch.abs(values.numel() * step - period) < 16 * eps * abs(period)
    starts_half_step = torch.abs(values[0] - step / 2) < 16 * eps * step
    return bool(covers_period and starts_half_step)


def poynting_flux(
    sim,
    layer,
    x_points=None,
    y_points=None,
    z_prop=0.0,
    *,
    x_cells=None,
    y_cells=None,
):
    """
    Compute the z-directed Poynting flux through an XY plane.

    Coordinates can be supplied in one of two mutually exclusive forms:

    * ``x_points`` and ``y_points`` are actual sampling coordinates. The flux
      is integrated with the trapezoidal rule between the first and last point
      on each axis.
    * ``x_cells`` and ``y_cells`` are centers of adjacent, equally sized cells.
      Each field sample represents the full cell around its center, so midpoint
      quadrature is used: ``sum(Sz) * dx * dy``.

    The cell form is appropriate for the cell-centered grids generated by
    :class:`solwa.geometry`. Passing only a contiguous subset of those centers
    integrates only the corresponding subset of cells.

    Parameters
    ----------
    sim : object
        Simulation object providing the ``poynting_xy`` method.
    layer : int
        Index of the layer for which to evaluate the Poynting flux. Use -1 for
        the input interface.
    x_points, y_points : array-like or torch.Tensor, optional
        Actual x/y sampling coordinates for trapezoidal integration. Both must
        be provided together and cannot be mixed with ``x_cells``/``y_cells``.
    z_prop : float, optional
        z-position at which to evaluate the Poynting flux. Default is 0.0.
    x_cells, y_cells : array-like or torch.Tensor, optional
        Centers of adjacent, equally sized x/y cells for midpoint quadrature.
        Both must be provided together and cannot be mixed with point axes.

    Returns
    -------
    torch.Tensor
        Scalar tensor containing the integrated z-component of the Poynting
        vector.

    Raises
    ------
    ValueError
        If point and cell coordinates are mixed, an axis pair is incomplete,
        cell centers are not suitable for inferring a cell width, or the
        coordinates cannot be represented in the simulation's precision.

    Warns
    -----
    UserWarning
        If ``x_points`` or ``y_points`` are the cell-center grid of the
        simulation's full period, as produced by ``geometry.x``/``geometry.y``.
        Trapezoidal integration over cell centers leaves out half a cell at
        each edge; pass them as ``x_cells``/``y_cells`` to integrate the full
        cells. Subsets of cell centers cannot be told apart from endpoint
        grids and do not warn.
    """
    has_points = x_points is not None or y_points is not None
    has_cells = x_cells is not None or y_cells is not None

    if has_points and has_cells:
        raise ValueError(
            "Use either x_points/y_points or x_cells/y_cells, not a mix of both."
        )
    if not has_points and not has_cells:
        raise ValueError(
            "Provide either x_points/y_points or x_cells/y_cells for integration."
        )

    if has_points:
        if x_points is None or y_points is None:
            raise ValueError("x_points and y_points must be provided together.")
        x_axis, x_quad = _axis_tensors(x_points, sim, "x_points")
        y_axis, y_quad = _axis_tensors(y_points, sim, "y_points")
        x_centered = _looks_cell_centered(x_quad, _period(sim, 0))
        y_centered = _looks_cell_centered(y_quad, _period(sim, 1))
        if x_centered or y_centered:
            warnings.warn(
                "x_points/y_points look like cell centers. Trapezoidal "
                "integration only spans from the first to the last center and "
                "leaves out half a cell at each edge. Pass cell centers as "
                "x_cells/y_cells to integrate the full cells.",
                UserWarning,
                stacklevel=2,
            )
    else:
        if x_cells is None or y_cells is None:
            raise ValueError("x_cells and y_cells must be provided together.")
        x_axis, x_quad = _axis_tensors(x_cells, sim, "x_cells")
        y_axis, y_quad = _axis_tensors(y_cells, sim, "y_cells")

    if x_quad.ndim != 1 or y_quad.ndim != 1:
        raise ValueError("Poynting-flux integration axes must be one-dimensional.")

    Sz = sim.poynting_xy(layer, x_axis, y_axis, z_prop=z_prop)[2]

    if has_cells:
        dx = _cell_step(x_quad, "x_cells")
        dy = _cell_step(y_quad, "y_cells")
        return Sz.sum() * dx * dy

    # Integrate with the input precision of the coordinates (promoted with
    # the field dtype), so a lower-precision simulation cannot shift them.
    coord_dtype = torch.promote_types(x_quad.dtype, y_quad.dtype)
    dtype = torch.promote_types(Sz.dtype, coord_dtype)
    Sz = Sz.to(dtype)
    return torch.trapezoid(
        torch.trapezoid(Sz, y_quad.to(dtype), dim=1), x_quad.to(dtype), dim=0
    )
