import torch


def _is_cell_centered_axis(points):
    if points.ndim != 1 or points.numel() < 2:
        return False

    spacing = points[1:] - points[:-1]
    step = spacing[0]
    return torch.allclose(spacing, torch.full_like(spacing, step)) and bool(
        torch.isclose(points[0], 0.5 * step)
    )


def poynting_flux(sim, layer, x_points, y_points, z_prop):
    """
    Compute the z-component of the Poynting flux integrated over specified
    x and y grids for a given layer and z-position.

    This function accepts explicit x and y coordinate arrays and is therefore
    useful when you want to integrate the Poynting flux over a custom grid
    instead of the global ``solwa.rcwa_geo.x``/``y`` arrays.

    Parameters
    ----------
    sim : object
        Simulation object providing the ``poynting_xy`` method used to
        evaluate the Poynting vector on the specified grid.
    layer : int
        Index of the layer for which to evaluate the Poynting flux. See
        the ``layer`` argument of ``sim.poynting_xy`` (use -1 for the input
        interface).
    x_points : array-like or torch.Tensor
        1D array of x coordinates used for the integration (must match the
        x-dimension of the arrays returned by ``sim.poynting_xy`` when called
        with these points).
    y_points : array-like or torch.Tensor
        1D array of y coordinates used for the integration.
    z_prop : float
        z-position (same units as the simulation geometry) at which to
        evaluate the Poynting flux.

    Returns
    -------
    torch.Tensor
        A scalar tensor containing the integrated z-component of the Poynting
        vector over the provided x/y grids.

    Raises
    ------
    ValueError
        If the shapes of ``x_points``/``y_points`` are incompatible with the
        arrays produced by ``sim.poynting_xy``. The function does not explicitly
        validate shapes but ``torch.trapz`` will raise if the dimensions are
        inconsistent.

    See Also
    --------
    sim.poynting_xy : Returns the Poynting vector components on a specified grid.
    """
    Sz = sim.poynting_xy(layer, x_points, y_points, z_prop=z_prop)[2]
    x_points = torch.as_tensor(x_points, dtype=Sz.real.dtype, device=Sz.device)
    y_points = torch.as_tensor(y_points, dtype=Sz.real.dtype, device=Sz.device)

    if _is_cell_centered_axis(x_points) and _is_cell_centered_axis(y_points):
        dx = x_points[1] - x_points[0]
        dy = y_points[1] - y_points[0]
        return Sz.sum() * dx * dy

    return torch.trapz(torch.trapz(Sz, y_points, dim=1), x_points, dim=0)
