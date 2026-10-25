import numpy as np
import torch
from scipy.interpolate import interp1d


class _MaterialFn(torch.autograd.Function):
    """
    Custom PyTorch autograd function for material refractive index interpolation.

    Implements forward and backward passes for differentiable material property
    evaluation with finite-difference gradient computation.
    """

    @staticmethod
    def forward(ctx, wavelength, nk_data, n_interp, k_interp, dl=0.005) -> torch.Tensor:
        """
        Forward pass: interpolate material refractive index at given wavelength.

        Supports both scalar and batched wavelength tensors.

        Parameters
        ----------
        ctx : context object
            PyTorch context for saving variables for backward pass.
        wavelength : torch.Tensor
            Wavelength(s) at which to evaluate the refractive index. Can be a
            scalar (0-D) tensor or a 1-D batch tensor of shape (B,).
        nk_data : numpy.ndarray
            Array of wavelength, n, k data points.
        n_interp : callable
            Interpolation function for real part of refractive index.
        k_interp : callable
            Interpolation function for imaginary part of refractive index.
        dl : float, optional
            Wavelength step for finite-difference gradient computation. Default is 0.005.

        Returns
        -------
        torch.Tensor
            Complex refractive index (n + ik) at the given wavelength(s). Shape
            matches the input ``wavelength`` shape.
        """
        wavelength_np = wavelength.detach().cpu().numpy()
        is_scalar = wavelength_np.ndim == 0
        if is_scalar:
            wavelength_np = wavelength_np.reshape(1)

        out_dtype = (
            torch.complex128
            if (
                (wavelength.dtype is torch.float64)
                or (wavelength.dtype is torch.complex128)
            )
            else torch.complex64
        )

        lo, hi = nk_data[0, 0], nk_data[-1, 0]

        def _interp(wl):
            """Vectorised boundary-clamped nk interpolation."""
            wl_clamp = np.clip(wl, lo, hi)
            n_arr = n_interp(wl_clamp)
            k_arr = k_interp(wl_clamp)
            n_arr = np.where(
                wl < lo, nk_data[0, 1], np.where(wl > hi, nk_data[-1, 1], n_arr)
            )
            k_arr = np.where(
                wl < lo, nk_data[0, 2], np.where(wl > hi, nk_data[-1, 2], k_arr)
            )
            return n_arr + 1.0j * k_arr

        nk_value = _interp(wavelength_np)
        nk_value_m = _interp(wavelength_np - dl)
        nk_value_p = _interp(wavelength_np + dl)

        dnk_dl = (nk_value_p - nk_value_m) / (2 * dl)
        ctx.dnk_dl = torch.tensor(
            dnk_dl[0] if is_scalar else dnk_dl,
            dtype=out_dtype,
            device=wavelength.device,
        )

        result = nk_value[0] if is_scalar else nk_value
        return torch.tensor(result, dtype=out_dtype, device=wavelength.device)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        """
        Backward pass: compute gradient with respect to wavelength.

        Parameters
        ----------
        ctx : context object
            PyTorch context containing saved variables from forward pass.
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output.

        Returns
        -------
        tuple
            Gradients with respect to inputs (grad_wavelength, None, None, None, None).
        """
        grad = 2 * torch.real(torch.conj(grad_output) * ctx.dnk_dl)
        return grad, None, None, None, None


class Material:
    """
    Material refractive index interpolator with automatic differentiation support.

    Loads material optical properties (n, k) from a data file and provides
    wavelength-dependent refractive index with gradient computation support.

    Parameters
    ----------
    nk_file : str
        Path to file containing wavelength-dependent refractive index data.
        File format: each line contains wavelength, n, k (3 columns) or
        index, wavelength, n, k (4 columns).
    dl : float, optional
        Wavelength step for finite-difference gradient computation. Default is 0.005.
    *args
        Additional positional arguments (currently unused).
    **kwargs
        Additional keyword arguments (currently unused).

    Attributes
    ----------
    nk_data : numpy.ndarray
        Array of wavelength, n, k data points.
    n_interp : scipy.interpolate.interp1d
        Cubic interpolation function for real part of refractive index.
    k_interp : scipy.interpolate.interp1d
        Cubic interpolation function for imaginary part of refractive index.
    dl : float
        Wavelength step for gradient computation.
    """

    def __init__(self, nk_file, dl=0.005, *args, **kwargs):
        f_nk = open(nk_file)
        data = f_nk.readlines()
        f_nk.close()

        nk_data = []
        for i in range(len(data)):
            _line = data[i].split()
            _lamb0 = _line[0]
            if len(_line) == 3:
                _n = _line[1]
                _k = _line[2]
            elif len(_line) == 4:
                _n = _line[2]
                _k = _line[3]
            else:
                raise ValueError("unknown dimensions of refraction data")

            nk_data.append([float(_lamb0), float(_n), float(_k)])
        nk_data = np.array(nk_data)

        self.n_interp = interp1d(nk_data[:, 0], nk_data[:, 1], kind="cubic")
        self.k_interp = interp1d(nk_data[:, 0], nk_data[:, 2], kind="cubic")

        self.nk_data = nk_data
        self.dl = dl

    def apply(self, wavelength, dl=None) -> torch.Tensor:
        """
        Evaluate material refractive index at specified wavelength.

        Parameters
        ----------
        wavelength : float or torch.Tensor
            Wavelength at which to evaluate the refractive index.
        dl : float, optional
            Wavelength step for finite-difference gradient computation.
            If None, uses the value set during initialization. Default is None.

        Returns
        -------
        torch.Tensor
            Complex refractive index (n + ik) at the given wavelength.
        """
        if dl is None:
            dl = self.dl

        return _MaterialFn.apply(wavelength, self.nk_data, self.n_interp, self.k_interp, dl)  # type: ignore
