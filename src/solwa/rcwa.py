import functools
import warnings

import torch
from math import pi
from .torch_eig import Eig


def _vcat(tensors):
    """Block-row concatenation that keeps leading (wavelength) batch dims."""
    return torch.cat(tensors, dim=-2)


def _hcat(tensors):
    """Block-column concatenation that keeps leading (wavelength) batch dims."""
    return torch.cat(tensors, dim=-1)


def _precision_io(method):
    """Cast tensor arguments to the compute precision and results back to the
    user-facing dtype when ``auto_precision`` is enabled (no-op otherwise)."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "auto_precision", False) or self._io_depth > 0:
            return method(self, *args, **kwargs)
        self._io_depth += 1
        try:
            args = [self._cast(a, self._dtype) for a in args]
            kwargs = {k: self._cast(v, self._dtype) for k, v in kwargs.items()}
            return self._cast(method(self, *args, **kwargs), self._io_dtype)
        finally:
            self._io_depth -= 1

    return wrapper


class rcwa:
    """
    Rigorous Coupled-Wave Analysis (RCWA) simulation engine.

    Implements the Fourier Modal Method (FMM) for simulating electromagnetic
    wave propagation through periodic structures. Supports GPU acceleration
    and automatic differentiation for optimization.

    Uses Lorentz-Heaviside units with speed of light = 1 and time harmonics
    notation exp(-jωt).

    Examples
    --------
    >>> import torch
    >>> import solwa
    >>> sim = solwa.rcwa(freq=1/500, order=[5, 5], L=[300, 300])
    >>> sim.add_input_layer(eps=1.0)
    >>> sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    >>> sim.add_layer(thickness=100, eps=2.25)
    >>> sim.solve_global_smatrix()
    """

    # Simulation setting
    def __init__(
        self,
        freq,
        order,
        L,
        *,
        dtype=torch.complex64,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        offload_device=None,
        stable_eig_grad=True,
        avoid_Pinv_instability=False,
        max_Pinv_instability=0.005,
        factorization="li",
        auto_precision=False,
    ):
        """
        Initialize Rigorous Coupled Wave Analysis (RCWA) simulation.

        Uses Lorentz-Heaviside units with speed of light = 1 and
        time harmonics notation exp(-jωt).

        Parameters
        ----------
        freq : float or torch.Tensor
            Simulation frequency (unit: length^-1). A 1-D tensor of
            frequencies solves all of them in one batched run (experimental,
            emits a warning; geometry is shared). Permittivities may then
            carry a leading wavelength axis (``[B]`` for homogeneous media,
            ``[B, nx, ny]`` for patterns), and results get a leading axis of
            size B. Field methods and
            ``return_layer`` do not support batched frequencies yet.
        order : list of int
            Fourier truncation order [x_order, y_order].
        L : list of float
            Lattice constant [Lx, Ly] (unit: length).
        dtype : torch.dtype, optional
            Simulation data type (torch.complex64 or torch.complex128). Default is torch.complex64.
        device : torch.device, optional
            Simulation device (torch.device('cpu') or torch.device('cuda')). Default is CUDA if available, otherwise CPU.
        offload_device : torch.device or str or None, optional
            Device to offload tensors that are not currently in use. When set, completed
            layer tensors are automatically moved to this device after each layer is added
            and brought back to the compute device on demand. Typical use is
            ``offload_device=torch.device('cpu')`` to keep GPU memory usage low when
            simulating many layers. Default is None (offloading disabled).
        stable_eig_grad : bool, optional
            Stabilize gradient calculation of eigendecomposition. Default is True.
        avoid_Pinv_instability : bool, optional
            Avoid instability of P inverse (P: H to E field transformation). Default is False.
        max_Pinv_instability : float, optional
            Allowed maximum instability value for P inverse. Default is 0.005.
        factorization : str, optional
            Fourier factorization of the in-plane permittivity. 'li' applies Li's
            inverse rule to the field components normal to the material interfaces
            (L. Li, JOSA A 14, 2758 (1997)), which converges much faster for TM
            polarization and crossed gratings. 'laurent' uses the plain Laurent rule
            (the behaviour of solwa <= 0.2.0). Default is 'li'.
        auto_precision : bool, optional
            Mixed-precision mode. If True, the layer eigendecomposition always
            runs in complex128 and all other computations run in complex64,
            whatever ``dtype`` is. In complex64 the eigendecomposition is what
            limits accuracy, so this mode gets close to complex128 results
            while keeping roughly the memory footprint of complex64 and most of
            its speed advantage over complex128.

            ``dtype`` then only sets the interface precision:

            * Tensor arguments of the public methods (permittivities, angles,
              field coordinates, ...) are cast to complex64 / float32.
            * Results of the public methods (``S_parameters``, ``field_*``,
              ``poynting*``, ``diffraction_angle``, ``return_layer``) are
              returned in ``dtype``, or its real counterpart for real values.
            * Internal attributes (e.g. ``S``, ``C``, ``kz_norm``, the layer
              S-matrices) stay in complex64.

            The casts are differentiable, so autograd works as usual and
            gradients reach the input tensors in their own dtype. Gradients
            are usually much more accurate than with plain complex64, but not
            always at complex128 level: the matrix inversions and their
            backward pass still run in complex64, so very small gradients can
            remain inaccurate. Default is False.
        """

        # Hardware
        if dtype != torch.complex64 and dtype != torch.complex128:
            raise ValueError("Invalid simulation data type")
        self.auto_precision = bool(auto_precision)
        self._io_depth = 0
        self._io_dtype = dtype
        if self.auto_precision:
            self._dtype = torch.complex64
            self._eig_dtype = torch.complex128
        else:
            self._dtype = dtype
            self._eig_dtype = None
        self._device = device

        # Device offloading
        if offload_device is None:
            self._offload_device = None
        elif isinstance(offload_device, torch.device):
            self._offload_device = offload_device
        else:
            self._offload_device = torch.device(offload_device)

        if factorization not in ("li", "laurent"):
            raise ValueError("Invalid factorization. Choose 'li' or 'laurent'.")
        self.factorization = factorization

        # Stabilize the gradient of eigendecomposition
        self.stable_eig_grad = True if stable_eig_grad else False

        # Stability setting for inverse matrix of P and Q
        if avoid_Pinv_instability is True:
            self.avoid_Pinv_instability = True
            self.max_Pinv_instability = max_Pinv_instability
            self.Pinv_instability = []
            self.Qinv_instability = []
        else:
            self.avoid_Pinv_instability = False
            self.max_Pinv_instability = None
            self.Pinv_instability = None
            self.Qinv_instability = None

        # Simulation parameters (tensor inputs follow the compute precision)
        freq = self._cast(freq, self._dtype)
        L = self._cast(L, self._dtype)
        self.freq = torch.as_tensor(
            freq, dtype=self._dtype, device=self._device
        )  # unit^-1
        if self.freq.dim() > 0:
            warnings.warn(
                "Support for batched wavelengths is experimental and results "
                "should be treated with caution.",
                UserWarning,
                stacklevel=2,
            )
        self.omega = 2 * pi * self.freq.real  # same as k0a
        self.L = torch.as_tensor(L, dtype=self._dtype, device=self._device)

        # Fourier order
        self.order = order
        self.order_x = torch.linspace(
            -self.order[0],
            self.order[0],
            2 * self.order[0] + 1,
            dtype=torch.int64,
            device=self._device,
        )
        self.order_y = torch.linspace(
            -self.order[1],
            self.order[1],
            2 * self.order[1] + 1,
            dtype=torch.int64,
            device=self._device,
        )
        self.order_N = len(self.order_x) * len(self.order_y)

        # Lattice vector
        self.L = L  # unit
        self.Gx_norm, self.Gy_norm = 1 / (L[0] * self.freq), 1 / (L[1] * self.freq)

        # Input and output layer (Default: free space)
        self.eps_in = torch.tensor(1.0, dtype=self._dtype, device=self._device)
        self.mu_in = torch.tensor(1.0, dtype=self._dtype, device=self._device)
        self.eps_out = torch.tensor(1.0, dtype=self._dtype, device=self._device)
        self.mu_out = torch.tensor(1.0, dtype=self._dtype, device=self._device)

        # Internal layers
        self.layer_N = 0  # total number of layers
        self.thickness = []
        self.eps_conv, self.mu_conv = [], []

        # Internal layer eigenmodes
        self.P, self.Q = [], []
        self.kz_norm, self.E_eigvec, self.H_eigvec = [], [], []

        # Internal layer mode coupling coefficiencts
        self.Cf, self.Cb = [], []

        # Single layer scattering matrices
        self.layer_S11, self.layer_S21, self.layer_S12, self.layer_S22 = [], [], [], []

    @_precision_io
    def add_input_layer(self, eps=1.0, mu=1.0):
        """
        Add input layer to the simulation.

        If this function is not used, simulation will be performed under
        free space input layer.

        Parameters
        ----------
        eps : float or torch.Tensor, optional
            Relative permittivity of the input layer. Default is 1.0.
        mu : float or torch.Tensor, optional
            Relative permeability of the input layer. Default is 1.0.
        """

        self.eps_in = torch.as_tensor(eps, dtype=self._dtype, device=self._device)
        self.mu_in = torch.as_tensor(mu, dtype=self._dtype, device=self._device)
        self.Sin = []

    @_precision_io
    def add_output_layer(self, eps=1.0, mu=1.0):
        """
        Add output layer to the simulation.

        If this function is not used, simulation will be performed under
        free space output layer.

        Parameters
        ----------
        eps : float or torch.Tensor, optional
            Relative permittivity of the output layer. Default is 1.0.
        mu : float or torch.Tensor, optional
            Relative permeability of the output layer. Default is 1.0.
        """

        self.eps_out = torch.as_tensor(eps, dtype=self._dtype, device=self._device)
        self.mu_out = torch.as_tensor(mu, dtype=self._dtype, device=self._device)
        self.Sout = []

    @_precision_io
    def set_incident_angle(self, inc_ang, azi_ang, angle_layer="input"):
        """
        Set incident angle for the simulation.

        Parameters
        ----------
        inc_ang : float or torch.Tensor
            Incident angle (unit: radian).
        azi_ang : float or torch.Tensor
            Azimuthal angle (unit: radian).
        angle_layer : str, optional
            Reference layer to calculate angle. Options are 'i', 'in', 'input' for
            input layer, or 'o', 'out', 'output' for output layer. Default is 'input'.
        """

        self.inc_ang = torch.as_tensor(inc_ang, dtype=self._dtype, device=self._device)
        self.azi_ang = torch.as_tensor(azi_ang, dtype=self._dtype, device=self._device)

        if angle_layer in ["i", "in", "input"]:
            self.angle_layer = "input"
        elif angle_layer in ["o", "out", "output"]:
            self.angle_layer = "output"
        else:
            raise ValueError("Invalid angle layer")

        self._kvectors()

    @_precision_io
    def add_layer(self, thickness, eps=1.0, mu=1.0):
        """
        Add an internal layer to the simulation.

        Parameters
        ----------
        thickness : float
            Layer thickness (unit: length).
        eps : float or torch.Tensor, optional
            Relative permittivity of the layer. Can be a scalar for homogeneous
            material or a tensor for inhomogeneous material. Default is 1.0.
        mu : float or torch.Tensor, optional
            Relative permeability of the layer. Can be a scalar for homogeneous
            material or a tensor for inhomogeneous material. Default is 1.0.
        """

        is_eps_homogenous = (
            isinstance(eps, float) or isinstance(eps, complex) or (eps.dim() <= 1)
        )
        is_mu_homogenous = (
            isinstance(mu, float) or isinstance(mu, float) or (mu.dim() <= 1)
        )

        self.eps_conv.append(
            self._bc(eps, 2)
            * torch.eye(self.order_N, dtype=self._dtype, device=self._device)
            if is_eps_homogenous
            else self._material_conv(eps)
        )
        self.mu_conv.append(
            self._bc(mu, 2)
            * torch.eye(self.order_N, dtype=self._dtype, device=self._device)
            if is_mu_homogenous
            else self._material_conv(mu)
        )

        self.layer_N += 1
        self.thickness.append(thickness)

        self._layer_is_homogeneous = is_eps_homogenous and is_mu_homogenous
        if self._layer_is_homogeneous:
            self._eigen_decomposition_homogenous(eps, mu)
        elif not is_eps_homogenous and self.factorization == "li":
            self._eigen_decomposition(*self._material_conv_li(eps))
        else:
            self._eigen_decomposition()

        self._solve_layer_smatrix()

        if self._offload_device is not None:
            self._offload_layer_data()

    # Solve simulation
    def solve_global_smatrix(self):
        """
        Solve the global scattering matrix (S-matrix) for the entire structure.

        Combines all layer scattering matrices into a global S-matrix using
        the recursive doubling algorithm.
        """

        # Initialization
        if self.layer_N > 0:
            S11 = self._d(self.layer_S11[0])
            S21 = self._d(self.layer_S21[0])
            S12 = self._d(self.layer_S12[0])
            S22 = self._d(self.layer_S22[0])
            C = [[self._d(self.Cf[0])], [self._d(self.Cb[0])]]
        else:
            shape = self.freq.shape + (2 * self.order_N, 2 * self.order_N)
            eye = torch.eye(2 * self.order_N, dtype=self._dtype, device=self._device)
            S11 = eye.expand(shape)
            S21 = torch.zeros(shape, dtype=self._dtype, device=self._device)
            S12 = torch.zeros(shape, dtype=self._dtype, device=self._device)
            S22 = eye.expand(shape)
            C = [[], []]

        # Connection
        for i in range(self.layer_N - 1):
            [S11, S21, S12, S22], C = self._RS_prod(
                Sm=[S11, S21, S12, S22],
                Sn=[
                    self._d(self.layer_S11[i + 1]),
                    self._d(self.layer_S21[i + 1]),
                    self._d(self.layer_S12[i + 1]),
                    self._d(self.layer_S22[i + 1]),
                ],
                Cm=C,
                Cn=[[self._d(self.Cf[i + 1])], [self._d(self.Cb[i + 1])]],
            )

        if hasattr(self, "Sin"):
            # input layer coupling
            [S11, S21, S12, S22], C = self._RS_prod(
                Sm=[self.Sin[0], self.Sin[1], self.Sin[2], self.Sin[3]],
                Sn=[S11, S21, S12, S22],
                Cm=[[], []],
                Cn=C,
            )

        if hasattr(self, "Sout"):
            # output layer coupling
            [S11, S21, S12, S22], C = self._RS_prod(
                Sm=[S11, S21, S12, S22],
                Sn=[self.Sout[0], self.Sout[1], self.Sout[2], self.Sout[3]],
                Cm=C,
                Cn=[[], []],
            )

        self.S = [S11, S21, S12, S22]
        self.C = C

        if self._offload_device is not None:
            self.C = [
                [self._to_offload(t) for t in self.C[0]],
                [self._to_offload(t) for t in self.C[1]],
            ]

    # Returns
    @_precision_io
    def diffraction_angle(self, orders, *, layer="output", unit="radian"):
        """
        Calculate diffraction angles for the selected orders.

        Parameters
        ----------
        orders : array-like
            Selected diffraction orders. Recommended shape is Nx2.
        layer : str, optional
            Selected layer. Options are 'i', 'in', 'input' for input layer,
            or 'o', 'out', 'output' for output layer. Default is 'output'.
        unit : str, optional
            Unit of the output angles. Options are 'r', 'rad', 'radian' for
            radians, or 'd', 'deg', 'degree' for degrees. Default is 'radian'.

        Returns
        -------
        tuple of torch.Tensor
            (inclination_angle, azimuthal_angle) for each order.
        """

        orders = torch.as_tensor(
            orders, dtype=torch.int64, device=self._device
        ).reshape([-1, 2])

        if layer in ["i", "in", "input"]:
            layer = "input"
        elif layer in ["o", "out", "output"]:
            layer = "output"
        else:
            raise ValueError("Invalid layer selected")

        if unit in ["r", "rad", "radian"]:
            unit = "radian"
        elif unit in ["d", "deg", "degree"]:
            unit = "degree"
        else:
            raise ValueError("Invalid unit. Set as 'radian' or 'degree'.")

        # Matching indices
        order_indices = self._matching_indices(orders)

        eps = self.eps_in if layer == "input" else self.eps_out
        mu = self.mu_in if layer == "input" else self.mu_out

        kx_norm = self.Kx_norm_dn[..., order_indices]
        ky_norm = self.Ky_norm_dn[..., order_indices]
        Kt_norm_dn = torch.sqrt(kx_norm**2 + ky_norm**2)
        kz_norm = torch.sqrt(self._bc(eps * mu, 1) - kx_norm**2 - ky_norm**2)
        inc_angle = torch.atan2(torch.real(Kt_norm_dn), torch.real(kz_norm))
        azi_angle = torch.atan2(torch.real(ky_norm), torch.real(kx_norm))

        if unit == "degree":
            inc_angle = (180.0 / pi) * inc_angle
            azi_angle = (180.0 / pi) * azi_angle

        return inc_angle, azi_angle

    @_precision_io
    def return_layer(self, layer_num, nx=100, ny=100):
        """
        Return spatial distributions of permittivity and permeability for the selected layer.

        The permittivity and permeability are recovered from the truncated Fourier orders
        using inverse Fourier transform.

        Parameters
        ----------
        layer_num : int
            Selected layer index.
        nx : int, optional
            Number of grid points in x-direction. Default is 100.
        ny : int, optional
            Number of grid points in y-direction. Default is 100.

        Returns
        -------
        tuple of torch.Tensor
            (eps_recover, mu_recover) containing the recovered permittivity and permeability distributions.
        """

        self._require_scalar_freq("return_layer")

        eps_fft = torch.zeros([nx, ny], dtype=self._dtype, device=self._device)
        mu_fft = torch.zeros([nx, ny], dtype=self._dtype, device=self._device)
        eps_conv = self._d(self.eps_conv[layer_num])
        mu_conv = self._d(self.mu_conv[layer_num])
        for i in range(-2 * self.order[0], 2 * self.order[0] + 1):
            for j in range(-2 * self.order[1], 2 * self.order[1] + 1):
                if i >= 0 and j >= 0:
                    eps_fft[i, j] = eps_conv[i * (2 * self.order[1] + 1) + j, 0]
                    mu_fft[i, j] = mu_conv[i * (2 * self.order[1] + 1) + j, 0]
                elif i >= 0 and j < 0:
                    eps_fft[i, j] = eps_conv[i * (2 * self.order[1] + 1), -j]
                    mu_fft[i, j] = mu_conv[i * (2 * self.order[1] + 1), -j]
                elif i < 0 and j >= 0:
                    eps_fft[i, j] = eps_conv[j, -i * (2 * self.order[1] + 1)]
                    mu_fft[i, j] = mu_conv[j, -i * (2 * self.order[1] + 1)]
                else:
                    eps_fft[i, j] = eps_conv[0, -i * (2 * self.order[1] + 1) - j]
                    mu_fft[i, j] = mu_conv[0, -i * (2 * self.order[1] + 1) - j]

        eps_recover = torch.fft.ifftn(eps_fft) * nx * ny
        mu_recover = torch.fft.ifftn(mu_fft) * nx * ny

        return eps_recover, mu_recover

    @_precision_io
    def S_parameters(
        self,
        orders,
        *,
        direction="forward",
        port="transmission",
        polarization="xx",
        ref_order=[0, 0],
        power_norm=True,
        evanscent=1e-3,
    ):
        """
        Calculate S-parameters for specified orders and polarizations.

        Parameters
        ----------
        orders : array-like
            Selected diffraction orders. Recommended shape is Nx2.
        direction : str, optional
            Direction of light propagation. Options are 'f' or 'forward', 'b' or 'backward'.
            Default is 'forward'.
        port : str, optional
            Port specification. Options are 't' or 'transmission', 'r' or 'reflection'.
            Default is 'transmission'.
        polarization : str, optional
            Input and output polarization. For xy-polarization: 'xx', 'yx', 'xy', 'yy'.
            For ps-polarization: 'pp', 'sp', 'ps', 'ss'. Default is 'xx'.
        ref_order : array-like, optional
            Reference order for calculating S-parameters. Recommended shape is 1x2 or Nx2.
            Default is [0, 0].
        power_norm : bool, optional
            If True, the absolute square of S-parameters corresponds to the ratio of power.
            Default is True.
        evanscent : float, optional
            Criteria for judging the evanescent field. If power_norm=True and
            real(kz_norm)/imag(kz_norm) < evanescent, function returns 0. Default is 1e-3.

        Returns
        -------
        torch.Tensor
            S-parameters for the specified orders and polarizations.
        """

        orders = torch.as_tensor(
            orders, dtype=torch.int64, device=self._device
        ).reshape([-1, 2])

        if direction in ["f", "forward"]:
            direction = "forward"
        elif direction in ["b", "backward"]:
            direction = "backward"
        else:
            raise ValueError(
                "Invalid propagation direction. Set as 'forward' or 'backward'."
            )

        if port in ["t", "transmission"]:
            port = "transmission"
        elif port in ["r", "reflection"]:
            port = "reflection"
        else:
            raise ValueError("Invalid port. Set as 'transmission' or 'reflection'.")

        if polarization not in ["xx", "yx", "xy", "yy", "pp", "sp", "ps", "ss"]:
            raise ValueError(
                "Invalid polarization. Choose one of 'xx','yx','xy','yy','pp','sp','ps','ss'."
            )

        ref_order = torch.as_tensor(
            ref_order, dtype=torch.int64, device=self._device
        ).reshape([1, 2])

        # Matching order indices
        order_indices = self._matching_indices(orders)
        ref_order_index = self._matching_indices(ref_order)

        if polarization in ["xx", "yx", "xy", "yy"]:
            # Matching order indices with polarization
            if polarization == "yx" or polarization == "yy":
                order_indices = order_indices + self.order_N
            if polarization == "xy" or polarization == "yy":
                ref_order_index = ref_order_index + self.order_N

            # power normalization factor
            if power_norm:
                Kz_norm_dn_in_complex = torch.sqrt(
                    self._bc(self.eps_in * self.mu_in, 1)
                    - self.Kx_norm_dn**2
                    - self.Ky_norm_dn**2
                )
                is_evanescent_in = (
                    torch.abs(
                        torch.real(Kz_norm_dn_in_complex)
                        / torch.imag(Kz_norm_dn_in_complex)
                    )
                    < evanscent
                )
                Kz_norm_dn_in = torch.where(
                    is_evanescent_in,
                    torch.real(torch.zeros_like(Kz_norm_dn_in_complex)),
                    torch.real(Kz_norm_dn_in_complex),
                )
                Kz_norm_dn_in = torch.hstack((Kz_norm_dn_in, Kz_norm_dn_in))

                Kz_norm_dn_out_complex = torch.sqrt(
                    self._bc(self.eps_out * self.mu_out, 1)
                    - self.Kx_norm_dn**2
                    - self.Ky_norm_dn**2
                )
                is_evanescent_out = (
                    torch.abs(
                        torch.real(Kz_norm_dn_out_complex)
                        / torch.imag(Kz_norm_dn_out_complex)
                    )
                    < evanscent
                )
                Kz_norm_dn_out = torch.where(
                    is_evanescent_out,
                    torch.real(torch.zeros_like(Kz_norm_dn_out_complex)),
                    torch.real(Kz_norm_dn_out_complex),
                )
                Kz_norm_dn_out = torch.hstack((Kz_norm_dn_out, Kz_norm_dn_out))

                Kx_norm_dn = torch.hstack(
                    (torch.real(self.Kx_norm_dn), torch.real(self.Kx_norm_dn))
                )
                Ky_norm_dn = torch.hstack(
                    (torch.real(self.Ky_norm_dn), torch.real(self.Ky_norm_dn))
                )

                if polarization == "xx":
                    numerator_pol, denominator_pol = Kx_norm_dn, Kx_norm_dn
                elif polarization == "xy":
                    numerator_pol, denominator_pol = Kx_norm_dn, Ky_norm_dn
                elif polarization == "yx":
                    numerator_pol, denominator_pol = Ky_norm_dn, Kx_norm_dn
                elif polarization == "yy":
                    numerator_pol, denominator_pol = Ky_norm_dn, Ky_norm_dn

                if direction == "forward" and port == "transmission":
                    numerator_kz = Kz_norm_dn_out
                    denominator_kz = Kz_norm_dn_in
                elif direction == "forward" and port == "reflection":
                    numerator_kz = Kz_norm_dn_in
                    denominator_kz = Kz_norm_dn_in
                elif direction == "backward" and port == "reflection":
                    numerator_kz = Kz_norm_dn_out
                    denominator_kz = Kz_norm_dn_out
                elif direction == "backward" and port == "transmission":
                    numerator_kz = Kz_norm_dn_in
                    denominator_kz = Kz_norm_dn_out

                # Safe kz: an evanescent order or reference has kz = 0, and the
                # inf/NaN would reach the shared gradient of the whole batch.
                order_kz = numerator_kz[..., order_indices]
                ref_kz = denominator_kz[..., ref_order_index]
                invalid = (order_kz == 0) | (ref_kz == 0)
                order_kz = torch.where(invalid, 1.0, order_kz)
                ref_kz = torch.where(invalid, 1.0, ref_kz)
                normalization = torch.sqrt(
                    (1 + (numerator_pol[..., order_indices] / order_kz) ** 2)
                    / (1 + (denominator_pol[..., ref_order_index] / ref_kz) ** 2)
                )
                normalization = normalization * torch.sqrt(order_kz / ref_kz)
                normalization = torch.where(invalid, 0.0, normalization)
            else:
                normalization = 1.0

            # Get S-parameters
            if direction == "forward" and port == "transmission":
                S = self.S[0][..., order_indices, ref_order_index] * normalization
            elif direction == "forward" and port == "reflection":
                S = self.S[1][..., order_indices, ref_order_index] * normalization
            elif direction == "backward" and port == "reflection":
                S = self.S[2][..., order_indices, ref_order_index] * normalization
            elif direction == "backward" and port == "transmission":
                S = self.S[3][..., order_indices, ref_order_index] * normalization

            S = torch.where(torch.isinf(S), torch.zeros_like(S), S)
            S = torch.where(torch.isnan(S), torch.zeros_like(S), S)

            return S

        elif polarization in ["pp", "sp", "ps", "ss"]:
            if direction == "forward" and port == "transmission":
                idx = 0
                order_sign, ref_sign = 1, 1
                order_k0_norm2 = self._bc(self.eps_out * self.mu_out, 1)
                ref_k0_norm2 = self._bc(self.eps_in * self.mu_in, 1)
            elif direction == "forward" and port == "reflection":
                idx = 1
                order_sign, ref_sign = -1, 1
                order_k0_norm2 = self._bc(self.eps_in * self.mu_in, 1)
                ref_k0_norm2 = self._bc(self.eps_in * self.mu_in, 1)
            elif direction == "backward" and port == "reflection":
                idx = 2
                order_sign, ref_sign = 1, -1
                order_k0_norm2 = self._bc(self.eps_out * self.mu_out, 1)
                ref_k0_norm2 = self._bc(self.eps_out * self.mu_out, 1)
            elif direction == "backward" and port == "transmission":
                idx = 3
                order_sign, ref_sign = -1, -1
                order_k0_norm2 = self._bc(self.eps_in * self.mu_in, 1)
                ref_k0_norm2 = self._bc(self.eps_out * self.mu_out, 1)

            order_Kx_norm_dn = self.Kx_norm_dn[..., order_indices]
            order_Ky_norm_dn = self.Ky_norm_dn[..., order_indices]
            order_Kt_norm_dn = torch.sqrt(order_Kx_norm_dn**2 + order_Ky_norm_dn**2)
            order_Kz_norm_dn = order_sign * torch.abs(
                torch.real(
                    torch.sqrt(
                        order_k0_norm2 - order_Kx_norm_dn**2 - order_Ky_norm_dn**2
                    )
                )
            )
            order_Kz_norm_dn_complex = torch.sqrt(
                order_k0_norm2 - order_Kx_norm_dn**2 - order_Ky_norm_dn**2
            )
            order_is_evanescent = (
                torch.abs(
                    torch.real(order_Kz_norm_dn_complex)
                    / torch.imag(order_Kz_norm_dn_complex)
                )
                < evanscent
            )

            order_inc_angle = torch.atan2(
                torch.real(order_Kt_norm_dn), order_Kz_norm_dn
            )
            order_azi_angle = torch.atan2(
                torch.real(order_Ky_norm_dn), torch.real(order_Kx_norm_dn)
            )

            ref_Kx_norm_dn = self.Kx_norm_dn[..., ref_order_index]
            ref_Ky_norm_dn = self.Ky_norm_dn[..., ref_order_index]
            ref_Kt_norm_dn = torch.sqrt(ref_Kx_norm_dn**2 + ref_Ky_norm_dn**2)
            ref_Kz_norm_dn = ref_sign * torch.abs(
                torch.real(
                    torch.sqrt(ref_k0_norm2 - ref_Kx_norm_dn**2 - ref_Ky_norm_dn**2)
                )
            )
            ref_Kz_norm_dn_complex = torch.sqrt(
                ref_k0_norm2 - ref_Kx_norm_dn**2 - ref_Ky_norm_dn**2
            )
            ref_is_evanescent = (
                torch.abs(
                    torch.real(ref_Kz_norm_dn_complex)
                    / torch.imag(ref_Kz_norm_dn_complex)
                )
                < evanscent
            )

            ref_inc_angle = torch.atan2(torch.real(ref_Kt_norm_dn), ref_Kz_norm_dn)
            ref_azi_angle = torch.atan2(
                torch.real(ref_Ky_norm_dn), torch.real(ref_Kx_norm_dn)
            )

            xx = self.S[idx][..., order_indices, ref_order_index]
            xy = self.S[idx][..., order_indices, ref_order_index + self.order_N]
            yx = self.S[idx][..., order_indices + self.order_N, ref_order_index]
            yy = self.S[idx][
                ..., order_indices + self.order_N, ref_order_index + self.order_N
            ]

            xx = torch.where(order_is_evanescent, torch.zeros_like(xx), xx)
            xy = torch.where(order_is_evanescent, torch.zeros_like(xy), xy)
            yx = torch.where(order_is_evanescent, torch.zeros_like(yx), yx)
            yy = torch.where(order_is_evanescent, torch.zeros_like(yy), yy)

            if ref_is_evanescent.all():
                S = torch.zeros_like(xx)
                return S

            if polarization == "pp":
                S = (
                    torch.cos(order_azi_angle)
                    / torch.cos(order_inc_angle)
                    * torch.cos(ref_inc_angle)
                    * torch.cos(ref_azi_angle)
                    * xx
                    + torch.sin(order_azi_angle)
                    / torch.cos(order_inc_angle)
                    * torch.cos(ref_inc_angle)
                    * torch.cos(ref_azi_angle)
                    * yx
                    + torch.cos(order_azi_angle)
                    / torch.cos(order_inc_angle)
                    * torch.cos(ref_inc_angle)
                    * torch.sin(ref_azi_angle)
                    * xy
                    + torch.sin(order_azi_angle)
                    / torch.cos(order_inc_angle)
                    * torch.cos(ref_inc_angle)
                    * torch.sin(ref_azi_angle)
                    * yy
                )
            elif polarization == "ps":
                S = (
                    torch.cos(order_azi_angle)
                    / torch.cos(order_inc_angle)
                    * (-1)
                    * torch.sin(ref_azi_angle)
                    * xx
                    + torch.sin(order_azi_angle)
                    / torch.cos(order_inc_angle)
                    * (-1)
                    * torch.sin(ref_azi_angle)
                    * yx
                    + torch.cos(order_azi_angle)
                    / torch.cos(order_inc_angle)
                    * torch.cos(ref_azi_angle)
                    * xy
                    + torch.sin(order_azi_angle)
                    / torch.cos(order_inc_angle)
                    * torch.cos(ref_azi_angle)
                    * yy
                )
            elif polarization == "sp":
                S = (
                    -torch.sin(order_azi_angle)
                    * torch.cos(ref_inc_angle)
                    * torch.cos(ref_azi_angle)
                    * xx
                    + torch.cos(order_azi_angle)
                    * torch.cos(ref_inc_angle)
                    * torch.cos(ref_azi_angle)
                    * yx
                    + -torch.sin(order_azi_angle)
                    * torch.cos(ref_inc_angle)
                    * torch.sin(ref_azi_angle)
                    * xy
                    + torch.cos(order_azi_angle)
                    * torch.cos(ref_inc_angle)
                    * torch.sin(ref_azi_angle)
                    * yy
                )
            elif polarization == "ss":
                S = (
                    -torch.sin(order_azi_angle) * (-1) * torch.sin(ref_azi_angle) * xx
                    + torch.cos(order_azi_angle) * (-1) * torch.sin(ref_azi_angle) * yx
                    + -torch.sin(order_azi_angle) * torch.cos(ref_azi_angle) * xy
                    + torch.cos(order_azi_angle) * torch.cos(ref_azi_angle) * yy
                )

            if power_norm:
                Kz_norm_dn_in_complex = torch.sqrt(
                    self._bc(self.eps_in * self.mu_in, 1)
                    - self.Kx_norm_dn**2
                    - self.Ky_norm_dn**2
                )
                is_evanescent_in = (
                    torch.abs(
                        torch.real(Kz_norm_dn_in_complex)
                        / torch.imag(Kz_norm_dn_in_complex)
                    )
                    < evanscent
                )
                Kz_norm_dn_in = torch.where(
                    is_evanescent_in,
                    torch.real(torch.zeros_like(Kz_norm_dn_in_complex)),
                    torch.real(Kz_norm_dn_in_complex),
                )
                Kz_norm_dn_in = torch.hstack((Kz_norm_dn_in, Kz_norm_dn_in))

                Kz_norm_dn_out_complex = torch.sqrt(
                    self._bc(self.eps_out * self.mu_out, 1)
                    - self.Kx_norm_dn**2
                    - self.Ky_norm_dn**2
                )
                is_evanescent_out = (
                    torch.abs(
                        torch.real(Kz_norm_dn_out_complex)
                        / torch.imag(Kz_norm_dn_out_complex)
                    )
                    < evanscent
                )
                Kz_norm_dn_out = torch.where(
                    is_evanescent_out,
                    torch.abs(torch.real(Kz_norm_dn_out_complex)),
                    torch.real(Kz_norm_dn_out_complex),
                )
                Kz_norm_dn_out = torch.hstack((Kz_norm_dn_out, Kz_norm_dn_out))

                Kx_norm_dn = torch.hstack(
                    (torch.real(self.Kx_norm_dn), torch.real(self.Kx_norm_dn))
                )
                Ky_norm_dn = torch.hstack(
                    (torch.real(self.Ky_norm_dn), torch.real(self.Ky_norm_dn))
                )

                if direction == "forward" and port == "transmission":
                    numerator_kz = Kz_norm_dn_out
                    denominator_kz = Kz_norm_dn_in
                elif direction == "forward" and port == "reflection":
                    numerator_kz = Kz_norm_dn_in
                    denominator_kz = Kz_norm_dn_in
                elif direction == "backward" and port == "reflection":
                    numerator_kz = Kz_norm_dn_out
                    denominator_kz = Kz_norm_dn_out
                elif direction == "backward" and port == "transmission":
                    numerator_kz = Kz_norm_dn_in
                    denominator_kz = Kz_norm_dn_out

                # Safe kz: sqrt(0) and 1/0 would put NaN into the shared gradient.
                order_kz = numerator_kz[..., order_indices]
                ref_kz = denominator_kz[..., ref_order_index]
                invalid = (order_kz == 0) | (ref_kz == 0)
                normalization = torch.sqrt(
                    torch.where(invalid, 1.0, order_kz)
                    / torch.where(invalid, 1.0, ref_kz)
                )
                normalization = torch.where(invalid, 0.0, normalization)
            else:
                normalization = 1.0

            S = torch.where(torch.isinf(S), torch.zeros_like(S), S)
            S = torch.where(torch.isnan(S), torch.zeros_like(S), S)

            return torch.where(ref_is_evanescent, 0.0, S * normalization)

        else:
            return None

    @_precision_io
    def source_planewave(
        self, *, amplitude=[1.0, 0.0], direction="forward", notation="xy"
    ):
        """
        Generate a plane wave source.

        Parameters
        ----------
        amplitude : list or array-like, optional
            Amplitudes at the matched diffraction orders.
            For 'xy' notation: [Ex_amp, Ey_amp].
            For 'ps' notation: [Ep_amp, Es_amp].
            Recommended shape is 1x2. Default is [1.0, 0.0].
        direction : str, optional
            Incident direction. Options are 'f' or 'forward', 'b' or 'backward'.
            Default is 'forward'.
        notation : str, optional
            Amplitude notation. Options are 'xy' for xy-polarization,
            'ps' for ps-polarization. Default is 'xy'.
        """

        self.source_fourier(
            amplitude=amplitude, orders=[0, 0], direction=direction, notation=notation
        )

    @_precision_io
    def source_fourier(self, *, amplitude, orders, direction="forward", notation="xy"):
        """
        Generate a Fourier source with multiple orders.

        Parameters
        ----------
        amplitude : array-like
            Amplitudes at the matched diffraction orders.
            Format: [([Ex_amp, Ey_amp] at orders[0]), ([Ex_amp, Ey_amp] at orders[1]), ...].
            Recommended shape is Nx2.
        orders : array-like
            Diffraction orders corresponding to each amplitude. Recommended shape is Nx2.
        direction : str, optional
            Incident direction. Options are 'f' or 'forward', 'b' or 'backward'.
            Default is 'forward'.
        notation : str, optional
            Amplitude notation. Options are 'xy' for xy-polarization,
            'ps' for ps-polarization. Default is 'xy'.
        """
        amplitude = torch.as_tensor(
            amplitude, dtype=self._dtype, device=self._device
        ).reshape([-1, 2])
        orders = torch.as_tensor(
            orders, dtype=torch.int64, device=self._device
        ).reshape([-1, 2])

        if direction in ["f", "forward"]:
            direction = "forward"
        elif direction in ["b", "backward"]:
            direction = "backward"
        else:
            raise ValueError(
                "Invalid source direction. Set as 'forward' or 'backward'."
            )

        if notation not in ["xy", "ps"]:
            raise ValueError(
                "Invalid amplitude notation. Set as 'xy' or 'ps' notation."
            )

        # Matching indices
        order_indices = self._matching_indices(orders)

        self.source_direction = direction

        E_i = torch.zeros([2 * self.order_N, 1], dtype=self._dtype, device=self._device)
        E_i[order_indices, 0] = amplitude[:, 0]
        E_i[order_indices + self.order_N, 0] = amplitude[:, 1]

        # Convert ps-pol to xy-pol
        if notation == "ps":
            if direction == "forward":
                eps, mu = self.eps_in, self.mu_in
                sign = 1
            else:
                eps, mu = self.eps_out, self.mu_out
                sign = -1

            Kt_norm_dn = torch.sqrt(self.Kx_norm_dn**2 + self.Ky_norm_dn**2)
            Kz_norm_dn = sign * torch.abs(
                torch.real(
                    torch.sqrt(
                        self._bc(eps * mu, 1) - self.Kx_norm_dn**2 - self.Ky_norm_dn**2
                    )
                )
            )

            inc_angle = torch.atan2(torch.real(Kt_norm_dn), Kz_norm_dn)
            azi_angle = torch.atan2(
                torch.real(self.Ky_norm_dn), torch.real(self.Kx_norm_dn)
            )

            tmp1 = _vcat(
                (
                    torch.diag_embed(torch.cos(inc_angle) * torch.cos(azi_angle)),
                    torch.diag_embed(torch.cos(inc_angle) * torch.sin(azi_angle)),
                )
            )
            tmp2 = _vcat(
                (
                    torch.diag_embed(-torch.sin(azi_angle)),
                    torch.diag_embed(torch.cos(azi_angle)),
                )
            )
            ps2xy = _hcat((tmp1, tmp2))

            E_i = torch.matmul(ps2xy.to(self._dtype), E_i)

        self.E_i = E_i

    @_precision_io
    def field_xz(self, x_axis, z_axis, y):
        """
        Calculate XZ-plane electromagnetic field distribution.

        Returns the electric and magnetic field components at the specified y position.

        Parameters
        ----------
        x_axis : torch.Tensor
            x-direction sampling coordinates.
        z_axis : torch.Tensor
            z-direction sampling coordinates.
        y : float
            Selected y coordinate position.

        Returns
        -------
        tuple
            ([Ex, Ey, Ez], [Hx, Hy, Hz]) where each component is a torch.Tensor.
        """

        self._require_scalar_freq("field_xz")

        if not isinstance(x_axis, torch.Tensor) or not isinstance(z_axis, torch.Tensor):
            raise TypeError("x and z axis must be torch.Tensor type.")

        x_axis = x_axis.reshape([-1, 1, 1])

        Kx_norm, Ky_norm = self.Kx_norm, self.Ky_norm

        Ex_split, Ey_split, Ez_split = [], [], []
        Hx_split, Hy_split, Hz_split = [], [], []

        # layer number
        zp = torch.zeros(len(self.thickness), device=self._device)
        zm = torch.zeros(len(self.thickness), device=self._device)
        layer_num = torch.zeros([len(z_axis)], dtype=torch.int64, device=self._device)
        layer_num[z_axis < 0.0] = -1

        for ti in range(len(self.thickness)):
            zp[ti:] += self.thickness[ti]
        zm[1:] = zp[0:-1]

        for bi in range(len(zp)):
            layer_num[z_axis > zp[bi]] += 1

        prev_layer_num = -2
        for zi in range(len(z_axis)):
            # Input and output layers
            if layer_num[zi] == -1 or layer_num[zi] == self.layer_N:
                Kx_norm_dn = self.Kx_norm_dn
                Ky_norm_dn = self.Ky_norm_dn

                if layer_num[zi] == -1:
                    z_prop = z_axis[zi] if z_axis[zi] <= 0.0 else 0.0
                    if layer_num[zi] != prev_layer_num:
                        eps = self.eps_in if hasattr(self, "eps_in") else 1.0
                        mu = self.mu_in if hasattr(self, "mu_in") else 1.0
                        Vi = self.Vi if hasattr(self, "Vi") else self.Vf
                        Kz_norm_dn = torch.sqrt(
                            eps * mu - Kx_norm_dn**2 - Ky_norm_dn**2
                        )
                        Kz_norm_dn = torch.where(
                            torch.imag(Kz_norm_dn) > 0,
                            torch.conj(Kz_norm_dn),
                            Kz_norm_dn,
                        ).reshape([-1, 1])
                        Kz_norm_dn = torch.vstack((Kz_norm_dn, Kz_norm_dn))
                elif layer_num[zi] == self.layer_N:
                    if len(zp) == 0:
                        z_prop = z_axis[zi]
                    else:
                        z_prop = (
                            z_axis[zi] - zp[-1] if z_axis[zi] - zp[-1] >= 0.0 else 0.0
                        )
                    if layer_num[zi] != prev_layer_num:
                        eps = self.eps_out if hasattr(self, "eps_in") else 1.0
                        mu = self.mu_out if hasattr(self, "mu_in") else 1.0
                        Vo = self.Vo if hasattr(self, "Vo") else self.Vf
                        Kz_norm_dn = torch.sqrt(
                            eps * mu - Kx_norm_dn**2 - Ky_norm_dn**2
                        )
                        Kz_norm_dn = torch.where(
                            torch.imag(Kz_norm_dn) < 0,
                            torch.conj(Kz_norm_dn),
                            Kz_norm_dn,
                        ).reshape([-1, 1])
                        Kz_norm_dn = torch.vstack((Kz_norm_dn, Kz_norm_dn))

                # Phase
                z_phase = torch.exp(1.0j * self.omega * Kz_norm_dn * z_prop)

                # Fourier domain fields
                # [diffraction order]
                if layer_num[zi] == -1 and self.source_direction == "forward":
                    Exy_p = self.E_i * z_phase
                    Hxy_p = torch.matmul(Vi, Exy_p)
                    Exy_m = torch.matmul(self.S[1], self.E_i) * torch.conj(z_phase)
                    Hxy_m = torch.matmul(-Vi, Exy_m)
                elif layer_num[zi] == -1 and self.source_direction == "backward":
                    Exy_p = torch.zeros_like(self.E_i)
                    Hxy_p = torch.zeros_like(self.E_i)
                    Exy_m = torch.matmul(self.S[3], self.E_i) * torch.conj(z_phase)
                    Hxy_m = torch.matmul(-Vi, Exy_m)
                elif (
                    layer_num[zi] == self.layer_N and self.source_direction == "forward"
                ):
                    Exy_p = torch.matmul(self.S[0], self.E_i) * z_phase
                    Hxy_p = torch.matmul(Vo, Exy_p)
                    Exy_m = torch.zeros_like(self.E_i)
                    Hxy_m = torch.zeros_like(self.E_i)
                elif (
                    layer_num[zi] == self.layer_N
                    and self.source_direction == "backward"
                ):
                    Exy_p = torch.matmul(self.S[2], self.E_i) * z_phase
                    Hxy_p = torch.matmul(Vo, Exy_p)
                    Exy_m = self.E_i * torch.conj(z_phase)
                    Hxy_m = torch.matmul(-Vo, Exy_m)

                Ex_mn = Exy_p[: self.order_N] + Exy_m[: self.order_N]
                Ey_mn = Exy_p[self.order_N :] + Exy_m[self.order_N :]
                Hz_mn = (
                    torch.matmul(Kx_norm, Ey_mn) / mu
                    - torch.matmul(Ky_norm, Ex_mn) / mu
                )
                Hx_mn = Hxy_p[: self.order_N] + Hxy_m[: self.order_N]
                Hy_mn = Hxy_p[self.order_N :] + Hxy_m[self.order_N :]
                Ez_mn = (
                    torch.matmul(Ky_norm, Hx_mn) / eps
                    - torch.matmul(Kx_norm, Hy_mn) / eps
                )

                # Spatial domain fields
                xy_phase = torch.exp(
                    1.0j * self.omega * (self.Kx_norm_dn * x_axis + self.Ky_norm_dn * y)
                )
                Ex_split.append(torch.sum(Ex_mn.reshape(1, 1, -1) * xy_phase, dim=2))
                Ey_split.append(torch.sum(Ey_mn.reshape(1, 1, -1) * xy_phase, dim=2))
                Ez_split.append(torch.sum(Ez_mn.reshape(1, 1, -1) * xy_phase, dim=2))
                Hx_split.append(torch.sum(Hx_mn.reshape(1, 1, -1) * xy_phase, dim=2))
                Hy_split.append(torch.sum(Hy_mn.reshape(1, 1, -1) * xy_phase, dim=2))
                Hz_split.append(torch.sum(Hz_mn.reshape(1, 1, -1) * xy_phase, dim=2))

            # Internal layers
            else:
                z_prop = z_axis[zi] - zm[layer_num[zi]]

                if layer_num[zi] != prev_layer_num:
                    if self.source_direction == "forward":
                        C = torch.matmul(self._d(self.C[0][layer_num[zi]]), self.E_i)
                    elif self.source_direction == "backward":
                        C = torch.matmul(self._d(self.C[1][layer_num[zi]]), self.E_i)

                    kz_norm = self._d(self.kz_norm[layer_num[zi]])
                    E_eigvec = self._d(self.E_eigvec[layer_num[zi]])
                    H_eigvec = self._d(self.H_eigvec[layer_num[zi]])

                    Cp = torch.diag(C[: 2 * self.order_N, 0])
                    Cm = torch.diag(C[2 * self.order_N :, 0])

                    eps_conv_inv = torch.linalg.inv(
                        self._d(self.eps_conv[layer_num[zi]])
                    )
                    mu_conv_inv = torch.linalg.inv(self._d(self.mu_conv[layer_num[zi]]))

                # Phase
                z_phase_p = torch.diag(torch.exp(1.0j * self.omega * kz_norm * z_prop))
                z_phase_m = torch.diag(
                    torch.exp(
                        1.0j
                        * self.omega
                        * kz_norm
                        * (self.thickness[layer_num[zi]] - z_prop)
                    )
                )

                # Fourier domain fields
                # [diffraction order, eigenmode number]
                Exy_p = torch.matmul(E_eigvec, z_phase_p)
                Ex_p = Exy_p[: self.order_N, :]
                Ey_p = Exy_p[self.order_N :, :]
                Hz_p = torch.matmul(
                    mu_conv_inv, torch.matmul(Kx_norm, Ey_p)
                ) - torch.matmul(mu_conv_inv, torch.matmul(Ky_norm, Ex_p))
                Exy_m = torch.matmul(E_eigvec, z_phase_m)
                Ex_m = Exy_m[: self.order_N, :]
                Ey_m = Exy_m[self.order_N :, :]
                Hz_m = torch.matmul(
                    mu_conv_inv, torch.matmul(Kx_norm, Ey_m)
                ) - torch.matmul(mu_conv_inv, torch.matmul(Ky_norm, Ex_m))
                Hxy_p = torch.matmul(H_eigvec, z_phase_p)
                Hx_p = Hxy_p[: self.order_N, :]
                Hy_p = Hxy_p[self.order_N :, :]
                Ez_p = torch.matmul(
                    eps_conv_inv, torch.matmul(Ky_norm, Hx_p)
                ) - torch.matmul(eps_conv_inv, torch.matmul(Kx_norm, Hy_p))
                Hxy_m = torch.matmul(-H_eigvec, z_phase_m)
                Hx_m = Hxy_m[: self.order_N, :]
                Hy_m = Hxy_m[self.order_N :, :]
                Ez_m = torch.matmul(
                    eps_conv_inv, torch.matmul(Ky_norm, Hx_m)
                ) - torch.matmul(eps_conv_inv, torch.matmul(Kx_norm, Hy_m))

                Ex_mn = torch.sum(
                    torch.matmul(Ex_p, Cp) + torch.matmul(Ex_m, Cm), dim=1
                )
                Ey_mn = torch.sum(
                    torch.matmul(Ey_p, Cp) + torch.matmul(Ey_m, Cm), dim=1
                )
                Ez_mn = torch.sum(
                    torch.matmul(Ez_p, Cp) + torch.matmul(Ez_m, Cm), dim=1
                )
                Hx_mn = torch.sum(
                    torch.matmul(Hx_p, Cp) + torch.matmul(Hx_m, Cm), dim=1
                )
                Hy_mn = torch.sum(
                    torch.matmul(Hy_p, Cp) + torch.matmul(Hy_m, Cm), dim=1
                )
                Hz_mn = torch.sum(
                    torch.matmul(Hz_p, Cp) + torch.matmul(Hz_m, Cm), dim=1
                )

                # Spatial domain fields
                xy_phase = torch.exp(
                    1.0j * self.omega * (self.Kx_norm_dn * x_axis + self.Ky_norm_dn * y)
                )
                Ex_split.append(torch.sum(Ex_mn.reshape(1, 1, -1) * xy_phase, dim=2))
                Ey_split.append(torch.sum(Ey_mn.reshape(1, 1, -1) * xy_phase, dim=2))
                Ez_split.append(torch.sum(Ez_mn.reshape(1, 1, -1) * xy_phase, dim=2))
                Hx_split.append(torch.sum(Hx_mn.reshape(1, 1, -1) * xy_phase, dim=2))
                Hy_split.append(torch.sum(Hy_mn.reshape(1, 1, -1) * xy_phase, dim=2))
                Hz_split.append(torch.sum(Hz_mn.reshape(1, 1, -1) * xy_phase, dim=2))

            prev_layer_num = layer_num[zi]

        Ex = torch.cat(Ex_split, dim=1)
        Ey = torch.cat(Ey_split, dim=1)
        Ez = torch.cat(Ez_split, dim=1)
        Hx = torch.cat(Hx_split, dim=1)
        Hy = torch.cat(Hy_split, dim=1)
        Hz = torch.cat(Hz_split, dim=1)

        return [Ex, Ey, Ez], [Hx, Hy, Hz]

    @_precision_io
    def field_yz(self, y_axis, z_axis, x):
        """
        Calculate YZ-plane electromagnetic field distribution.

        Returns the electric and magnetic field components at the specified x position.

        Parameters
        ----------
        y_axis : torch.Tensor
            y-direction sampling coordinates.
        z_axis : torch.Tensor
            z-direction sampling coordinates.
        x : float
            Selected x coordinate position.

        Returns
        -------
        tuple
            ([Ex, Ey, Ez], [Hx, Hy, Hz]) where each component is a torch.Tensor.
        """

        self._require_scalar_freq("field_yz")

        if not isinstance(y_axis, torch.Tensor) or not isinstance(z_axis, torch.Tensor):
            raise TypeError("y and z axis must be torch.Tensor type.")

        y_axis = y_axis.reshape([-1, 1, 1])

        Kx_norm, Ky_norm = self.Kx_norm, self.Ky_norm

        Ex_split, Ey_split, Ez_split = [], [], []
        Hx_split, Hy_split, Hz_split = [], [], []

        # layer number
        zp = torch.zeros(len(self.thickness), device=self._device)
        zm = torch.zeros(len(self.thickness), device=self._device)
        layer_num = torch.zeros([len(z_axis)], dtype=torch.int64, device=self._device)
        layer_num[z_axis < 0.0] = -1

        for ti in range(len(self.thickness)):
            zp[ti:] += self.thickness[ti]

        for bi in range(len(zp)):
            layer_num[z_axis > zp[bi]] += 1
        zm[1:] = zp[0:-1]

        prev_layer_num = -2
        for zi in range(len(z_axis)):
            # Input and output layers
            if layer_num[zi] == -1 or layer_num[zi] == self.layer_N:
                Kx_norm_dn = self.Kx_norm_dn
                Ky_norm_dn = self.Ky_norm_dn

                if layer_num[zi] == -1:
                    z_prop = z_axis[zi] if z_axis[zi] <= 0.0 else 0.0
                    if layer_num[zi] != prev_layer_num:
                        eps = self.eps_in if hasattr(self, "eps_in") else 1.0
                        mu = self.mu_in if hasattr(self, "mu_in") else 1.0
                        Vi = self.Vi if hasattr(self, "Vi") else self.Vf
                        Kz_norm_dn = torch.sqrt(
                            eps * mu - Kx_norm_dn**2 - Ky_norm_dn**2
                        )
                        Kz_norm_dn = torch.where(
                            torch.imag(Kz_norm_dn) > 0,
                            torch.conj(Kz_norm_dn),
                            Kz_norm_dn,
                        ).reshape([-1, 1])
                        Kz_norm_dn = torch.vstack((Kz_norm_dn, Kz_norm_dn))
                elif layer_num[zi] == self.layer_N:
                    if len(zp) == 0:
                        z_prop = z_axis[zi]
                    else:
                        z_prop = (
                            z_axis[zi] - zp[-1] if z_axis[zi] - zp[-1] >= 0.0 else 0.0
                        )
                    if layer_num[zi] != prev_layer_num:
                        eps = self.eps_out if hasattr(self, "eps_in") else 1.0
                        mu = self.mu_out if hasattr(self, "mu_in") else 1.0
                        Vo = self.Vo if hasattr(self, "Vo") else self.Vf
                        Kz_norm_dn = torch.sqrt(
                            eps * mu - Kx_norm_dn**2 - Ky_norm_dn**2
                        )
                        Kz_norm_dn = torch.where(
                            torch.imag(Kz_norm_dn) < 0,
                            torch.conj(Kz_norm_dn),
                            Kz_norm_dn,
                        ).reshape([-1, 1])
                        Kz_norm_dn = torch.vstack((Kz_norm_dn, Kz_norm_dn))

                # Phase
                z_phase = torch.exp(1.0j * self.omega * Kz_norm_dn * z_prop)

                # Fourier domain fields
                # [diffraction order]
                if layer_num[zi] == -1 and self.source_direction == "forward":
                    Exy_p = self.E_i * z_phase
                    Hxy_p = torch.matmul(Vi, Exy_p)
                    Exy_m = torch.matmul(self.S[1], self.E_i) * torch.conj(z_phase)
                    Hxy_m = torch.matmul(-Vi, Exy_m)
                elif layer_num[zi] == -1 and self.source_direction == "backward":
                    Exy_p = torch.zeros_like(self.E_i)
                    Hxy_p = torch.zeros_like(self.E_i)
                    Exy_m = torch.matmul(self.S[3], self.E_i) * torch.conj(z_phase)
                    Hxy_m = torch.matmul(-Vi, Exy_m)
                elif (
                    layer_num[zi] == self.layer_N and self.source_direction == "forward"
                ):
                    Exy_p = torch.matmul(self.S[0], self.E_i) * z_phase
                    Hxy_p = torch.matmul(Vo, Exy_p)
                    Exy_m = torch.zeros_like(self.E_i)
                    Hxy_m = torch.zeros_like(self.E_i)
                elif (
                    layer_num[zi] == self.layer_N
                    and self.source_direction == "backward"
                ):
                    Exy_p = torch.matmul(self.S[2], self.E_i) * z_phase
                    Hxy_p = torch.matmul(Vo, Exy_p)
                    Exy_m = self.E_i * torch.conj(z_phase)
                    Hxy_m = torch.matmul(-Vo, Exy_m)

                Ex_mn = Exy_p[: self.order_N] + Exy_m[: self.order_N]
                Ey_mn = Exy_p[self.order_N :] + Exy_m[self.order_N :]
                Hz_mn = (
                    torch.matmul(Kx_norm, Ey_mn) / mu
                    - torch.matmul(Ky_norm, Ex_mn) / mu
                )
                Hx_mn = Hxy_p[: self.order_N] + Hxy_m[: self.order_N]
                Hy_mn = Hxy_p[self.order_N :] + Hxy_m[self.order_N :]
                Ez_mn = (
                    torch.matmul(Ky_norm, Hx_mn) / eps
                    - torch.matmul(Kx_norm, Hy_mn) / eps
                )

                # Spatial domain fields
                xy_phase = torch.exp(
                    1.0j * self.omega * (self.Kx_norm_dn * x + self.Ky_norm_dn * y_axis)
                )
                Ex_split.append(torch.sum(Ex_mn.reshape(1, 1, -1) * xy_phase, dim=2))
                Ey_split.append(torch.sum(Ey_mn.reshape(1, 1, -1) * xy_phase, dim=2))
                Ez_split.append(torch.sum(Ez_mn.reshape(1, 1, -1) * xy_phase, dim=2))
                Hx_split.append(torch.sum(Hx_mn.reshape(1, 1, -1) * xy_phase, dim=2))
                Hy_split.append(torch.sum(Hy_mn.reshape(1, 1, -1) * xy_phase, dim=2))
                Hz_split.append(torch.sum(Hz_mn.reshape(1, 1, -1) * xy_phase, dim=2))

            # Internal layers
            else:
                if layer_num[zi] > 0:
                    z_prop = z_axis[zi] - zp[layer_num[zi] - 1]
                else:
                    z_prop = z_axis[zi]

                if layer_num[zi] != prev_layer_num:
                    if self.source_direction == "forward":
                        C = torch.matmul(self._d(self.C[0][layer_num[zi]]), self.E_i)
                    elif self.source_direction == "backward":
                        C = torch.matmul(self._d(self.C[1][layer_num[zi]]), self.E_i)

                    kz_norm = self._d(self.kz_norm[layer_num[zi]])
                    E_eigvec = self._d(self.E_eigvec[layer_num[zi]])
                    H_eigvec = self._d(self.H_eigvec[layer_num[zi]])

                    Cp = torch.diag(C[: 2 * self.order_N, 0])
                    Cm = torch.diag(C[2 * self.order_N :, 0])

                    eps_conv_inv = torch.linalg.inv(
                        self._d(self.eps_conv[layer_num[zi]])
                    )
                    mu_conv_inv = torch.linalg.inv(self._d(self.mu_conv[layer_num[zi]]))

                # Phase
                z_phase_p = torch.diag(torch.exp(1.0j * self.omega * kz_norm * z_prop))
                z_phase_m = torch.diag(
                    torch.exp(
                        1.0j
                        * self.omega
                        * kz_norm
                        * (self.thickness[layer_num[zi]] - z_prop)
                    )
                )

                # Fourier domain fields
                # [diffraction order, eigenmode number]
                Exy_p = torch.matmul(E_eigvec, z_phase_p)
                Ex_p = Exy_p[: self.order_N, :]
                Ey_p = Exy_p[self.order_N :, :]
                Hz_p = torch.matmul(
                    mu_conv_inv, torch.matmul(Kx_norm, Ey_p)
                ) - torch.matmul(mu_conv_inv, torch.matmul(Ky_norm, Ex_p))
                Exy_m = torch.matmul(E_eigvec, z_phase_m)
                Ex_m = Exy_m[: self.order_N, :]
                Ey_m = Exy_m[self.order_N :, :]
                Hz_m = torch.matmul(
                    mu_conv_inv, torch.matmul(Kx_norm, Ey_m)
                ) - torch.matmul(mu_conv_inv, torch.matmul(Ky_norm, Ex_m))
                Hxy_p = torch.matmul(H_eigvec, z_phase_p)
                Hx_p = Hxy_p[: self.order_N, :]
                Hy_p = Hxy_p[self.order_N :, :]
                Ez_p = torch.matmul(
                    eps_conv_inv, torch.matmul(Ky_norm, Hx_p)
                ) - torch.matmul(eps_conv_inv, torch.matmul(Kx_norm, Hy_p))
                Hxy_m = torch.matmul(-H_eigvec, z_phase_m)
                Hx_m = Hxy_m[: self.order_N, :]
                Hy_m = Hxy_m[self.order_N :, :]
                Ez_m = torch.matmul(
                    eps_conv_inv, torch.matmul(Ky_norm, Hx_m)
                ) - torch.matmul(eps_conv_inv, torch.matmul(Kx_norm, Hy_m))

                Ex_mn = torch.sum(
                    torch.matmul(Ex_p, Cp) + torch.matmul(Ex_m, Cm), dim=1
                )
                Ey_mn = torch.sum(
                    torch.matmul(Ey_p, Cp) + torch.matmul(Ey_m, Cm), dim=1
                )
                Ez_mn = torch.sum(
                    torch.matmul(Ez_p, Cp) + torch.matmul(Ez_m, Cm), dim=1
                )
                Hx_mn = torch.sum(
                    torch.matmul(Hx_p, Cp) + torch.matmul(Hx_m, Cm), dim=1
                )
                Hy_mn = torch.sum(
                    torch.matmul(Hy_p, Cp) + torch.matmul(Hy_m, Cm), dim=1
                )
                Hz_mn = torch.sum(
                    torch.matmul(Hz_p, Cp) + torch.matmul(Hz_m, Cm), dim=1
                )

                # Spatial domain fields
                xy_phase = torch.exp(
                    1.0j * self.omega * (self.Kx_norm_dn * x + self.Ky_norm_dn * y_axis)
                )
                Ex_split.append(torch.sum(Ex_mn.reshape(1, 1, -1) * xy_phase, dim=2))
                Ey_split.append(torch.sum(Ey_mn.reshape(1, 1, -1) * xy_phase, dim=2))
                Ez_split.append(torch.sum(Ez_mn.reshape(1, 1, -1) * xy_phase, dim=2))
                Hx_split.append(torch.sum(Hx_mn.reshape(1, 1, -1) * xy_phase, dim=2))
                Hy_split.append(torch.sum(Hy_mn.reshape(1, 1, -1) * xy_phase, dim=2))
                Hz_split.append(torch.sum(Hz_mn.reshape(1, 1, -1) * xy_phase, dim=2))

            prev_layer_num = layer_num[zi]

        Ex = torch.cat(Ex_split, dim=1)
        Ey = torch.cat(Ey_split, dim=1)
        Ez = torch.cat(Ez_split, dim=1)
        Hx = torch.cat(Hx_split, dim=1)
        Hy = torch.cat(Hy_split, dim=1)
        Hz = torch.cat(Hz_split, dim=1)

        return [Ex, Ey, Ez], [Hx, Hy, Hz]

    @_precision_io
    def field_xy(self, layer_num, x_axis, y_axis, z_prop=0.0):
        """
        Calculate XY-plane electromagnetic field distribution at a selected layer.

        Returns the field at z_prop distance from the lower boundary of the layer.
        For the input layer (layer_num=-1), z_prop is the distance from the upper
        boundary and should be negative. If positive value is entered for input layer,
        z_prop=0 is used.

        Parameters
        ----------
        layer_num : int
            Selected layer index. Use -1 for input layer.
        x_axis : torch.Tensor
            x-direction sampling coordinates.
        y_axis : torch.Tensor
            y-direction sampling coordinates.
        z_prop : float, optional
            z-direction distance from the lower boundary of the layer (for layer_num>-1),
            or distance from the upper boundary (should be negative for layer_num=-1).
            For layer_num>-1, lower boundary (z=0) faces input layer and positive z
            moves away from input. Default is 0.0.

        Returns
        -------
        tuple
            ([Ex, Ey, Ez], [Hx, Hy, Hz]) where each component is a torch.Tensor.
        """

        self._require_scalar_freq("field_xy")

        if not isinstance(layer_num, int):
            raise TypeError('Parameter "layer_num" must be int type.')

        if layer_num < -1 or layer_num > self.layer_N:
            raise IndexError("Layer number is out of range.")

        if not isinstance(x_axis, torch.Tensor) or not isinstance(y_axis, torch.Tensor):
            raise TypeError("x and y axis must be torch.Tensor type.")

        # [x, y, diffraction order]
        x_axis = x_axis.reshape([-1, 1, 1])
        y_axis = y_axis.reshape([1, -1, 1])

        Kx_norm, Ky_norm = self.Kx_norm, self.Ky_norm

        # Input and output layers
        if layer_num == -1 or layer_num == self.layer_N:
            Kx_norm_dn, Ky_norm_dn = self.Kx_norm_dn, self.Ky_norm_dn

            if layer_num == -1:
                z_prop = z_prop if z_prop <= 0.0 else 0.0
                eps = self.eps_in if hasattr(self, "eps_in") else 1.0
                mu = self.mu_in if hasattr(self, "mu_in") else 1.0
                Vi = self.Vi if hasattr(self, "Vi") else self.Vf
                Kz_norm_dn = torch.sqrt(eps * mu - Kx_norm_dn**2 - Ky_norm_dn**2)
                Kz_norm_dn = torch.where(
                    torch.imag(Kz_norm_dn) > 0, torch.conj(Kz_norm_dn), Kz_norm_dn
                ).reshape([-1, 1])
            elif layer_num == self.layer_N:
                z_prop = z_prop if z_prop >= 0.0 else 0.0
                eps = self.eps_out if hasattr(self, "eps_in") else 1.0
                mu = self.mu_out if hasattr(self, "mu_in") else 1.0
                Vo = self.Vo if hasattr(self, "Vo") else self.Vf
                Kz_norm_dn = torch.sqrt(eps * mu - Kx_norm_dn**2 - Ky_norm_dn**2)
                Kz_norm_dn = torch.where(
                    torch.imag(Kz_norm_dn) < 0, torch.conj(Kz_norm_dn), Kz_norm_dn
                ).reshape([-1, 1])

            # Phase
            Kz_norm_dn = torch.vstack((Kz_norm_dn, Kz_norm_dn))
            z_phase = torch.exp(1.0j * self.omega * Kz_norm_dn * z_prop)

            # Fourier domain fields
            # [diffraction order, diffraction order]
            if layer_num == -1 and self.source_direction == "forward":
                Exy_p = self.E_i * z_phase
                Hxy_p = torch.matmul(Vi, Exy_p)
                Exy_m = torch.matmul(self.S[1], self.E_i) * torch.conj(z_phase)
                Hxy_m = torch.matmul(-Vi, Exy_m)
            elif layer_num == -1 and self.source_direction == "backward":
                Exy_p = torch.zeros_like(self.E_i)
                Hxy_p = torch.zeros_like(self.E_i)
                Exy_m = torch.matmul(self.S[3], self.E_i) * torch.conj(z_phase)
                Hxy_m = torch.matmul(-Vi, Exy_m)
            elif layer_num == self.layer_N and self.source_direction == "forward":
                Exy_p = torch.matmul(self.S[0], self.E_i) * z_phase
                Hxy_p = torch.matmul(Vo, Exy_p)
                Exy_m = torch.zeros_like(self.E_i)
                Hxy_m = torch.zeros_like(self.E_i)
            elif layer_num == self.layer_N and self.source_direction == "backward":
                Exy_p = torch.matmul(self.S[2], self.E_i) * z_phase
                Hxy_p = torch.matmul(Vo, Exy_p)
                Exy_m = self.E_i * torch.conj(z_phase)
                Hxy_m = torch.matmul(-Vo, Exy_m)

            Ex_mn = Exy_p[: self.order_N] + Exy_m[: self.order_N]
            Ey_mn = Exy_p[self.order_N :] + Exy_m[self.order_N :]
            Hz_mn = (
                torch.matmul(Kx_norm, Ey_mn) / mu - torch.matmul(Ky_norm, Ex_mn) / mu
            )
            Hx_mn = Hxy_p[: self.order_N] + Hxy_m[: self.order_N]
            Hy_mn = Hxy_p[self.order_N :] + Hxy_m[self.order_N :]
            Ez_mn = (
                torch.matmul(Ky_norm, Hx_mn) / eps - torch.matmul(Kx_norm, Hy_mn) / eps
            )

            # Spatial domain fields
            xy_phase = torch.exp(
                1.0j
                * self.omega
                * (self.Kx_norm_dn * x_axis + self.Ky_norm_dn * y_axis)
            )
            Ex = torch.sum(Ex_mn.reshape(1, 1, -1) * xy_phase, dim=2)
            Ey = torch.sum(Ey_mn.reshape(1, 1, -1) * xy_phase, dim=2)
            Ez = torch.sum(Ez_mn.reshape(1, 1, -1) * xy_phase, dim=2)
            Hx = torch.sum(Hx_mn.reshape(1, 1, -1) * xy_phase, dim=2)
            Hy = torch.sum(Hy_mn.reshape(1, 1, -1) * xy_phase, dim=2)
            Hz = torch.sum(Hz_mn.reshape(1, 1, -1) * xy_phase, dim=2)

        # Internal layers
        else:
            if self.source_direction == "forward":
                C = torch.matmul(self._d(self.C[0][layer_num]), self.E_i)
            elif self.source_direction == "backward":
                C = torch.matmul(self._d(self.C[1][layer_num]), self.E_i)

            kz_norm = self._d(self.kz_norm[layer_num])
            E_eigvec = self._d(self.E_eigvec[layer_num])
            H_eigvec = self._d(self.H_eigvec[layer_num])

            Cp = torch.diag(C[: 2 * self.order_N, 0])
            Cm = torch.diag(C[2 * self.order_N :, 0])

            eps_conv_inv = torch.linalg.inv(self._d(self.eps_conv[layer_num]))
            mu_conv_inv = torch.linalg.inv(self._d(self.mu_conv[layer_num]))

            # Phase
            z_phase_p = torch.diag(torch.exp(1.0j * self.omega * kz_norm * z_prop))
            z_phase_m = torch.diag(
                torch.exp(
                    1.0j * self.omega * kz_norm * (self.thickness[layer_num] - z_prop)
                )
            )

            # Fourier domain fields
            # [diffraction order, eigenmode number]
            Exy_p = torch.matmul(E_eigvec, z_phase_p)
            Ex_p = Exy_p[: self.order_N, :]
            Ey_p = Exy_p[self.order_N :, :]
            Hz_p = torch.matmul(
                mu_conv_inv, torch.matmul(Kx_norm, Ey_p)
            ) - torch.matmul(mu_conv_inv, torch.matmul(Ky_norm, Ex_p))
            Exy_m = torch.matmul(E_eigvec, z_phase_m)
            Ex_m = Exy_m[: self.order_N, :]
            Ey_m = Exy_m[self.order_N :, :]
            Hz_m = torch.matmul(
                mu_conv_inv, torch.matmul(Kx_norm, Ey_m)
            ) - torch.matmul(mu_conv_inv, torch.matmul(Ky_norm, Ex_m))
            Hxy_p = torch.matmul(H_eigvec, z_phase_p)
            Hx_p = Hxy_p[: self.order_N, :]
            Hy_p = Hxy_p[self.order_N :, :]
            Ez_p = torch.matmul(
                eps_conv_inv, torch.matmul(Ky_norm, Hx_p)
            ) - torch.matmul(eps_conv_inv, torch.matmul(Kx_norm, Hy_p))
            Hxy_m = torch.matmul(-H_eigvec, z_phase_m)
            Hx_m = Hxy_m[: self.order_N, :]
            Hy_m = Hxy_m[self.order_N :, :]
            Ez_m = torch.matmul(
                eps_conv_inv, torch.matmul(Ky_norm, Hx_m)
            ) - torch.matmul(eps_conv_inv, torch.matmul(Kx_norm, Hy_m))

            Ex_mn = torch.sum(torch.matmul(Ex_p, Cp) + torch.matmul(Ex_m, Cm), dim=1)
            Ey_mn = torch.sum(torch.matmul(Ey_p, Cp) + torch.matmul(Ey_m, Cm), dim=1)
            Ez_mn = torch.sum(torch.matmul(Ez_p, Cp) + torch.matmul(Ez_m, Cm), dim=1)
            Hx_mn = torch.sum(torch.matmul(Hx_p, Cp) + torch.matmul(Hx_m, Cm), dim=1)
            Hy_mn = torch.sum(torch.matmul(Hy_p, Cp) + torch.matmul(Hy_m, Cm), dim=1)
            Hz_mn = torch.sum(torch.matmul(Hz_p, Cp) + torch.matmul(Hz_m, Cm), dim=1)

            # Spatial domain fields
            xy_phase = torch.exp(
                1.0j
                * self.omega
                * (self.Kx_norm_dn * x_axis + self.Ky_norm_dn * y_axis)
            )
            Ex = torch.sum(Ex_mn.reshape(1, 1, -1) * xy_phase, dim=2)
            Ey = torch.sum(Ey_mn.reshape(1, 1, -1) * xy_phase, dim=2)
            Ez = torch.sum(Ez_mn.reshape(1, 1, -1) * xy_phase, dim=2)
            Hx = torch.sum(Hx_mn.reshape(1, 1, -1) * xy_phase, dim=2)
            Hy = torch.sum(Hy_mn.reshape(1, 1, -1) * xy_phase, dim=2)
            Hz = torch.sum(Hz_mn.reshape(1, 1, -1) * xy_phase, dim=2)

        return [Ex, Ey, Ez], [Hx, Hy, Hz]

    @_precision_io
    def poynting(self, E, H):
        """
        Compute the time-averaged Poynting vector for phasor fields.

        Uses the exp(-jωt) convention: <S> = 0.5 * Re(E × H*).

        Parameters
        ----------
        E : tuple of torch.Tensor
            Electric field components (Ex, Ey, Ez) on the same grid.
        H : tuple of torch.Tensor
            Magnetic field components (Hx, Hy, Hz) on the same grid with same shape as E.

        Returns
        -------
        tuple of torch.Tensor
            (Sx, Sy, Sz) Poynting vector components on the grid.
        """
        Ex, Ey, Ez = E
        Hx, Hy, Hz = H
        Sx = 0.5 * torch.real(Ey * Hz.conj() - Ez * Hy.conj())
        Sy = 0.5 * torch.real(Ez * Hx.conj() - Ex * Hz.conj())
        Sz = 0.5 * torch.real(Ex * Hy.conj() - Ey * Hx.conj())
        return (Sx, Sy, Sz)

    @_precision_io
    def poynting_xy(self, layer_num, x_axis, y_axis, z_prop=0.0):
        """
        Compute the Poynting vector on an XY plane inside a chosen layer.

        Convenience wrapper around field_xy that reconstructs E and H fields
        on an XY plane, then computes the Poynting vector. For layer absorption
        analysis, Sz is typically used by comparing values at z_prop=0 and z_prop=thickness.

        Parameters
        ----------
        layer_num : int
            Selected layer index. Use -1 for input layer.
        x_axis : torch.Tensor
            x-direction sampling coordinates.
        y_axis : torch.Tensor
            y-direction sampling coordinates.
        z_prop : float, optional
            z-direction distance from layer boundary (same convention as field_xy). Default is 0.0.

        Returns
        -------
        tuple of torch.Tensor
            (Sx, Sy, Sz) Poynting vector components on the (x, y) grid.
        """
        E, H = self.field_xy(layer_num, x_axis, y_axis, z_prop)
        return self.poynting(E, H)

    @_precision_io
    def poynting_flux(
        self,
        layer_num,
        x_points=None,
        y_points=None,
        z_prop=0.0,
        *,
        x_cells=None,
        y_cells=None,
    ):
        """
        Compute the Poynting flux through an XY plane inside a chosen layer.

        This thin wrapper delegates to ``solwa.utils.poynting_flux``.
        Coordinates can be supplied either as actual sampling points or as
        centers of finite cells. The two forms are mutually exclusive.

        Parameters
        ----------
        layer_num : int
            Index of the layer in which the XY plane is located. Use -1 for
            the input layer.
        x_points, y_points : array-like or torch.Tensor, optional
            Actual x/y sampling coordinates. The flux is integrated between
            these points with the trapezoidal rule. Both axes must be supplied
            together. Passing cell centers here (e.g. ``rcwa_geo.x``) emits a
            warning; use ``x_cells``/``y_cells`` for those.
        z_prop : float, optional
            z-position at which to evaluate the flux. Default is 0.0.
        x_cells, y_cells : array-like or torch.Tensor, optional
            Centers of adjacent, equally sized x/y cells. Each sample
            represents the full cell around its center and midpoint
            quadrature is used. Both axes must be supplied together.

        Returns
        -------
        torch.Tensor
            Integrated z-component of the Poynting vector.

        Raises
        ------
        ValueError
            If point and cell coordinates are mixed or an axis pair is
            incomplete.
        """
        from .utils import poynting_flux

        return poynting_flux(
            self,
            layer_num,
            x_points=x_points,
            y_points=y_points,
            z_prop=z_prop,
            x_cells=x_cells,
            y_cells=y_cells,
        )

    # Internal functions
    @staticmethod
    def _cast(obj, dtype):
        """Cast complex/floating tensors (also inside lists/tuples) to *dtype*
        or its real counterpart; other objects are returned unchanged."""
        if isinstance(obj, torch.Tensor):
            if obj.is_complex():
                return obj.to(dtype)
            if obj.is_floating_point():
                return obj.to(dtype.to_real())
            return obj
        if isinstance(obj, (list, tuple)):
            return type(obj)(rcwa._cast(o, dtype) for o in obj)
        return obj

    def _bc(self, x, n):
        """Append *n* singleton dims to a per-wavelength tensor so it broadcasts
        against [B, ...] data. Scalars and 0-d tensors pass unchanged, as do
        size-1 tensors unless ``freq`` itself is a (possibly length-1) batch."""
        if isinstance(x, torch.Tensor) and (
            x.numel() > 1 or (x.dim() > 0 and self.freq.dim() > 0)
        ):
            return x.reshape(x.shape + (1,) * n)
        return x

    def _require_scalar_freq(self, name):
        if self.freq.dim() > 0:
            raise NotImplementedError(
                f"{name} does not support batched frequencies yet. "
                "Run a separate simulation with a scalar freq for each wavelength."
            )

    def _d(self, tensor):
        """Return *tensor* on the compute device, loading from offload device if needed."""
        if self._offload_device is not None:
            return tensor.to(self._device)
        return tensor

    def _to_offload(self, tensor):
        """Move *tensor* to the offload device, using pinned memory for CPU offloading from CUDA."""
        d = self._offload_device
        if d.type == "cpu" and tensor.is_cuda and not tensor.requires_grad:
            pinned = torch.empty(
                tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True
            )
            pinned.copy_(tensor)
            return pinned
        return tensor.to(d, non_blocking=d.type == "cpu" and tensor.is_cuda)

    def _offload_layer_data(self):
        """Move the most recently added layer's tensors to the offload device."""
        self.P[-1] = self._to_offload(self.P[-1])
        self.Q[-1] = self._to_offload(self.Q[-1])
        self.eps_conv[-1] = self._to_offload(self.eps_conv[-1])
        self.mu_conv[-1] = self._to_offload(self.mu_conv[-1])
        self.kz_norm[-1] = self._to_offload(self.kz_norm[-1])
        self.E_eigvec[-1] = self._to_offload(self.E_eigvec[-1])
        self.H_eigvec[-1] = self._to_offload(self.H_eigvec[-1])
        self.Cf[-1] = self._to_offload(self.Cf[-1])
        self.Cb[-1] = self._to_offload(self.Cb[-1])
        self.layer_S11[-1] = self._to_offload(self.layer_S11[-1])
        self.layer_S21[-1] = self._to_offload(self.layer_S21[-1])
        self.layer_S12[-1] = self._to_offload(self.layer_S12[-1])
        self.layer_S22[-1] = self._to_offload(self.layer_S22[-1])

    def _matching_indices(self, orders):
        orders[orders[:, 0] < -self.order[0], 0] = int(-self.order[0])
        orders[orders[:, 0] > self.order[0], 0] = int(self.order[0])
        orders[orders[:, 1] < -self.order[1], 1] = int(-self.order[1])
        orders[orders[:, 1] > self.order[1], 1] = int(self.order[1])
        order_indices = (
            len(self.order_y) * (orders[:, 0] + int(self.order[0]))
            + orders[:, 1]
            + int(self.order[1])
        )

        return order_indices

    def _kvectors(self):
        if self.angle_layer == "input":
            self.kx0_norm = (
                torch.real(torch.sqrt(self.eps_in * self.mu_in))
                * torch.sin(self.inc_ang)
                * torch.cos(self.azi_ang)
            )
            self.ky0_norm = (
                torch.real(torch.sqrt(self.eps_in * self.mu_in))
                * torch.sin(self.inc_ang)
                * torch.sin(self.azi_ang)
            )
        else:
            self.kx0_norm = (
                torch.real(torch.sqrt(self.eps_out * self.mu_out))
                * torch.sin(self.inc_ang)
                * torch.cos(self.azi_ang)
            )
            self.ky0_norm = (
                torch.real(torch.sqrt(self.eps_out * self.mu_out))
                * torch.sin(self.inc_ang)
                * torch.sin(self.azi_ang)
            )

        # Free space k-vectors and E to H transformation matrix
        bc = self._bc
        self.kx_norm = bc(self.kx0_norm, 1) + self.order_x * bc(self.Gx_norm, 1)
        self.ky_norm = bc(self.ky0_norm, 1) + self.order_y * bc(self.Gy_norm, 1)

        kx_norm_grid, ky_norm_grid = torch.broadcast_tensors(
            self.kx_norm[..., :, None], self.ky_norm[..., None, :]
        )

        self.Kx_norm_dn = kx_norm_grid.flatten(-2)
        self.Ky_norm_dn = ky_norm_grid.flatten(-2)
        self.Kx_norm = torch.diag_embed(self.Kx_norm_dn)
        self.Ky_norm = torch.diag_embed(self.Ky_norm_dn)

        Kz_norm_dn = torch.sqrt(1.0 - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
        Kz_norm_dn = torch.where(
            torch.imag(Kz_norm_dn) < 0, torch.conj(Kz_norm_dn), Kz_norm_dn
        )
        tmp1 = _vcat(
            (
                torch.diag_embed(-self.Ky_norm_dn * self.Kx_norm_dn / Kz_norm_dn),
                torch.diag_embed(Kz_norm_dn + self.Kx_norm_dn**2 / Kz_norm_dn),
            )
        )
        tmp2 = _vcat(
            (
                torch.diag_embed(-Kz_norm_dn - self.Ky_norm_dn**2 / Kz_norm_dn),
                torch.diag_embed(self.Kx_norm_dn * self.Ky_norm_dn / Kz_norm_dn),
            )
        )
        self.Vf = _hcat((tmp1, tmp2))

        if hasattr(self, "Sin"):
            # Input layer k-vectors and E to H transformation matrix
            Kz_norm_dn = torch.sqrt(
                self._bc(self.eps_in * self.mu_in, 1)
                - self.Kx_norm_dn**2
                - self.Ky_norm_dn**2
            )
            Kz_norm_dn = torch.where(
                torch.imag(Kz_norm_dn) < 0, torch.conj(Kz_norm_dn), Kz_norm_dn
            )
            tmp1 = _vcat(
                (
                    torch.diag_embed(-self.Ky_norm_dn * self.Kx_norm_dn / Kz_norm_dn),
                    torch.diag_embed(Kz_norm_dn + self.Kx_norm_dn**2 / Kz_norm_dn),
                )
            )
            tmp2 = _vcat(
                (
                    torch.diag_embed(-Kz_norm_dn - self.Ky_norm_dn**2 / Kz_norm_dn),
                    torch.diag_embed(self.Kx_norm_dn * self.Ky_norm_dn / Kz_norm_dn),
                )
            )
            self.Vi = _hcat((tmp1, tmp2))

            Vtmp1 = torch.linalg.inv(self.Vf + self.Vi)
            Vtmp2 = self.Vf - self.Vi

            # Input layer S-matrix
            self.Sin.append(2 * torch.matmul(Vtmp1, self.Vi))  # Tf S11
            self.Sin.append(-torch.matmul(Vtmp1, Vtmp2))  # Rf S21
            self.Sin.append(torch.matmul(Vtmp1, Vtmp2))  # Rb S12
            self.Sin.append(2 * torch.matmul(Vtmp1, self.Vf))  # Tb S22

        if hasattr(self, "Sout"):
            # Output layer k-vectors and E to H transformation matrix
            Kz_norm_dn = torch.sqrt(
                self._bc(self.eps_out * self.mu_out, 1)
                - self.Kx_norm_dn**2
                - self.Ky_norm_dn**2
            )
            Kz_norm_dn = torch.where(
                torch.imag(Kz_norm_dn) < 0, torch.conj(Kz_norm_dn), Kz_norm_dn
            )
            tmp1 = _vcat(
                (
                    torch.diag_embed(-self.Ky_norm_dn * self.Kx_norm_dn / Kz_norm_dn),
                    torch.diag_embed(Kz_norm_dn + self.Kx_norm_dn**2 / Kz_norm_dn),
                )
            )
            tmp2 = _vcat(
                (
                    torch.diag_embed(-Kz_norm_dn - self.Ky_norm_dn**2 / Kz_norm_dn),
                    torch.diag_embed(self.Kx_norm_dn * self.Ky_norm_dn / Kz_norm_dn),
                )
            )
            self.Vo = _hcat((tmp1, tmp2))

            Vtmp1 = torch.linalg.inv(self.Vf + self.Vo)
            Vtmp2 = self.Vf - self.Vo

            # Output layer S-matrix
            self.Sout.append(2 * torch.matmul(Vtmp1, self.Vf))  # Tf S11
            self.Sout.append(torch.matmul(Vtmp1, Vtmp2))  # Rf S21
            self.Sout.append(-torch.matmul(Vtmp1, Vtmp2))  # Rb S12
            self.Sout.append(2 * torch.matmul(Vtmp1, self.Vo))  # Tb S22

    def _material_conv(self, material):
        material_N = material.shape[-2] * material.shape[-1]

        # Validate material grid size before any FFT or advanced indexing.
        # The convolution matrix indexes FFT coefficients by order differences,
        # which can reach ±(max_order - min_order) along each axis. The material
        # grid must be large enough to support these indices via wraparound.
        max_ix_range = 2 * int(self.order[0])
        max_iy_range = 2 * int(self.order[1])
        if material.shape[0] <= max_ix_range or material.shape[1] <= max_iy_range:
            req_nx = max_ix_range + 1
            req_ny = max_iy_range + 1
            raise ValueError(
                f"Inhomogeneous material grid is too small for the selected RCWA order. "
                f"For order_x=[{int(self.order_x.min())}, ..., {int(self.order_x.max())}] and "
                f"order_y=[{int(self.order_y.min())}, ..., {int(self.order_y.max())}], the convolution "
                f"matrix uses index differences up to ±{max_ix_range} (x) and ±{max_iy_range} (y), so "
                f"material.shape must be at least [{req_nx}, {req_ny}] (strictly greater than "
                f"[{max_ix_range}, {max_iy_range}]). Got {list(material.shape)}. Increase the material grid "
                f"size or reduce the RCWA order."
            )

        # Matching indices
        order_x_grid, order_y_grid = torch.meshgrid(
            self.order_x, self.order_y, indexing="ij"
        )
        ox = order_x_grid.to(torch.int64).reshape([-1])
        oy = order_y_grid.to(torch.int64).reshape([-1])

        ind = torch.arange(len(self.order_x) * len(self.order_y), device=self._device)
        indx, indy = torch.meshgrid(
            ind.to(torch.int64), ind.to(torch.int64), indexing="ij"
        )

        material_fft = torch.fft.fft2(material) / material_N

        material_fft_real = torch.real(material_fft)
        material_fft_imag = torch.imag(material_fft)

        material_convmat_real = material_fft_real[
            ..., ox[indx] - ox[indy], oy[indx] - oy[indy]
        ]
        material_convmat_imag = material_fft_imag[
            ..., ox[indx] - ox[indy], oy[indx] - oy[indy]
        ]

        material_convmat = torch.complex(material_convmat_real, material_convmat_imag)

        return material_convmat

    def _material_conv_li(self, material):
        """Li's convolution matrices (eps_xx, eps_yy) for the in-plane E components.

        eps_xx = [[ [[1/eps]]_x^-1 ]]_y : inverse rule along x, Laurent along y.
        eps_yy = [[ [[1/eps]]_y^-1 ]]_x : inverse rule along y, Laurent along x.
        Rows/columns are ordered like ``_material_conv`` (x order major, y minor).
        """
        nx, ny = material.shape[-2:]
        ox, oy = self.order_x, self.order_y
        inv_material = 1.0 / material.to(self._dtype)
        dx = ox[:, None] - ox[None, :]
        dy = oy[:, None] - oy[None, :]

        # eps_xx: per y-sample Toeplitz of 1/eps along x, inverted, then Laurent along y
        coef_x = torch.fft.fft(inv_material, dim=-2) / nx  # [..., nx, ny]
        a_x = torch.linalg.inv(coef_x.transpose(-2, -1)[..., dx])  # [..., ny, Mx, Mx]
        a_x = torch.fft.fft(a_x, dim=-3) / ny  # [..., q, Mx, Mx]
        eps_xx = a_x[..., dy, :, :].movedim(-2, -4).movedim(-1, -2)
        eps_xx = eps_xx.flatten(-4, -3).flatten(-2, -1)

        # eps_yy: per x-sample Toeplitz of 1/eps along y, inverted, then Laurent along x
        coef_y = torch.fft.fft(inv_material, dim=-1) / ny  # [..., nx, ny]
        a_y = torch.linalg.inv(coef_y[..., dy])  # [..., nx, My, My]
        a_y = torch.fft.fft(a_y, dim=-3) / nx  # [..., p, My, My]
        eps_yy = a_y[..., dx, :, :].transpose(-3, -2)
        eps_yy = eps_yy.flatten(-4, -3).flatten(-2, -1)

        return eps_xx, eps_yy

    def _eigen_decomposition_homogenous(self, eps, mu):
        # H to E transformation matirx
        self.P.append(
            _hcat(
                (
                    _vcat((torch.zeros_like(self.mu_conv[-1]), -self.mu_conv[-1])),
                    _vcat((self.mu_conv[-1], torch.zeros_like(self.mu_conv[-1]))),
                )
            )
            + 1
            / self._bc(eps, 2)
            * torch.matmul(
                _vcat((self.Kx_norm, self.Ky_norm)),
                _hcat((self.Ky_norm, -self.Kx_norm)),
            )
        )
        # E to H transformation matrix
        self.Q.append(
            _hcat(
                (
                    _vcat((torch.zeros_like(self.eps_conv[-1]), self.eps_conv[-1])),
                    _vcat((-self.eps_conv[-1], torch.zeros_like(self.eps_conv[-1]))),
                )
            )
            + 1
            / self._bc(mu, 2)
            * torch.matmul(
                _vcat((self.Kx_norm, self.Ky_norm)),
                _hcat((-self.Ky_norm, self.Kx_norm)),
            )
        )

        E_eigvec = torch.eye(
            self.P[-1].shape[-1], dtype=self._dtype, device=self._device
        )
        kz_norm = torch.sqrt(
            self._bc(eps * mu, 1) - self.Kx_norm_dn**2 - self.Ky_norm_dn**2
        )
        kz_norm = torch.where(
            torch.imag(kz_norm) < 0, torch.conj(kz_norm), kz_norm
        )  # Normalized kz for positive mode
        kz_norm = torch.cat((kz_norm, kz_norm), dim=-1)

        self.kz_norm.append(kz_norm)
        self.E_eigvec.append(E_eigvec)

    def _eigen_decomposition(self, eps_xx=None, eps_yy=None):
        # Q multiplies Ex with eps_xx and Ey with eps_yy (Laurent: both eps_conv)
        if eps_xx is None:
            eps_xx = eps_yy = self.eps_conv[-1]
        # H to E transformation matirx
        P_tmp = torch.matmul(
            _vcat((self.Kx_norm, self.Ky_norm)),
            torch.linalg.inv(self.eps_conv[-1]),
        )
        self.P.append(
            _hcat(
                (
                    _vcat((torch.zeros_like(self.mu_conv[-1]), -self.mu_conv[-1])),
                    _vcat((self.mu_conv[-1], torch.zeros_like(self.mu_conv[-1]))),
                )
            )
            + torch.matmul(P_tmp, _hcat((self.Ky_norm, -self.Kx_norm)))
        )
        # E to H transformation matrix
        Q_tmp = torch.matmul(
            _vcat((self.Kx_norm, self.Ky_norm)),
            torch.linalg.inv(self.mu_conv[-1]),
        )
        self.Q.append(
            _hcat(
                (
                    _vcat((torch.zeros_like(eps_xx), eps_xx)),
                    _vcat((-eps_yy, torch.zeros_like(eps_yy))),
                )
            )
            + torch.matmul(Q_tmp, _hcat((-self.Ky_norm, self.Kx_norm)))
        )

        # Eigen-decomposition
        if self._eig_dtype is None:
            PQ = torch.matmul(self.P[-1], self.Q[-1])
        else:
            # Form the product in the eig precision as well: a complex64 product
            # breaks exact mode degeneracies (e.g. uniform layers), which the
            # accurate eigensolver then resolves into near-parallel eigenvectors.
            PQ = torch.matmul(
                self.P[-1].to(self._eig_dtype), self.Q[-1].to(self._eig_dtype)
            )
        if self.stable_eig_grad is True:
            kz_norm, E_eigvec = Eig.apply(PQ)
        else:
            kz_norm, E_eigvec = torch.linalg.eig(PQ)

        kz_norm = torch.sqrt(kz_norm)
        self.kz_norm.append(
            torch.where(torch.imag(kz_norm) < 0, -kz_norm, kz_norm).to(self._dtype)
        )  # Normalized kz for positive mode
        self.E_eigvec.append(E_eigvec.to(self._dtype))

    def _solve_layer_smatrix(self):
        Kz_norm = torch.diag_embed(self.kz_norm[-1])
        phase = torch.diag_embed(
            torch.exp(
                1.0j * self._bc(self.omega, 1) * self.kz_norm[-1] * self.thickness[-1]
            )
        )

        Pinv_tmp, info = torch.linalg.inv_ex(self.P[-1])
        if torch.any(info != 0):
            self._raise_singular_P(info)
        if self.avoid_Pinv_instability:
            eye = torch.eye(
                self.P[-1].shape[-1], dtype=self.P[-1].dtype, device=self.P[-1].device
            )
            Pinv_ins_tmp1 = torch.max(
                torch.abs(torch.matmul(self.P[-1].detach(), Pinv_tmp.detach()) - eye)
            )
            Pinv_ins_tmp2 = torch.max(
                torch.abs(torch.matmul(Pinv_tmp.detach(), self.P[-1].detach()) - eye)
            )
            Qinv_ins = torch.max(
                torch.abs(
                    torch.matmul(
                        self.Q[-1].detach(), torch.linalg.inv(self.Q[-1].detach())
                    )
                    - eye
                )
            )

            self.Pinv_instability.append(torch.maximum(Pinv_ins_tmp1, Pinv_ins_tmp2))
            self.Qinv_instability.append(Qinv_ins)

            if self.Pinv_instability[-1] < self.max_Pinv_instability:
                self.H_eigvec.append(
                    torch.matmul(Pinv_tmp, torch.matmul(self.E_eigvec[-1], Kz_norm))
                )
            else:
                self.H_eigvec.append(
                    torch.matmul(
                        self.Q[-1],
                        torch.matmul(self.E_eigvec[-1], torch.linalg.inv(Kz_norm)),
                    )
                )
        else:
            self.H_eigvec.append(
                torch.matmul(Pinv_tmp, torch.matmul(self.E_eigvec[-1], Kz_norm))
            )

        Ctmp1 = _vcat(
            (
                self.E_eigvec[-1]
                + torch.matmul(torch.linalg.inv(self.Vf), self.H_eigvec[-1]),
                torch.matmul(
                    self.E_eigvec[-1]
                    - torch.matmul(torch.linalg.inv(self.Vf), self.H_eigvec[-1]),
                    phase,
                ),
            )
        )
        Ctmp2 = _vcat(
            (
                torch.matmul(
                    self.E_eigvec[-1]
                    - torch.matmul(torch.linalg.inv(self.Vf), self.H_eigvec[-1]),
                    phase,
                ),
                self.E_eigvec[-1]
                + torch.matmul(torch.linalg.inv(self.Vf), self.H_eigvec[-1]),
            )
        )
        Ctmp = _hcat((Ctmp1, Ctmp2))

        # Mode coupling coefficients
        self.Cf.append(
            torch.matmul(
                torch.linalg.inv(Ctmp),
                _vcat(
                    (
                        2
                        * torch.eye(
                            2 * self.order_N, dtype=self._dtype, device=self._device
                        ),
                        torch.zeros(
                            [2 * self.order_N, 2 * self.order_N],
                            dtype=self._dtype,
                            device=self._device,
                        ),
                    )
                ),
            )
        )
        self.Cb.append(
            torch.matmul(
                torch.linalg.inv(Ctmp),
                _vcat(
                    (
                        torch.zeros(
                            [2 * self.order_N, 2 * self.order_N],
                            dtype=self._dtype,
                            device=self._device,
                        ),
                        2
                        * torch.eye(
                            2 * self.order_N, dtype=self._dtype, device=self._device
                        ),
                    )
                ),
            )
        )

        self.layer_S11.append(
            torch.matmul(
                torch.matmul(self.E_eigvec[-1], phase),
                self.Cf[-1][..., : 2 * self.order_N, :],
            )
            + torch.matmul(self.E_eigvec[-1], self.Cf[-1][..., 2 * self.order_N :, :])
        )
        self.layer_S21.append(
            torch.matmul(self.E_eigvec[-1], self.Cf[-1][..., : 2 * self.order_N, :])
            + torch.matmul(
                torch.matmul(self.E_eigvec[-1], phase),
                self.Cf[-1][..., 2 * self.order_N :, :],
            )
            - torch.eye(2 * self.order_N, dtype=self._dtype, device=self._device)
        )
        self.layer_S12.append(
            torch.matmul(
                torch.matmul(self.E_eigvec[-1], phase),
                self.Cb[-1][..., : 2 * self.order_N, :],
            )
            + torch.matmul(self.E_eigvec[-1], self.Cb[-1][..., 2 * self.order_N :, :])
            - torch.eye(2 * self.order_N, dtype=self._dtype, device=self._device)
        )
        self.layer_S22.append(
            torch.matmul(self.E_eigvec[-1], self.Cb[-1][..., : 2 * self.order_N, :])
            + torch.matmul(
                torch.matmul(self.E_eigvec[-1], phase),
                self.Cb[-1][..., 2 * self.order_N :, :],
            )
        )

    def _raise_singular_P(self, info):
        # P (H to E) is singular when a mode has kz = 0 in this layer. In a
        # homogeneous layer the modes are the diffraction orders, so this is a
        # Rayleigh anomaly; patterned-layer eigenvalues are in solver order.
        bad = torch.nonzero(info != 0).flatten().tolist() or [0]
        wl = (1 / self.freq.real).reshape(-1).tolist()
        wl_txt = ", ".join(
            f"{wl[b]:g}" + (f" (batch index {b})" if self.freq.dim() > 0 else "")
            for b in bad
        )
        kz = torch.abs(self.kz_norm[-1]).reshape(-1, 2 * self.order_N)[bad[0]]
        if self._layer_is_homogeneous:
            i = int(torch.argmin(kz[: self.order_N]))
            ny = len(self.order_y)
            order = (int(self.order_x[i // ny]), int(self.order_y[i % ny]))
            where = f"min |kz| = {float(kz[i]):.2e} for order {order}"
            cause = (
                "that diffraction order lies exactly on a Rayleigh anomaly of "
                "this layer (kz = 0, e.g. n * period / wavelength is an integer "
                "at normal incidence)"
            )
        else:
            i = int(torch.argmin(kz))
            where = f"min |kz| = {float(kz[i]):.2e} for eigenmode {i}"
            cause = "an eigenmode of this patterned layer has kz close to 0"
        raise torch.linalg.LinAlgError(
            f"Layer {self.layer_N - 1}: the H-to-E matrix P is singular at "
            f"wavelength {wl_txt} ({where}). Likely cause: {cause}. Shift the "
            "wavelength or angle slightly, or give the layer permittivity a "
            "small imaginary part."
        )

    def _RS_prod(self, Sm, Sn, Cm, Cn):
        # S11 = S[0] / S21 = S[1] / S12 = S[2] / S22 = S[3]
        # Cf = C[0] / Cb = C[1]

        tmp1 = torch.linalg.inv(
            torch.eye(2 * self.order_N, dtype=self._dtype, device=self._device)
            - torch.matmul(Sm[2], Sn[1])
        )
        tmp2 = torch.linalg.inv(
            torch.eye(2 * self.order_N, dtype=self._dtype, device=self._device)
            - torch.matmul(Sn[1], Sm[2])
        )

        # Layer S-matrix
        S11 = torch.matmul(Sn[0], torch.matmul(tmp1, Sm[0]))
        S21 = Sm[1] + torch.matmul(
            Sm[3], torch.matmul(tmp2, torch.matmul(Sn[1], Sm[0]))
        )
        S12 = Sn[2] + torch.matmul(
            Sn[0], torch.matmul(tmp1, torch.matmul(Sm[2], Sn[3]))
        )
        S22 = torch.matmul(Sm[3], torch.matmul(tmp2, Sn[3]))

        # Mode coupling coefficients
        C = [[], []]
        for m in range(len(Cm[0])):
            C[0].append(
                Cm[0][m]
                + torch.matmul(Cm[1][m], torch.matmul(tmp2, torch.matmul(Sn[1], Sm[0])))
            )
            C[1].append(torch.matmul(Cm[1][m], torch.matmul(tmp2, Sn[3])))

        for n in range(len(Cn[0])):
            C[0].append(torch.matmul(Cn[0][n], torch.matmul(tmp1, Sm[0])))
            C[1].append(
                Cn[1][n]
                + torch.matmul(Cn[0][n], torch.matmul(tmp1, torch.matmul(Sm[2], Sn[3])))
            )

        return [S11, S21, S12, S22], C
