import torch
from math import pi
from .torch_eig import Eig


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
        symmetry_axis=None,
        symmetry_tolerance=1e-6,
    ):
        """
        Initialize Rigorous Coupled Wave Analysis (RCWA) simulation.

        Uses Lorentz-Heaviside units with speed of light = 1 and
        time harmonics notation exp(-jωt).

        Parameters
        ----------
        freq : float or torch.Tensor
            Simulation frequency (unit: length^-1).
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
        symmetry_axis : {None, "x", "y"}, optional
            Enable mirror-symmetry reduction about the selected, centered unit-cell
            axis. ``"x"`` means reflection across the x-axis (``y -> -y``), and
            ``"y"`` means reflection across the y-axis (``x -> -x``). The Bloch
            wave vector normal to the axis must be zero. Default is None.
        symmetry_tolerance : float, optional
            Absolute and relative tolerance used when validating symmetric material
            grids and compatible incidence. Default is 1e-6.
        """

        # Hardware
        if dtype != torch.complex64 and dtype != torch.complex128:
            raise ValueError("Invalid simulation data type")
        else:
            self._dtype = dtype
        self._device = device

        # Device offloading
        if offload_device is None:
            self._offload_device = None
        elif isinstance(offload_device, torch.device):
            self._offload_device = offload_device
        else:
            self._offload_device = torch.device(offload_device)

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

        # Simulation parameters
        self.freq = torch.as_tensor(
            freq, dtype=self._dtype, device=self._device
        )  # unit^-1
        self.omega = 2 * pi * freq  # same as k0a
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

        # Optional mirror-symmetry (parity) basis.  The basis separates all
        # tangential-field matrices into two independent blocks of size order_N.
        self._configure_symmetry(symmetry_axis, symmetry_tolerance)

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

        if self.symmetry_axis is not None:
            self._P_blocks, self._Q_blocks = [], []
            self._kz_blocks, self._E_eigvec_blocks, self._H_eigvec_blocks = [], [], []
            self._Cf_blocks, self._Cb_blocks = [], []
            self._layer_S_blocks = [[], [], [], []]

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
            isinstance(eps, float)
            or isinstance(eps, complex)
            or (eps.dim() == 0)
            or ((eps.dim() == 1) and eps.shape[0] == 1)
        )
        is_mu_homogenous = (
            isinstance(mu, float)
            or isinstance(mu, float)
            or (mu.dim() == 0)
            or ((mu.dim() == 1) and mu.shape[0] == 1)
        )

        if self.symmetry_axis is not None:
            if not is_eps_homogenous:
                self._set_symmetry_sampling(eps)
                self._validate_material_symmetry(eps, "eps")
            if not is_mu_homogenous:
                self._set_symmetry_sampling(mu)
                self._validate_material_symmetry(mu, "mu")

        self.eps_conv.append(
            eps * torch.eye(self.order_N, dtype=self._dtype, device=self._device)
            if is_eps_homogenous
            else self._material_conv(eps)
        )
        self.mu_conv.append(
            mu * torch.eye(self.order_N, dtype=self._dtype, device=self._device)
            if is_mu_homogenous
            else self._material_conv(mu)
        )

        self.layer_N += 1
        self.thickness.append(thickness)

        if is_eps_homogenous and is_mu_homogenous:
            self._eigen_decomposition_homogenous(eps, mu)
        else:
            self._eigen_decomposition()

        if self.symmetry_axis is None:
            self._solve_layer_smatrix()
        else:
            self._solve_layer_smatrix_symmetry()

        if self._offload_device is not None:
            self._offload_layer_data()

    # Solve simulation
    def solve_global_smatrix(self):
        """
        Solve the global scattering matrix (S-matrix) for the entire structure.

        Combines all layer scattering matrices into a global S-matrix using
        the recursive doubling algorithm.
        """

        if self.symmetry_axis is not None:
            self._solve_global_smatrix_symmetry()
            return

        # Initialization
        if self.layer_N > 0:
            S11 = self._d(self.layer_S11[0])
            S21 = self._d(self.layer_S21[0])
            S12 = self._d(self.layer_S12[0])
            S22 = self._d(self.layer_S22[0])
            C = [[self._d(self.Cf[0])], [self._d(self.Cb[0])]]
        else:
            S11 = torch.eye(2 * self.order_N, dtype=self._dtype, device=self._device)
            S21 = torch.zeros(2 * self.order_N, dtype=self._dtype, device=self._device)
            S12 = torch.zeros(2 * self.order_N, dtype=self._dtype, device=self._device)
            S22 = torch.eye(2 * self.order_N, dtype=self._dtype, device=self._device)
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
                [t.to(self._offload_device) for t in self.C[0]],
                [t.to(self._offload_device) for t in self.C[1]],
            ]

    # Returns
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

        kx_norm = self.Kx_norm_dn[order_indices]
        ky_norm = self.Ky_norm_dn[order_indices]
        Kt_norm_dn = torch.sqrt(kx_norm**2 + ky_norm**2)
        kz_norm = torch.sqrt(eps * mu - kx_norm**2 - ky_norm**2)
        inc_angle = torch.atan2(torch.real(Kt_norm_dn), torch.real(kz_norm))
        azi_angle = torch.atan2(torch.real(ky_norm), torch.real(kx_norm))

        if unit == "degree":
            inc_angle = (180.0 / pi) * inc_angle
            azi_angle = (180.0 / pi) * azi_angle

        return inc_angle, azi_angle

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
                    self.eps_in * self.mu_in - self.Kx_norm_dn**2 - self.Ky_norm_dn**2
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
                    self.eps_out * self.mu_out - self.Kx_norm_dn**2 - self.Ky_norm_dn**2
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

                normalization = torch.sqrt(
                    (
                        1
                        + (numerator_pol[order_indices] / numerator_kz[order_indices])
                        ** 2
                    )
                    / (
                        1
                        + (
                            denominator_pol[ref_order_index]
                            / denominator_kz[ref_order_index]
                        )
                        ** 2
                    )
                )
                normalization = normalization * torch.sqrt(
                    numerator_kz[order_indices] / denominator_kz[ref_order_index]
                )
            else:
                normalization = 1.0

            # Get S-parameters
            if direction == "forward" and port == "transmission":
                S = self.S[0][order_indices, ref_order_index] * normalization
            elif direction == "forward" and port == "reflection":
                S = self.S[1][order_indices, ref_order_index] * normalization
            elif direction == "backward" and port == "reflection":
                S = self.S[2][order_indices, ref_order_index] * normalization
            elif direction == "backward" and port == "transmission":
                S = self.S[3][order_indices, ref_order_index] * normalization

            S = torch.where(torch.isinf(S), torch.zeros_like(S), S)
            S = torch.where(torch.isnan(S), torch.zeros_like(S), S)

            return S

        elif polarization in ["pp", "sp", "ps", "ss"]:
            if direction == "forward" and port == "transmission":
                idx = 0
                order_sign, ref_sign = 1, 1
                order_k0_norm2 = self.eps_out * self.mu_out
                ref_k0_norm2 = self.eps_in * self.mu_in
            elif direction == "forward" and port == "reflection":
                idx = 1
                order_sign, ref_sign = -1, 1
                order_k0_norm2 = self.eps_in * self.mu_in
                ref_k0_norm2 = self.eps_in * self.mu_in
            elif direction == "backward" and port == "reflection":
                idx = 2
                order_sign, ref_sign = 1, -1
                order_k0_norm2 = self.eps_out * self.mu_out
                ref_k0_norm2 = self.eps_out * self.mu_out
            elif direction == "backward" and port == "transmission":
                idx = 3
                order_sign, ref_sign = -1, -1
                order_k0_norm2 = self.eps_in * self.mu_in
                ref_k0_norm2 = self.eps_out * self.mu_out

            order_Kx_norm_dn = self.Kx_norm_dn[order_indices]
            order_Ky_norm_dn = self.Ky_norm_dn[order_indices]
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

            ref_Kx_norm_dn = self.Kx_norm_dn[ref_order_index]
            ref_Ky_norm_dn = self.Ky_norm_dn[ref_order_index]
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

            xx = self.S[idx][order_indices, ref_order_index]
            xy = self.S[idx][order_indices, ref_order_index + self.order_N]
            yx = self.S[idx][order_indices + self.order_N, ref_order_index]
            yy = self.S[idx][
                order_indices + self.order_N, ref_order_index + self.order_N
            ]

            xx = torch.where(order_is_evanescent, torch.zeros_like(xx), xx)
            xy = torch.where(order_is_evanescent, torch.zeros_like(xy), xy)
            yx = torch.where(order_is_evanescent, torch.zeros_like(yx), yx)
            yy = torch.where(order_is_evanescent, torch.zeros_like(yy), yy)

            if ref_is_evanescent:
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
                    self.eps_in * self.mu_in - self.Kx_norm_dn**2 - self.Ky_norm_dn**2
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
                    self.eps_out * self.mu_out - self.Kx_norm_dn**2 - self.Ky_norm_dn**2
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

                normalization = torch.sqrt(
                    numerator_kz[order_indices] / denominator_kz[ref_order_index]
                )
            else:
                normalization = 1.0

            S = torch.where(torch.isinf(S), torch.zeros_like(S), S)
            S = torch.where(torch.isnan(S), torch.zeros_like(S), S)

            return S * normalization

        else:
            return None

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
                    torch.sqrt(eps * mu - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
                )
            )

            inc_angle = torch.atan2(torch.real(Kt_norm_dn), Kz_norm_dn)
            azi_angle = torch.atan2(
                torch.real(self.Ky_norm_dn), torch.real(self.Kx_norm_dn)
            )

            tmp1 = torch.vstack(
                (
                    torch.diag(torch.cos(inc_angle) * torch.cos(azi_angle)),
                    torch.diag(torch.cos(inc_angle) * torch.sin(azi_angle)),
                )
            )
            tmp2 = torch.vstack(
                (torch.diag(-torch.sin(azi_angle)), torch.diag(torch.cos(azi_angle)))
            )
            ps2xy = torch.hstack((tmp1, tmp2))

            E_i = torch.matmul(ps2xy.to(self._dtype), E_i)

        self.E_i = E_i

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

        self._materialize_symmetry_data()
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

        self._materialize_symmetry_data()
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

        if not isinstance(layer_num, int):
            raise TypeError('Parameter "layer_num" must be int type.')

        if layer_num < -1 or layer_num > self.layer_N:
            raise IndexError("Layer number is out of range.")

        self._materialize_symmetry_data()
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

    def poynting_flux(self, layer_num, x_axis, y_axis, z_prop=0.0):
        """
        Hint:
        Computes the Poynting flux through an XY plane inside the chosen layer.

        This is a higher-level convenience wrapper that delegates to
        ``solwa.utils.poynting_flux``. It typically:
            1) Reconstructs the electromagnetic fields on an (x, y) grid inside
               ``layer_num`` at the relative position ``z_prop`` (0 at the
               layer entrance, 1 at the layer exit), using the same conventions
               as :meth:`field_xy`.
            2) Evaluates the time-averaged Poynting vector on that plane.
            3) Aggregates the result to obtain a flux / power quantity.

        Parameters
        ----------
        layer_num : int
            Index of the layer in which the XY plane is located.
        x_axis, y_axis : 1D array-like
            Sample points along the x and y directions (same convention as
            :meth:`field_xy` and :meth:`poynting_xy`).
        z_prop : float, optional
            Normalized position within the layer (0 at the entrance interface,
            1 at the exit interface). Default is 0.0.

        Returns
        -------
        Any
            The Poynting-flux-related quantity as defined by
            ``solwa.utils.poynting_flux`` (see that function for details).
        """
        from .utils import poynting_flux

        return poynting_flux(self, layer_num, x_axis, y_axis, z_prop)

    # Internal functions
    def _materialize_symmetry_data(self):
        """Lazily reconstruct full-basis data needed only by field APIs."""
        if self.symmetry_axis is None:
            return

        destination = self._offload_device or self._device
        public_layer_lists = (
            self.layer_S11,
            self.layer_S21,
            self.layer_S12,
            self.layer_S22,
        )
        for layer in range(self.layer_N):
            if self.E_eigvec[layer] is not None:
                continue
            e_blocks = [self._d(block) for block in self._E_eigvec_blocks[layer]]
            h_blocks = [self._d(block) for block in self._H_eigvec_blocks[layer]]
            cf_blocks = [self._d(block) for block in self._Cf_blocks[layer]]
            cb_blocks = [self._d(block) for block in self._Cb_blocks[layer]]
            self.E_eigvec[layer] = (self._Te @ torch.block_diag(*e_blocks)).to(
                destination
            )
            self.H_eigvec[layer] = (self._Th @ torch.block_diag(*h_blocks)).to(
                destination
            )
            self.Cf[layer] = self._assemble_coupling_blocks(cf_blocks).to(destination)
            self.Cb[layer] = self._assemble_coupling_blocks(cb_blocks).to(destination)
            for index, public in enumerate(public_layer_lists):
                blocks = [
                    self._d(block) for block in self._layer_S_blocks[index][layer]
                ]
                public[layer] = self._assemble_boundary_blocks(blocks).to(destination)

        if hasattr(self, "_global_C_blocks") and not self.C[0]:
            for direction in range(2):
                for layer in range(len(self._global_C_blocks[0][direction])):
                    blocks = [
                        self._d(self._global_C_blocks[parity][direction][layer])
                        for parity in range(2)
                    ]
                    self.C[direction].append(
                        self._assemble_coupling_blocks(blocks).to(destination)
                    )

    def _configure_symmetry(self, symmetry_axis, tolerance):
        """Create electric- and magnetic-field parity bases for a mirror axis."""
        if symmetry_axis is None:
            self.symmetry_axis = None
            self.symmetry_tolerance = float(tolerance)
            return

        axis = str(symmetry_axis).lower()
        if axis not in ("x", "y"):
            raise ValueError("symmetry_axis must be None, 'x', or 'y'")
        if tolerance <= 0:
            raise ValueError("symmetry_tolerance must be positive")

        self.symmetry_axis = axis
        self.symmetry_tolerance = float(tolerance)

        nx_order, ny_order = len(self.order_x), len(self.order_y)
        harmonic_indices = torch.arange(
            self.order_N, dtype=torch.int64, device=self._device
        ).reshape(nx_order, ny_order)
        mirror_dim = 1 if axis == "x" else 0
        mirror = torch.flip(harmonic_indices, dims=(mirror_dim,)).reshape(-1)

        # Tangential polar-vector (E) and axial-vector (H) signs under reflection.
        if axis == "x":
            e_signs, h_signs = (1.0, -1.0), (-1.0, 1.0)
        else:
            e_signs, h_signs = (-1.0, 1.0), (1.0, -1.0)

        self._symmetry_mirror = mirror
        self._symmetry_e_signs = e_signs
        self._symmetry_h_signs = h_signs
        self._symmetry_sample_count = None
        self._Te, e_sizes = self._parity_basis(mirror, e_signs)
        self._Th, h_sizes = self._parity_basis(mirror, h_signs)
        if e_sizes != h_sizes:
            raise RuntimeError("Electric and magnetic symmetry block sizes differ")
        self._symmetry_block_sizes = e_sizes
        split = e_sizes[0]
        self._symmetry_slices = (slice(0, split), slice(split, sum(e_sizes)))

    def _parity_basis(self, mirror, component_signs, harmonic_phases=None):
        """Return a unitary basis ordered by even then odd mirror parity."""
        mirror_cpu = mirror.detach().cpu().tolist()
        if harmonic_phases is None:
            harmonic_phases = torch.ones(
                self.order_N, dtype=self._dtype, device=self._device
            )
        phases_cpu = harmonic_phases.detach().cpu().tolist()
        columns = {1: [], -1: []}
        root_two = 2.0**0.5

        for component, sign in enumerate(component_signs):
            visited = set()
            offset = component * self.order_N
            for i, j in enumerate(mirror_cpu):
                if i in visited:
                    continue
                visited.add(i)
                visited.add(j)
                if i == j:
                    columns[int(sign)].append(((offset + i, 1.0),))
                else:
                    signed_phase = sign * phases_cpu[i]
                    columns[1].append(
                        (
                            (offset + i, 1.0 / root_two),
                            (offset + j, signed_phase / root_two),
                        )
                    )
                    columns[-1].append(
                        (
                            (offset + i, 1.0 / root_two),
                            (offset + j, -signed_phase / root_two),
                        )
                    )

        descriptors = columns[1] + columns[-1]
        basis = torch.zeros(
            (2 * self.order_N, 2 * self.order_N),
            dtype=self._dtype,
            device=self._device,
        )
        for column, entries in enumerate(descriptors):
            for row, value in entries:
                basis[row, column] = value
        return basis, (len(columns[1]), len(columns[-1]))

    def _set_symmetry_sampling(self, material):
        """Match the parity phase to the half-cell-centered sampled material grid."""
        sample_count = material.shape[1 if self.symmetry_axis == "x" else 0]
        if self._symmetry_sample_count is not None:
            if sample_count != self._symmetry_sample_count:
                raise ValueError(
                    "all patterned layers must use the same sampling count along "
                    f"the coordinate normal to symmetry_axis='{self.symmetry_axis}'"
                )
            return
        if self.layer_N:
            raise ValueError(
                "when symmetry_axis is enabled, add a patterned layer before any "
                "homogeneous internal layers so the sampled mirror phase can be inferred"
            )

        order_x_grid, order_y_grid = torch.meshgrid(
            self.order_x, self.order_y, indexing="ij"
        )
        normal_order = order_y_grid if self.symmetry_axis == "x" else order_x_grid
        phases = torch.exp(
            -2.0j * pi * normal_order.reshape(-1).to(self._dtype) / sample_count
        )
        self._Te, e_sizes = self._parity_basis(
            self._symmetry_mirror, self._symmetry_e_signs, phases
        )
        self._Th, h_sizes = self._parity_basis(
            self._symmetry_mirror, self._symmetry_h_signs, phases
        )
        if e_sizes != h_sizes:
            raise RuntimeError("Electric and magnetic symmetry block sizes differ")
        self._symmetry_sample_count = sample_count
        self._refresh_symmetry_boundary_blocks()

    def _refresh_symmetry_boundary_blocks(self):
        """Rebuild transformed exterior operators after selecting an axis phase."""
        if not hasattr(self, "Vf"):
            return
        vf_transformed = self._Th.mH @ self.Vf @ self._Te
        self._Vf_blocks = self._split_symmetry_matrix(vf_transformed)
        if hasattr(self, "Sin"):
            self._Sin_blocks = [
                self._split_symmetry_matrix(self._Te.mH @ matrix @ self._Te)
                for matrix in self.Sin
            ]
        if hasattr(self, "Sout"):
            self._Sout_blocks = [
                self._split_symmetry_matrix(self._Te.mH @ matrix @ self._Te)
                for matrix in self.Sout
            ]

    def _validate_material_symmetry(self, material, name):
        """Reject a layer grid that does not match the requested centered mirror."""
        if not isinstance(material, torch.Tensor) or material.dim() != 2:
            raise ValueError(
                f"{name} must be a 2-D tensor when symmetry_axis is enabled"
            )
        mirror_dim = 1 if self.symmetry_axis == "x" else 0
        reflected = torch.flip(material, dims=(mirror_dim,))
        if not torch.allclose(
            material.detach(),
            reflected.detach(),
            rtol=self.symmetry_tolerance,
            atol=self.symmetry_tolerance,
        ):
            coordinate = "y" if self.symmetry_axis == "x" else "x"
            raise ValueError(
                f"{name} is not mirror symmetric for symmetry_axis="
                f"'{self.symmetry_axis}' ({coordinate} -> -{coordinate})"
            )

    def _split_symmetry_matrix(self, matrix):
        """Extract the two diagonal parity blocks from a transformed matrix."""
        return [matrix[s, s] for s in self._symmetry_slices]

    def _assemble_boundary_blocks(self, blocks):
        """Transform a parity-block boundary operator back to the Fourier basis."""
        transformed = torch.block_diag(*blocks)
        return torch.matmul(self._Te, torch.matmul(transformed, self._Te.mH))

    def _assemble_coupling_blocks(self, blocks):
        """Assemble modal coefficients and map physical boundary fields to parity."""
        dimension = 2 * self.order_N
        transformed = torch.zeros(
            (2 * dimension, dimension), dtype=self._dtype, device=self._device
        )
        for block, target in zip(blocks, self._symmetry_slices):
            size = target.stop - target.start
            transformed[target, target] = block[:size]
            lower = slice(dimension + target.start, dimension + target.stop)
            transformed[lower, target] = block[size:]
        return torch.matmul(transformed, self._Te.mH)

    def _d(self, tensor):
        """Return *tensor* on the compute device, loading from offload device if needed."""
        if self._offload_device is not None:
            return tensor.to(self._device)
        return tensor

    def _offload_layer_data(self):
        """Move the most recently added layer's tensors to the offload device."""
        d = self._offload_device
        self.P[-1] = self.P[-1].to(d)
        self.Q[-1] = self.Q[-1].to(d)
        self.eps_conv[-1] = self.eps_conv[-1].to(d)
        self.mu_conv[-1] = self.mu_conv[-1].to(d)
        self.kz_norm[-1] = self.kz_norm[-1].to(d)
        if self.E_eigvec[-1] is not None:
            self.E_eigvec[-1] = self.E_eigvec[-1].to(d)
            self.H_eigvec[-1] = self.H_eigvec[-1].to(d)
            self.Cf[-1] = self.Cf[-1].to(d)
            self.Cb[-1] = self.Cb[-1].to(d)
            self.layer_S11[-1] = self.layer_S11[-1].to(d)
            self.layer_S21[-1] = self.layer_S21[-1].to(d)
            self.layer_S12[-1] = self.layer_S12[-1].to(d)
            self.layer_S22[-1] = self.layer_S22[-1].to(d)
        if self.symmetry_axis is not None:
            self._P_blocks[-1] = [block.to(d) for block in self._P_blocks[-1]]
            self._Q_blocks[-1] = [block.to(d) for block in self._Q_blocks[-1]]
            self._kz_blocks[-1] = [block.to(d) for block in self._kz_blocks[-1]]
            self._E_eigvec_blocks[-1] = [
                block.to(d) for block in self._E_eigvec_blocks[-1]
            ]
            self._H_eigvec_blocks[-1] = [
                block.to(d) for block in self._H_eigvec_blocks[-1]
            ]
            self._Cf_blocks[-1] = [block.to(d) for block in self._Cf_blocks[-1]]
            self._Cb_blocks[-1] = [block.to(d) for block in self._Cb_blocks[-1]]
            for matrices in self._layer_S_blocks:
                matrices[-1] = [block.to(d) for block in matrices[-1]]

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

        if self.symmetry_axis is not None:
            normal_component = (
                self.ky0_norm if self.symmetry_axis == "x" else self.kx0_norm
            )
            if torch.any(
                torch.abs(normal_component.detach()) > self.symmetry_tolerance
            ):
                component = "ky" if self.symmetry_axis == "x" else "kx"
                raise ValueError(
                    f"symmetry_axis='{self.symmetry_axis}' requires {component}0=0; "
                    "the incident Bloch wave vector breaks the requested mirror symmetry"
                )

        # Free space k-vectors and E to H transformation matrix
        self.kx_norm = self.kx0_norm + self.order_x * self.Gx_norm
        self.ky_norm = self.ky0_norm + self.order_y * self.Gy_norm

        kx_norm_grid, ky_norm_grid = torch.meshgrid(
            self.kx_norm, self.ky_norm, indexing="ij"
        )

        self.Kx_norm_dn = torch.reshape(kx_norm_grid, (-1,))
        self.Ky_norm_dn = torch.reshape(ky_norm_grid, (-1,))
        self.Kx_norm = torch.diag(self.Kx_norm_dn)
        self.Ky_norm = torch.diag(self.Ky_norm_dn)

        Kz_norm_dn = torch.sqrt(1.0 - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
        Kz_norm_dn = torch.where(
            torch.imag(Kz_norm_dn) < 0, torch.conj(Kz_norm_dn), Kz_norm_dn
        )
        tmp1 = torch.vstack(
            (
                torch.diag(-self.Ky_norm_dn * self.Kx_norm_dn / Kz_norm_dn),
                torch.diag(Kz_norm_dn + self.Kx_norm_dn**2 / Kz_norm_dn),
            )
        )
        tmp2 = torch.vstack(
            (
                torch.diag(-Kz_norm_dn - self.Ky_norm_dn**2 / Kz_norm_dn),
                torch.diag(self.Kx_norm_dn * self.Ky_norm_dn / Kz_norm_dn),
            )
        )
        self.Vf = torch.hstack((tmp1, tmp2))

        if hasattr(self, "Sin"):
            # Input layer k-vectors and E to H transformation matrix
            Kz_norm_dn = torch.sqrt(
                self.eps_in * self.mu_in - self.Kx_norm_dn**2 - self.Ky_norm_dn**2
            )
            Kz_norm_dn = torch.where(
                torch.imag(Kz_norm_dn) < 0, torch.conj(Kz_norm_dn), Kz_norm_dn
            )
            tmp1 = torch.vstack(
                (
                    torch.diag(-self.Ky_norm_dn * self.Kx_norm_dn / Kz_norm_dn),
                    torch.diag(Kz_norm_dn + self.Kx_norm_dn**2 / Kz_norm_dn),
                )
            )
            tmp2 = torch.vstack(
                (
                    torch.diag(-Kz_norm_dn - self.Ky_norm_dn**2 / Kz_norm_dn),
                    torch.diag(self.Kx_norm_dn * self.Ky_norm_dn / Kz_norm_dn),
                )
            )
            self.Vi = torch.hstack((tmp1, tmp2))

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
                self.eps_out * self.mu_out - self.Kx_norm_dn**2 - self.Ky_norm_dn**2
            )
            Kz_norm_dn = torch.where(
                torch.imag(Kz_norm_dn) < 0, torch.conj(Kz_norm_dn), Kz_norm_dn
            )
            tmp1 = torch.vstack(
                (
                    torch.diag(-self.Ky_norm_dn * self.Kx_norm_dn / Kz_norm_dn),
                    torch.diag(Kz_norm_dn + self.Kx_norm_dn**2 / Kz_norm_dn),
                )
            )
            tmp2 = torch.vstack(
                (
                    torch.diag(-Kz_norm_dn - self.Ky_norm_dn**2 / Kz_norm_dn),
                    torch.diag(self.Kx_norm_dn * self.Ky_norm_dn / Kz_norm_dn),
                )
            )
            self.Vo = torch.hstack((tmp1, tmp2))

            Vtmp1 = torch.linalg.inv(self.Vf + self.Vo)
            Vtmp2 = self.Vf - self.Vo

            # Output layer S-matrix
            self.Sout.append(2 * torch.matmul(Vtmp1, self.Vf))  # Tf S11
            self.Sout.append(torch.matmul(Vtmp1, Vtmp2))  # Rf S21
            self.Sout.append(-torch.matmul(Vtmp1, Vtmp2))  # Rb S12
            self.Sout.append(2 * torch.matmul(Vtmp1, self.Vo))  # Tb S22

        if self.symmetry_axis is not None:
            vf_transformed = torch.matmul(
                self._Th.mH, torch.matmul(self.Vf, self._Te)
            )
            self._Vf_blocks = self._split_symmetry_matrix(vf_transformed)
            if hasattr(self, "Sin"):
                self._Sin_blocks = [
                    self._split_symmetry_matrix(
                        torch.matmul(self._Te.mH, torch.matmul(matrix, self._Te))
                    )
                    for matrix in self.Sin
                ]
            if hasattr(self, "Sout"):
                self._Sout_blocks = [
                    self._split_symmetry_matrix(
                        torch.matmul(self._Te.mH, torch.matmul(matrix, self._Te))
                    )
                    for matrix in self.Sout
                ]

    def _material_conv(self, material):
        material_N = material.shape[0] * material.shape[1]

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
            ox[indx] - ox[indy], oy[indx] - oy[indy]
        ]
        material_convmat_imag = material_fft_imag[
            ox[indx] - ox[indy], oy[indx] - oy[indy]
        ]

        material_convmat = torch.complex(material_convmat_real, material_convmat_imag)

        return material_convmat

    def _eigen_decomposition_homogenous(self, eps, mu):
        # H to E transformation matirx
        self.P.append(
            torch.hstack(
                (
                    torch.vstack(
                        (torch.zeros_like(self.mu_conv[-1]), -self.mu_conv[-1])
                    ),
                    torch.vstack(
                        (self.mu_conv[-1], torch.zeros_like(self.mu_conv[-1]))
                    ),
                )
            )
            + 1
            / eps
            * torch.matmul(
                torch.vstack((self.Kx_norm, self.Ky_norm)),
                torch.hstack((self.Ky_norm, -self.Kx_norm)),
            )
        )
        # E to H transformation matrix
        self.Q.append(
            torch.hstack(
                (
                    torch.vstack(
                        (torch.zeros_like(self.eps_conv[-1]), self.eps_conv[-1])
                    ),
                    torch.vstack(
                        (-self.eps_conv[-1], torch.zeros_like(self.eps_conv[-1]))
                    ),
                )
            )
            + 1
            / mu
            * torch.matmul(
                torch.vstack((self.Kx_norm, self.Ky_norm)),
                torch.hstack((-self.Ky_norm, self.Kx_norm)),
            )
        )

        kz_norm = torch.sqrt(eps * mu - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
        kz_norm = torch.where(
            torch.imag(kz_norm) < 0, torch.conj(kz_norm), kz_norm
        )  # Normalized kz for positive mode
        kz_norm = torch.cat((kz_norm, kz_norm))

        if self.symmetry_axis is not None:
            p_transformed = torch.matmul(
                self._Te.mH, torch.matmul(self.P[-1], self._Th)
            )
            q_transformed = torch.matmul(
                self._Th.mH, torch.matmul(self.Q[-1], self._Te)
            )
            p_blocks = self._split_symmetry_matrix(p_transformed)
            q_blocks = self._split_symmetry_matrix(q_transformed)
            kz_transformed = torch.matmul(
                self._Te.mH, torch.matmul(torch.diag(kz_norm), self._Te)
            )
            kz_blocks = [
                torch.diagonal(kz_transformed[s, s]) for s in self._symmetry_slices
            ]
            e_blocks = [
                torch.eye(size, dtype=self._dtype, device=self._device)
                for size in self._symmetry_block_sizes
            ]
            self._P_blocks.append(p_blocks)
            self._Q_blocks.append(q_blocks)
            self._kz_blocks.append(kz_blocks)
            self._E_eigvec_blocks.append(e_blocks)
            self.kz_norm.append(torch.cat(kz_blocks))
            self.E_eigvec.append(None)
            return

        E_eigvec = torch.eye(
            self.P[-1].shape[-1], dtype=self._dtype, device=self._device
        )

        self.kz_norm.append(kz_norm)
        self.E_eigvec.append(E_eigvec)

    def _eigen_decomposition(self):
        # H to E transformation matirx
        P_tmp = torch.matmul(
            torch.vstack((self.Kx_norm, self.Ky_norm)),
            torch.linalg.inv(self.eps_conv[-1]),
        )
        self.P.append(
            torch.hstack(
                (
                    torch.vstack(
                        (torch.zeros_like(self.mu_conv[-1]), -self.mu_conv[-1])
                    ),
                    torch.vstack(
                        (self.mu_conv[-1], torch.zeros_like(self.mu_conv[-1]))
                    ),
                )
            )
            + torch.matmul(P_tmp, torch.hstack((self.Ky_norm, -self.Kx_norm)))
        )
        # E to H transformation matrix
        Q_tmp = torch.matmul(
            torch.vstack((self.Kx_norm, self.Ky_norm)),
            torch.linalg.inv(self.mu_conv[-1]),
        )
        self.Q.append(
            torch.hstack(
                (
                    torch.vstack(
                        (torch.zeros_like(self.eps_conv[-1]), self.eps_conv[-1])
                    ),
                    torch.vstack(
                        (-self.eps_conv[-1], torch.zeros_like(self.eps_conv[-1]))
                    ),
                )
            )
            + torch.matmul(Q_tmp, torch.hstack((-self.Ky_norm, self.Kx_norm)))
        )

        # Eigen-decomposition
        if self.symmetry_axis is not None:
            p_transformed = torch.matmul(
                self._Te.mH, torch.matmul(self.P[-1], self._Th)
            )
            q_transformed = torch.matmul(
                self._Th.mH, torch.matmul(self.Q[-1], self._Te)
            )
            p_blocks = self._split_symmetry_matrix(p_transformed)
            q_blocks = self._split_symmetry_matrix(q_transformed)
            kz_blocks, e_blocks = [], []
            for p_block, q_block in zip(p_blocks, q_blocks):
                operator = torch.matmul(p_block, q_block)
                if self.stable_eig_grad is True:
                    kz_block, e_block = Eig.apply(operator)
                else:
                    kz_block, e_block = torch.linalg.eig(operator)
                kz_block = torch.sqrt(kz_block)
                kz_blocks.append(
                    torch.where(torch.imag(kz_block) < 0, -kz_block, kz_block)
                )
                e_blocks.append(e_block)

            self._P_blocks.append(p_blocks)
            self._Q_blocks.append(q_blocks)
            self._kz_blocks.append(kz_blocks)
            self._E_eigvec_blocks.append(e_blocks)
            self.kz_norm.append(torch.cat(kz_blocks))
            self.E_eigvec.append(None)
            return

        if self.stable_eig_grad is True:
            kz_norm, E_eigvec = Eig.apply(torch.matmul(self.P[-1], self.Q[-1]))
        else:
            kz_norm, E_eigvec = torch.linalg.eig(torch.matmul(self.P[-1], self.Q[-1]))

        kz_norm = torch.sqrt(kz_norm)
        self.kz_norm.append(
            torch.where(torch.imag(kz_norm) < 0, -kz_norm, kz_norm)
        )  # Normalized kz for positive mode
        self.E_eigvec.append(E_eigvec)

    def _solve_layer_smatrix_symmetry(self):
        """Solve one layer as two independent mirror-parity systems."""
        p_blocks = [self._d(block) for block in self._P_blocks[-1]]
        q_blocks = [self._d(block) for block in self._Q_blocks[-1]]
        kz_blocks = [self._d(block) for block in self._kz_blocks[-1]]
        e_blocks = [self._d(block) for block in self._E_eigvec_blocks[-1]]
        vf_blocks = self._Vf_blocks
        pinv_blocks = [torch.linalg.inv(block) for block in p_blocks]

        use_p_inverse = True
        if self.avoid_Pinv_instability:
            p_errors, q_errors = [], []
            for p_block, q_block, pinv_block in zip(
                p_blocks, q_blocks, pinv_blocks
            ):
                identity = torch.eye(
                    p_block.shape[-1], dtype=self._dtype, device=self._device
                )
                qinv_block = torch.linalg.inv(q_block)
                p_errors.extend(
                    (
                        torch.max(torch.abs(p_block.detach() @ pinv_block.detach() - identity)),
                        torch.max(torch.abs(pinv_block.detach() @ p_block.detach() - identity)),
                    )
                )
                q_errors.extend(
                    (
                        torch.max(torch.abs(q_block.detach() @ qinv_block.detach() - identity)),
                        torch.max(torch.abs(qinv_block.detach() @ q_block.detach() - identity)),
                    )
                )
            p_error = torch.stack(p_errors).max()
            q_error = torch.stack(q_errors).max()
            self.Pinv_instability.append(p_error)
            self.Qinv_instability.append(q_error)
            use_p_inverse = bool(p_error < self.max_Pinv_instability)

        h_blocks, cf_blocks, cb_blocks = [], [], []
        layer_blocks = [[], [], [], []]
        for p_block, q_block, kz, e_block, vf_block, pinv_block in zip(
            p_blocks,
            q_blocks,
            kz_blocks,
            e_blocks,
            vf_blocks,
            pinv_blocks,
        ):
            size = p_block.shape[-1]
            identity = torch.eye(size, dtype=self._dtype, device=self._device)
            kz_matrix = torch.diag(kz)
            phase = torch.diag(
                torch.exp(1.0j * self.omega * kz * self.thickness[-1])
            )
            if use_p_inverse:
                h_block = pinv_block @ e_block @ kz_matrix
            else:
                h_block = q_block @ e_block @ torch.linalg.inv(kz_matrix)
            h_blocks.append(h_block)

            vf_h = torch.linalg.solve(vf_block, h_block)
            a = e_block + vf_h
            b = (e_block - vf_h) @ phase
            coupling = torch.cat(
                (torch.cat((a, b), dim=1), torch.cat((b, a), dim=1)), dim=0
            )
            rhs_f = torch.cat((2 * identity, torch.zeros_like(identity)), dim=0)
            rhs_b = torch.cat((torch.zeros_like(identity), 2 * identity), dim=0)
            cf = torch.linalg.solve(coupling, rhs_f)
            cb = torch.linalg.solve(coupling, rhs_b)
            cf_blocks.append(cf)
            cb_blocks.append(cb)

            e_phase = e_block @ phase
            layer_blocks[0].append(e_phase @ cf[:size] + e_block @ cf[size:])
            layer_blocks[1].append(
                e_block @ cf[:size] + e_phase @ cf[size:] - identity
            )
            layer_blocks[2].append(
                e_phase @ cb[:size] + e_block @ cb[size:] - identity
            )
            layer_blocks[3].append(e_block @ cb[:size] + e_phase @ cb[size:])

        self._H_eigvec_blocks.append(h_blocks)
        self.H_eigvec.append(None)
        self._Cf_blocks.append(cf_blocks)
        self._Cb_blocks.append(cb_blocks)
        self.Cf.append(None)
        self.Cb.append(None)

        public_lists = (
            self.layer_S11,
            self.layer_S21,
            self.layer_S12,
            self.layer_S22,
        )
        for index, public in enumerate(public_lists):
            self._layer_S_blocks[index].append(layer_blocks[index])
            public.append(None)

    def _solve_layer_smatrix(self):
        Kz_norm = torch.diag(self.kz_norm[-1])
        phase = torch.diag(
            torch.exp(1.0j * self.omega * self.kz_norm[-1] * self.thickness[-1])
        )

        Pinv_tmp = torch.linalg.inv(self.P[-1])
        if self.avoid_Pinv_instability:

            Pinv_ins_tmp1 = torch.max(
                torch.abs(
                    torch.matmul(self.P[-1].detach(), Pinv_tmp.detach())
                    - torch.eye(self.P[-1].shape[-1]).to(self.P[-1])
                )
            )
            Pinv_ins_tmp2 = torch.max(
                torch.abs(
                    torch.matmul(Pinv_tmp.detach(), self.P[-1].detach())
                    - torch.eye(self.P[-1].shape[-1]).to(self.P[-1])
                )
            )
            Qinv_ins_tmp1 = torch.max(
                torch.abs(
                    torch.matmul(
                        self.Q[-1].detach(), torch.linalg.inv(self.Q[-1]).detach()
                    )
                    - torch.eye(self.Q[-1].shape[-1]).to(self.Q[-1])
                )
            )
            Qinv_ins_tmp2 = torch.max(
                torch.abs(
                    torch.matmul(
                        self.Q[-1].detach(), torch.linalg.inv(self.Q[-1]).detach()
                    )
                    - torch.eye(self.Q[-1].shape[-1]).to(self.Q[-1])
                )
            )

            self.Pinv_instability.append(torch.maximum(Pinv_ins_tmp1, Pinv_ins_tmp2))
            self.Qinv_instability.append(torch.maximum(Qinv_ins_tmp1, Qinv_ins_tmp2))

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

        Ctmp1 = torch.vstack(
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
        Ctmp2 = torch.vstack(
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
        Ctmp = torch.hstack((Ctmp1, Ctmp2))

        # Mode coupling coefficients
        self.Cf.append(
            torch.matmul(
                torch.linalg.inv(Ctmp),
                torch.vstack(
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
                torch.vstack(
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
                self.Cf[-1][: 2 * self.order_N, :],
            )
            + torch.matmul(self.E_eigvec[-1], self.Cf[-1][2 * self.order_N :, :])
        )
        self.layer_S21.append(
            torch.matmul(self.E_eigvec[-1], self.Cf[-1][: 2 * self.order_N, :])
            + torch.matmul(
                torch.matmul(self.E_eigvec[-1], phase),
                self.Cf[-1][2 * self.order_N :, :],
            )
            - torch.eye(2 * self.order_N, dtype=self._dtype, device=self._device)
        )
        self.layer_S12.append(
            torch.matmul(
                torch.matmul(self.E_eigvec[-1], phase),
                self.Cb[-1][: 2 * self.order_N, :],
            )
            + torch.matmul(self.E_eigvec[-1], self.Cb[-1][2 * self.order_N :, :])
            - torch.eye(2 * self.order_N, dtype=self._dtype, device=self._device)
        )
        self.layer_S22.append(
            torch.matmul(self.E_eigvec[-1], self.Cb[-1][: 2 * self.order_N, :])
            + torch.matmul(
                torch.matmul(self.E_eigvec[-1], phase),
                self.Cb[-1][2 * self.order_N :, :],
            )
        )

    def _solve_global_smatrix_symmetry(self):
        """Connect layers independently in each parity block, then reconstruct S."""
        solved = [[], [], [], []]
        coupling_by_parity = []

        for parity, size in enumerate(self._symmetry_block_sizes):
            if self.layer_N > 0:
                current = [
                    self._d(self._layer_S_blocks[index][0][parity])
                    for index in range(4)
                ]
                coupling = [
                    [self._d(self._Cf_blocks[0][parity])],
                    [self._d(self._Cb_blocks[0][parity])],
                ]
            else:
                identity = torch.eye(size, dtype=self._dtype, device=self._device)
                zero = torch.zeros_like(identity)
                current = [identity, zero, zero, identity]
                coupling = [[], []]

            for layer in range(1, self.layer_N):
                next_layer = [
                    self._d(self._layer_S_blocks[index][layer][parity])
                    for index in range(4)
                ]
                current, coupling = self._RS_prod_reduced(
                    current,
                    next_layer,
                    coupling,
                    [
                        [self._d(self._Cf_blocks[layer][parity])],
                        [self._d(self._Cb_blocks[layer][parity])],
                    ],
                    size,
                )

            if hasattr(self, "Sin"):
                current, coupling = self._RS_prod_reduced(
                    [self._Sin_blocks[index][parity] for index in range(4)],
                    current,
                    [[], []],
                    coupling,
                    size,
                )
            if hasattr(self, "Sout"):
                current, coupling = self._RS_prod_reduced(
                    current,
                    [self._Sout_blocks[index][parity] for index in range(4)],
                    coupling,
                    [[], []],
                    size,
                )

            for index, block in enumerate(current):
                solved[index].append(block)
            coupling_by_parity.append(coupling)

        self.S = [self._assemble_boundary_blocks(blocks) for blocks in solved]
        self.C = [[], []]
        self._global_C_blocks = coupling_by_parity
        if self._offload_device is not None:
            self._global_C_blocks = [
                [
                    [tensor.to(self._offload_device) for tensor in direction]
                    for direction in parity
                ]
                for parity in coupling_by_parity
            ]

    def _RS_prod_reduced(self, Sm, Sn, Cm, Cn, size):
        """Redheffer star product for one mirror-parity block."""
        identity = torch.eye(size, dtype=self._dtype, device=self._device)
        tmp1 = torch.linalg.inv(identity - Sm[2] @ Sn[1])
        tmp2 = torch.linalg.inv(identity - Sn[1] @ Sm[2])

        result = [
            Sn[0] @ tmp1 @ Sm[0],
            Sm[1] + Sm[3] @ tmp2 @ Sn[1] @ Sm[0],
            Sn[2] + Sn[0] @ tmp1 @ Sm[2] @ Sn[3],
            Sm[3] @ tmp2 @ Sn[3],
        ]

        coupling = [[], []]
        for index in range(len(Cm[0])):
            coupling[0].append(Cm[0][index] + Cm[1][index] @ tmp2 @ Sn[1] @ Sm[0])
            coupling[1].append(Cm[1][index] @ tmp2 @ Sn[3])
        for index in range(len(Cn[0])):
            coupling[0].append(Cn[0][index] @ tmp1 @ Sm[0])
            coupling[1].append(
                Cn[1][index] + Cn[0][index] @ tmp1 @ Sm[2] @ Sn[3]
            )
        return result, coupling

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
