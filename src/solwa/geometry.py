import torch
import torch.fft


class geometry:
    """
    Geometry configuration for RCWA simulations.

    Provides methods for creating and manipulating 2D geometric shapes on a grid,
    including circles, ellipses, rectangles, and boolean operations.
    """

    def __init__(
        self,
        Lx: float = 1.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 100,
        edge_sharpness: float = 1000.0,
        *,
        dtype=torch.float32,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Initialize geometry configuration for RCWA simulations.

        Parameters
        ----------
        Lx : float, optional
            x-direction lattice constant. Default is 1.0.
        Ly : float, optional
            y-direction lattice constant. Default is 1.0.
        nx : int, optional
            x-axis sampling number. Default is 100.
        ny : int, optional
            y-axis sampling number. Default is 100.
        edge_sharpness : float, optional
            Sharpness of geometry edges. Default is 1000.0.
        dtype : torch.dtype, optional
            Geometry data type (torch.float32 or torch.float64). Default is torch.float32.
        device : torch.device, optional
            Geometry device (torch.device('cpu') or torch.device('cuda')). Default is CUDA if available, otherwise CPU.

        """
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.edge_sharpness = edge_sharpness

        self.dtype = dtype
        self.device = device

    def grid(self):
        """
        Update and initialize the spatial grid.

        Generates x and y coordinate arrays and meshgrid based on the current
        lattice constants (Lx, Ly) and sampling numbers (nx, ny).
        """

        self.x = (self.Lx / self.nx) * (
            torch.arange(self.nx, dtype=self.dtype, device=self.device) + 0.5
        )
        self.y = (self.Ly / self.ny) * (
            torch.arange(self.ny, dtype=self.dtype, device=self.device) + 0.5
        )
        self.x_grid, self.y_grid = torch.meshgrid(self.x, self.y, indexing="ij")

    def circle(self, R, Cx, Cy):
        """
        Generate a circular geometry.

        Parameters
        ----------
        R : float
            Radius of the circle.
        Cx : float
            x-coordinate of the circle center.
        Cy : float
            y-coordinate of the circle center.

        Returns
        -------
        torch.Tensor
            2D tensor representing the circular geometry with smooth edges.
        """

        self.grid()
        level = 1.0 - torch.sqrt(
            ((self.x_grid - Cx) / R) ** 2 + ((self.y_grid - Cy) / R) ** 2
        )
        return torch.sigmoid(self.edge_sharpness * level)

    def ellipse(self, Rx, Ry, Cx, Cy, theta=0.0):
        """
        Generate an elliptical geometry.

        Parameters
        ----------
        Rx : float
            Radius in the x-direction.
        Ry : float
            Radius in the y-direction.
        Cx : float
            x-coordinate of the ellipse center.
        Cy : float
            y-coordinate of the ellipse center.
        theta : float, optional
            Rotation angle in radians. Center is [Cx, Cy], rotation axis is z-axis. Default is 0.0.

        Returns
        -------
        torch.Tensor
            2D tensor representing the elliptical geometry with smooth edges.
        """

        theta = torch.as_tensor(theta, dtype=self.dtype, device=self.device)

        self.grid()
        level = 1.0 - torch.sqrt(
            (
                (
                    (self.x_grid - Cx) * torch.cos(theta)
                    + (self.y_grid - Cy) * torch.sin(theta)
                )
                / Rx
            )
            ** 2
            + (
                (
                    -(self.x_grid - Cx) * torch.sin(theta)
                    + (self.y_grid - Cy) * torch.cos(theta)
                )
                / Ry
            )
            ** 2
        )
        return torch.sigmoid(self.edge_sharpness * level)

    def square(self, W, Cx, Cy, theta=0.0):
        """
        Generate a square geometry.

        Parameters
        ----------
        W : float
            Width of the square.
        Cx : float
            x-coordinate of the square center.
        Cy : float
            y-coordinate of the square center.
        theta : float, optional
            Rotation angle in radians. Center is [Cx, Cy], rotation axis is z-axis. Default is 0.0.

        Returns
        -------
        torch.Tensor
            2D tensor representing the square geometry with smooth edges.
        """

        theta = torch.as_tensor(theta, dtype=self.dtype, device=self.device)

        self.grid()
        level = 1.0 - (
            torch.maximum(
                torch.abs(
                    (
                        (self.x_grid - Cx) * torch.cos(theta)
                        + (self.y_grid - Cy) * torch.sin(theta)
                    )
                    / (W / 2.0)
                ),
                torch.abs(
                    (
                        -(self.x_grid - Cx) * torch.sin(theta)
                        + (self.y_grid - Cy) * torch.cos(theta)
                    )
                    / (W / 2.0)
                ),
            )
        )
        return torch.sigmoid(self.edge_sharpness * level)

    def rectangle(self, Wx, Wy, Cx, Cy, theta=0.0):
        """
        Generate a rectangular geometry.

        Parameters
        ----------
        Wx : float
            Width in the x-direction.
        Wy : float
            Width in the y-direction.
        Cx : float
            x-coordinate of the rectangle center.
        Cy : float
            y-coordinate of the rectangle center.
        theta : float, optional
            Rotation angle in radians. Center is [Cx, Cy], rotation axis is z-axis. Default is 0.0.

        Returns
        -------
        torch.Tensor
            2D tensor representing the rectangular geometry with smooth edges.
        """

        theta = torch.as_tensor(theta, dtype=self.dtype, device=self.device)

        self.grid()
        level = 1.0 - (
            torch.maximum(
                torch.abs(
                    (
                        (self.x_grid - Cx) * torch.cos(theta)
                        + (self.y_grid - Cy) * torch.sin(theta)
                    )
                    / (Wx / 2.0)
                ),
                torch.abs(
                    (
                        -(self.x_grid - Cx) * torch.sin(theta)
                        + (self.y_grid - Cy) * torch.cos(theta)
                    )
                    / (Wy / 2.0)
                ),
            )
        )
        return torch.sigmoid(self.edge_sharpness * level)

    def rhombus(self, Wx, Wy, Cx, Cy, theta=0.0):
        """
        Generate a rhombus geometry.

        Parameters
        ----------
        Wx : float
            Diagonal length in the x-direction.
        Wy : float
            Diagonal length in the y-direction.
        Cx : float
            x-coordinate of the rhombus center.
        Cy : float
            y-coordinate of the rhombus center.
        theta : float, optional
            Rotation angle in radians. Center is [Cx, Cy], rotation axis is z-axis. Default is 0.0.

        Returns
        -------
        torch.Tensor
            2D tensor representing the rhombus geometry with smooth edges.
        """

        theta = torch.as_tensor(theta, dtype=self.dtype, device=self.device)

        self.grid()
        level = 1.0 - (
            torch.abs(
                (
                    (self.x_grid - Cx) * torch.cos(theta)
                    + (self.y_grid - Cy) * torch.sin(theta)
                )
                / (Wx / 2.0)
            )
            + torch.abs(
                (
                    -(self.x_grid - Cx) * torch.sin(theta)
                    + (self.y_grid - Cy) * torch.cos(theta)
                )
                / (Wy / 2.0)
            )
        )
        return torch.sigmoid(self.edge_sharpness * level)

    def super_ellipse(self, Wx, Wy, Cx, Cy, theta=0.0, power=2.0):
        """
        Generate a super-ellipse geometry.

        Parameters
        ----------
        Wx : float
            Width in the x-direction.
        Wy : float
            Width in the y-direction.
        Cx : float
            x-coordinate of the super-ellipse center.
        Cy : float
            y-coordinate of the super-ellipse center.
        theta : float, optional
            Rotation angle in radians. Center is [Cx, Cy], rotation axis is z-axis. Default is 0.0.
        power : float, optional
            Elliptic power parameter. Default is 2.0.

        Returns
        -------
        torch.Tensor
            2D tensor representing the super-ellipse geometry with smooth edges.
        """

        theta = torch.as_tensor(theta, dtype=self.dtype, device=self.device)

        self.grid()
        level = 1.0 - (
            torch.abs(
                (
                    (self.x_grid - Cx) * torch.cos(theta)
                    + (self.y_grid - Cy) * torch.sin(theta)
                )
                / (Wx / 2.0)
            )
            ** power
            + torch.abs(
                (
                    -(self.x_grid - Cx) * torch.sin(theta)
                    + (self.y_grid - Cy) * torch.cos(theta)
                )
                / (Wy / 2.0)
            )
            ** power
        ) ** (1 / power)
        return torch.sigmoid(self.edge_sharpness * level)

    def union(self, A, B):
        """
        Compute the union of two geometries (A ∪ B).

        Parameters
        ----------
        A : torch.Tensor
            First geometry tensor.
        B : torch.Tensor
            Second geometry tensor.

        Returns
        -------
        torch.Tensor
            Union of A and B.
        """

        return torch.maximum(A, B)

    def intersection(self, A, B):
        """
        Compute the intersection of two geometries (A ∩ B).

        Parameters
        ----------
        A : torch.Tensor
            First geometry tensor.
        B : torch.Tensor
            Second geometry tensor.

        Returns
        -------
        torch.Tensor
            Intersection of A and B.
        """

        return torch.minimum(A, B)

    def difference(self, A, B):
        """
        Compute the difference of two geometries (A - B = A ∩ B^c).

        Parameters
        ----------
        A : torch.Tensor
            First geometry tensor.
        B : torch.Tensor
            Second geometry tensor.

        Returns
        -------
        torch.Tensor
            Difference of A and B.
        """

        return torch.minimum(A, 1.0 - B)


class rcwa_geo:
    """
    Legacy class-based geometry configuration for RCWA simulations.

    Uses class variables and class methods for geometry generation.
    This class is deprecated and will be removed in a future version.
    Use the `geometry` class instead for new code.

    Class Attributes
    ----------------
    edge_sharpness : float
        Sharpness of geometry edges. Default is 100.0.
    Lx : float
        x-direction lattice constant. Default is 1.0.
    Ly : float
        y-direction lattice constant. Default is 1.0.
    nx : int
        x-axis sampling number. Default is 100.
    ny : int
        y-axis sampling number. Default is 100.
    dtype : torch.dtype
        Data type for geometry operations. Default is torch.float32.
    device : torch.device
        Device for geometry operations. Default is CUDA if available, otherwise CPU.
    """

    edge_sharpness = 100.0  # sharpness of edge
    Lx = 1.0  # x-direction Lattice constant
    Ly = 1.0  # y-direction Lattice constant
    nx = 100  # x-axis sampling number
    ny = 100  # y-axis sampling number
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        pass

    @classmethod
    def grid(cls):
        """
        Update and initialize the spatial grid.

        Generates x and y coordinate arrays and meshgrid based on the current
        class-level lattice constants (Lx, Ly) and sampling numbers (nx, ny).
        """

        cls.x = (cls.Lx / cls.nx) * (
            torch.arange(cls.nx, dtype=cls.dtype, device=cls.device) + 0.5
        )
        cls.y = (cls.Ly / cls.ny) * (
            torch.arange(cls.ny, dtype=cls.dtype, device=cls.device) + 0.5
        )
        cls.x_grid, cls.y_grid = torch.meshgrid(cls.x, cls.y, indexing="ij")

    @classmethod
    def circle(cls, R, Cx, Cy):
        """
        Generate a circular geometry.

        Parameters
        ----------
        R : float
            Radius of the circle.
        Cx : float
            x-coordinate of the circle center.
        Cy : float
            y-coordinate of the circle center.

        Returns
        -------
        torch.Tensor
            2D tensor representing the circular geometry with smooth edges.
        """

        cls.grid()
        level = 1.0 - torch.sqrt(
            ((cls.x_grid - Cx) / R) ** 2 + ((cls.y_grid - Cy) / R) ** 2
        )
        return torch.sigmoid(cls.edge_sharpness * level)

    @classmethod
    def ellipse(cls, Rx, Ry, Cx, Cy, theta=0.0):
        """
        Generate an elliptical geometry.

        Parameters
        ----------
        Rx : float
            Radius in the x-direction.
        Ry : float
            Radius in the y-direction.
        Cx : float
            x-coordinate of the ellipse center.
        Cy : float
            y-coordinate of the ellipse center.
        theta : float, optional
            Rotation angle in radians. Center is [Cx, Cy], rotation axis is z-axis. Default is 0.0.

        Returns
        -------
        torch.Tensor
            2D tensor representing the elliptical geometry with smooth edges.
        """

        theta = torch.as_tensor(theta, dtype=cls.dtype, device=cls.device)

        cls.grid()
        level = 1.0 - torch.sqrt(
            (
                (
                    (cls.x_grid - Cx) * torch.cos(theta)
                    + (cls.y_grid - Cy) * torch.sin(theta)
                )
                / Rx
            )
            ** 2
            + (
                (
                    -(cls.x_grid - Cx) * torch.sin(theta)
                    + (cls.y_grid - Cy) * torch.cos(theta)
                )
                / Ry
            )
            ** 2
        )
        return torch.sigmoid(cls.edge_sharpness * level)

    @classmethod
    def square(cls, W, Cx, Cy, theta=0.0):
        """
        Generate a square geometry.

        Parameters
        ----------
        W : float
            Width of the square.
        Cx : float
            x-coordinate of the square center.
        Cy : float
            y-coordinate of the square center.
        theta : float, optional
            Rotation angle in radians. Center is [Cx, Cy], rotation axis is z-axis. Default is 0.0.

        Returns
        -------
        torch.Tensor
            2D tensor representing the square geometry with smooth edges.
        """

        theta = torch.as_tensor(theta, dtype=cls.dtype, device=cls.device)

        cls.grid()
        level = 1.0 - (
            torch.maximum(
                torch.abs(
                    (
                        (cls.x_grid - Cx) * torch.cos(theta)
                        + (cls.y_grid - Cy) * torch.sin(theta)
                    )
                    / (W / 2.0)
                ),
                torch.abs(
                    (
                        -(cls.x_grid - Cx) * torch.sin(theta)
                        + (cls.y_grid - Cy) * torch.cos(theta)
                    )
                    / (W / 2.0)
                ),
            )
        )
        return torch.sigmoid(cls.edge_sharpness * level)

    @classmethod
    def rectangle(cls, Wx, Wy, Cx, Cy, theta=0.0):
        """
        Generate a rectangular geometry.

        Parameters
        ----------
        Wx : float
            Width in the x-direction.
        Wy : float
            Width in the y-direction.
        Cx : float
            x-coordinate of the rectangle center.
        Cy : float
            y-coordinate of the rectangle center.
        theta : float, optional
            Rotation angle in radians. Center is [Cx, Cy], rotation axis is z-axis. Default is 0.0.

        Returns
        -------
        torch.Tensor
            2D tensor representing the rectangular geometry with smooth edges.
        """

        theta = torch.as_tensor(theta, dtype=cls.dtype, device=cls.device)

        cls.grid()
        level = 1.0 - (
            torch.maximum(
                torch.abs(
                    (
                        (cls.x_grid - Cx) * torch.cos(theta)
                        + (cls.y_grid - Cy) * torch.sin(theta)
                    )
                    / (Wx / 2.0)
                ),
                torch.abs(
                    (
                        -(cls.x_grid - Cx) * torch.sin(theta)
                        + (cls.y_grid - Cy) * torch.cos(theta)
                    )
                    / (Wy / 2.0)
                ),
            )
        )
        return torch.sigmoid(cls.edge_sharpness * level)

    @classmethod
    def rhombus(cls, Wx, Wy, Cx, Cy, theta=0.0):
        """
        Generate a rhombus geometry.

        Parameters
        ----------
        Wx : float
            Diagonal length in the x-direction.
        Wy : float
            Diagonal length in the y-direction.
        Cx : float
            x-coordinate of the rhombus center.
        Cy : float
            y-coordinate of the rhombus center.
        theta : float, optional
            Rotation angle in radians. Center is [Cx, Cy], rotation axis is z-axis. Default is 0.0.

        Returns
        -------
        torch.Tensor
            2D tensor representing the rhombus geometry with smooth edges.
        """

        theta = torch.as_tensor(theta, dtype=cls.dtype, device=cls.device)

        cls.grid()
        level = 1.0 - (
            torch.abs(
                (
                    (cls.x_grid - Cx) * torch.cos(theta)
                    + (cls.y_grid - Cy) * torch.sin(theta)
                )
                / (Wx / 2.0)
            )
            + torch.abs(
                (
                    -(cls.x_grid - Cx) * torch.sin(theta)
                    + (cls.y_grid - Cy) * torch.cos(theta)
                )
                / (Wy / 2.0)
            )
        )
        return torch.sigmoid(cls.edge_sharpness * level)

    @classmethod
    def super_ellipse(cls, Wx, Wy, Cx, Cy, theta=0.0, power=2.0):
        """
        Generate a super-ellipse geometry.

        Parameters
        ----------
        Wx : float
            Width in the x-direction.
        Wy : float
            Width in the y-direction.
        Cx : float
            x-coordinate of the super-ellipse center.
        Cy : float
            y-coordinate of the super-ellipse center.
        theta : float, optional
            Rotation angle in radians. Center is [Cx, Cy], rotation axis is z-axis. Default is 0.0.
        power : float, optional
            Elliptic power parameter. Default is 2.0.

        Returns
        -------
        torch.Tensor
            2D tensor representing the super-ellipse geometry with smooth edges.
        """

        theta = torch.as_tensor(theta, dtype=cls.dtype, device=cls.device)

        cls.grid()
        level = 1.0 - (
            torch.abs(
                (
                    (cls.x_grid - Cx) * torch.cos(theta)
                    + (cls.y_grid - Cy) * torch.sin(theta)
                )
                / (Wx / 2.0)
            )
            ** power
            + torch.abs(
                (
                    -(cls.x_grid - Cx) * torch.sin(theta)
                    + (cls.y_grid - Cy) * torch.cos(theta)
                )
                / (Wy / 2.0)
            )
            ** power
        ) ** (1 / power)
        return torch.sigmoid(cls.edge_sharpness * level)

    @classmethod
    def union(cls, A, B):
        """
        Compute the union of two geometries (A ∪ B).

        Parameters
        ----------
        A : torch.Tensor
            First geometry tensor.
        B : torch.Tensor
            Second geometry tensor.

        Returns
        -------
        torch.Tensor
            Union of A and B.
        """

        return torch.maximum(A, B)

    @classmethod
    def intersection(cls, A, B):
        """
        Compute the intersection of two geometries (A ∩ B).

        Parameters
        ----------
        A : torch.Tensor
            First geometry tensor.
        B : torch.Tensor
            Second geometry tensor.

        Returns
        -------
        torch.Tensor
            Intersection of A and B.
        """

        return torch.minimum(A, B)

    @classmethod
    def difference(cls, A, B):
        """
        Compute the difference of two geometries (A - B = A ∩ B^c).

        Parameters
        ----------
        A : torch.Tensor
            First geometry tensor.
        B : torch.Tensor
            Second geometry tensor.

        Returns
        -------
        torch.Tensor
            Difference of A and B.
        """

        return torch.minimum(A, 1.0 - B)
