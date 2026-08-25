"""
SOLWA: PyTorch-based Rigorous Coupled-Wave Analysis.

GPU-accelerated Fourier Modal Method with automatic differentiation support
for metasurface design and optimization.

This package provides:
- RCWA simulation (rcwa class)
- Geometry generation utilities (geometry, rcwa_geo)
- Material property management (materials)
- Stable eigendecomposition (Eig)

Uses Lorentz-Heaviside units with speed of light = 1 and
time harmonics notation exp(-jωt).
"""

from . import materials as materials
from .geometry import geometry as geometry
from .geometry import rcwa_geo as rcwa_geo
from .rcwa import rcwa as rcwa
from .torch_eig import Eig as Eig

__author__ = """Johannes Zeiser"""
__version__ = "0.1.0"
