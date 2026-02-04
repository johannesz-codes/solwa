"""
TORCWA: PyTorch-based Rigorous Coupled-Wave Analysis.

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

from .torch_eig import Eig as Eig
from .geometry import geometry as geometry, rcwa_geo as rcwa_geo
from .rcwa import rcwa as rcwa
from . import materials as materials

__author__ = """Changhyun Kim"""
__version__ = "0.1.4.2"
