**solwa 0.3.0rc1**
======

* License: LGPL

Differentiable Rigorous Coupled-Wave Analysis (RCWA) package based on PyTorch

<br/>

Features
--------
**solwa** (**solar** + **rcwa**) is a PyTorch implementation of rigorous coupled-wave analysis (RCWA) based on **torcwa** https://github.com/kch3782/torcwa

* **GPU-accelerated** simulation on CUDA, with CPU support

* **Automatic differentiation** through simulations, material properties, and geometry for gradient-based optimization

* **Li Fourier factorization** for improved convergence, with the Laurent rule also available

* Geometry tools for common shapes and Boolean operations, including unions, intersections, and differences

* Wavelength-dependent refractive-index data with differentiable interpolation

* Scattering parameters, diffraction angles, electromagnetic field maps, and Poynting vector / power-flux calculations

* Optional layer offloading to reduce GPU memory use in multilayer simulations

* Units: Lorentz-Heaviside units

	* Speed of light: 1

	* Permittivity and permeability of vacuum: both 1

* Notation: exp(-*jωt*)

* [Performance guide](./docs/performance.md): choosing the order, precision and memory options

<br/>

Installation
------------
Install from PyPI using pip:

```bash
pip install solwa
```

For the latest nightly build of `main`, use `pip install --pre solwa`.

Alternatively, install from source by cloning this repository and running:

```bash
pip install .
```

PyTorch ≥ 2.10 is recommended for improved CUDA performance
(see PR: https://github.com/pytorch/pytorch/pull/166715)

Material Handling
------------
You can load dispersive materials from n/k tables with:

```python
import solwa

silicon = solwa.materials.Material("Materials_data/aSiH.txt")
silicon_lossless = solwa.materials.Material("Materials_data/aSiH.txt", lossless=True)
```

When `lossless=True`, the extinction coefficient `k` is forced to zero while
keeping the tabulated real refractive index `n`.

<br/>

significant changes from the original torcwa
------------
1. Constants are loaded from standard libraries instead of defining them again in the code.
2. Major cleanup of the code structure for better readability and maintainability.
3. Better error handling and messages.
4. Li Fourier factorization, with the Laurent rule available as an option.
5. Poynting-vector and power-flux calculations, including integration over selected areas.


<br/>

**solwa** Examples
---------------
1. [Example 0](./example/Example0.ipynb): Fresnel equation

2. [Example 1](./example/Example1.ipynb): Simulation with rectangular meta-atom  
Normal incidence / Parametric sweep on wavelength / View electromagnetic field

3. [Example 1-1](./example/Example1-1.ipynb): Simulation with stacked meta-atom  
Normal incidence / View electromagnetic field

4. [Example 2](./example/Example2.ipynb): Simulation with square meta-atom  
Oblique incidence / View electromagnetic field

5. [Example 3](./example/Example3.ipynb): Simulation with rectangular meta-atom  
Normal incidence / Parametric sweep on geometric parameters

6. [Example 4](./example/Example4.ipynb): Gradient calculation of cylindrical meta-atom  
Differentiation of transmittance with respect to radius

7. [Example 5](./example/Example5.ipynb): Shape optimization  
Maximize anisotropy

8. [Example 6](./example/Example6.ipynb): Topology optimization  
Maximize 1st order diffraction

9. [Example 7](./example/Example7.ipynb): Layer-by-layer absorption
Calculate absorbed power in each layer using Poynting flux

<br/>

Acknowledgements
----------------
The fundamental implementation, including mathematical formulation, was implemented in the **torcwa** package 
by Changhyun Kim and Byoungho Lee.

This **solwa** package is a modified version of **torcwa** for better usability.
It will be continuously updated to include more features and improvements.

If this package is useful for your research, please cite the following paper:
```
@article{
	title = {TORCWA: GPU-accelerated Fourier modal method and gradient-based optimization for metasurface design},
	journal = {Computer Physics Communications},
	volume = {282},
	pages = {108552},
	year = {2023},
	doi = {https://doi.org/10.1016/j.cpc.2022.108552},
	author = {Changhyun Kim and Byoungho Lee},
}
```

Work on the **solwa** package is supported by the ERC Consolidator Grant (No. 101125948, PHASE).

Furthermore, the authors acknowledge support by the state of Baden-Württemberg through bwHPC
through providing computational resources.



The work on the original **torcwa** was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. 2020R1A2B5B02002730) and Samsung Electronics Co., Ltd (IO201214-08164-01).

## GPU continuous integration

SOLWA can run its physics tests on a fresh Nebius GPU VM through GitHub Actions.
To run them on a pull request, add the `gpu-ci` label; they start after the CPU
tests pass.
See [GPU CI setup and recovery](ci/nebius/README.md) for the two secrets,
project/subnet configuration, the `gpu-ci` PR label, manual runs, and automatic
cleanup.
