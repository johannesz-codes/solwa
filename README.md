**solwa-0.1.0**
======

* License: LGPL

Differentiable Rigorous Coupled-Wave Analysis (RCWA) package based on PyTorch

<br/>

Features
--------
**solwa** (**solar** + **rcwa**) is a PyTorch implementation of rigorous coupled-wave analysis (RCWA) based on **torcwa** https://github.com/kch3782/torcwa

* **GPU-accelerated** simulation

* Supporting **automatic differentiation** for optimization

* Units: Lorentz-Heaviside units

	* Speed of light: 1

	* Permittivity and permeability of vacuum: both 1

* Notation: exp(-*jωt*)

<br/>

Installation
------------
currently only installation from source is supported

PyTorch ≥ 2.10 is recommended for improved CUDA performance
(see PR: https://github.com/pytorch/pytorch/pull/166715)

<br/>

significant changes from the original torcwa
------------
1. Constants are loaded from standard libraries instead of defining them again in the code.
2. Major cleanup of the code structure for better readability and maintainability.
3. Better error handling and messages.
4. Addition of functions for calculation of the poynting vector and power flux.


<br/>

SOLWA Examples
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
through providing computational resources



The work on the original **torcwa** was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. 2020R1A2B5B02002730) and Samsung Electronics Co., Ltd (IO201214-08164-01).