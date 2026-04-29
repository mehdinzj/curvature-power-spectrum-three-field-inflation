# Code for calculating the curvature power spectrum in three-field inflation

This repository contains a Python implementation for computing the background dynamics and perturbation evolution in three-field inflationary models, with a focus on the resulting curvature power spectrum.

## Overview

The code numerically solves the coupled equations governing multi-field inflation, including:

* Background field evolution
* Linear perturbations
* Computation of the curvature power spectrum

It is designed to support reproducible research and accompanies the results presented in the associated paper.

## Features

* Numerical integration of three-field inflationary dynamics
* Evolution of scalar perturbations
* Computation of the curvature power spectrum
* Modular structure for extending to other models or potentials

## Requirements

The code is written in Python and requires the following packages:

* numpy
* scipy
* matplotlib

Install dependencies with:

```
pip install -r requirements.txt
```

## Usage

Run the main script to perform the full computation:

```
python main.py
```

Additional scripts and modules can be found in the repository for specific tasks such as background evolution and perturbation analysis.

## Reproducibility

All results in the accompanying paper can be reproduced using the code in this repository.
A specific version of the code used in the paper will be archived and assigned a DOI via Zenodo.

## Citation

If you use this code in your research, please cite:

```
Author Name (Year). Curvature Power Spectrum in Three-Field Inflation. 
GitHub repository. DOI: (to be added)
```
## Contact

For questions or issues, please open an issue on the repository or contact the author.
