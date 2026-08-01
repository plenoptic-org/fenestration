(index-doc)=
# Fenestration

![Python version](https://img.shields.io/badge/python-3.10|3.11|3.12|3.13|3.14-blue.svg)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/format.json)](https://github.com/astral-sh/ruff)

`fenestration` is a python library for generating foveated pooling windows. These pooling windows tile the input image space such that windows are small at the center and grow in size towards the outer edge. This effect is meant to sample the input in a similar fashion to the human visual system, in which information is sampled more finely near the fovea and more coarsely in the periphery. These windows can be used in a variety of contexts, particularly for generating more biologically-inspired image metamers and processing images at multiple scales, like those created by the steerable pyramid implementation.

The original implementation of these pooling windows was published in [Freeman, J., & Simoncelli, E. P. (2011). Metamers of the ventral stream. Nature Neuroscience.](http://dx.doi.org/10.1038/nn.2889) Matlab code for these windows exist as part of Jeremy Freeman's [repo](https://github.com/freeman-lab/metamers/) for this paper. This repo is not a direct port of that code, but a conceptual reimplementation, following the math outlined in the supplemental materials, and also includes a version using Gaussian windows. More recently, these windows were also used to generate stimuli in [Broderick, W. F., Rufo, G., Winawer, J. & Simoncelli, E. P. (2023). Foveated metamers of the early visual system. eLife.](http://dx.doi.org/10.7554/eLife.90554.2)

For information on how to use the package, view the Tutorials and API pages linked at the top!

## Installation

To use `fenestration` yourself, you can install it directly from GitHub:

```bash
pip install git+https://github.com/plenoptic-org/fenestration.git
```

Alternatively, if you want use a local copy, you can clone the GitHub repository with the lines below, using SSH:

```bash
git clone git@github.com:plenoptic-org/fenestration.git
```

or HTTPS:

```bash
git clone https://github.com/plenoptic-org/fenestration.git
```

If you cloned the repository, navigate to the directory and run the following to create a virtual environment and download all dependencies:

```bash
cd fenestration
# create virtual environment
python -m venv .venv
# activate environment
.venv/Scripts/activate
# install all dependencies
pip install -e .
```

This code works with the python versions currently supported by [PyTorch](https://pytorch.org/get-started/locally/).

## Related packages

- [plenoptic](https://docs.plenoptic.org/docs/tags/2.1.0/index.html): This package is largely built to be compatible with plenoptic, particularly for metamer generation and for steerable pyramids. We recommend you check out plenoptic for many other stimulus processing tools!
- [pytorch](https://pytorch.org): Optimized tensor library for deep learning using GPUs and CPUs. We require many stimulus inputs to `fenestration` to be pytorch tensors in order to support GPU and CPU processing.

## Support

This package is supported by the [Simons Foundation Flatiron Institute's Center
for Computational
Neuroscience](https://www.simonsfoundation.org/flatiron/center-for-computational-neuroscience/).

```{toctree}
:hidden:

tutorials/index
api/index

```
