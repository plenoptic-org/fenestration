(index-doc)=
# Fenestration

![Python version](https://img.shields.io/badge/python-3.10|3.11|3.12|3.13|3.14-blue.svg)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/format.json)](https://github.com/astral-sh/ruff)

`fenestration` is a python library for generating foveated pooling windows. These pooling windows tile the input image space such that windows are small at the center and grow in size towards the outer edge. This effect is meant to sample the input in a similar fashion to the human visual system, in which information is sampled more finely near the fovea and more coarsely in the periphery. These windows can be used in a variety of contexts, particularly for generating more biologically-inspired image metamers and processing images at multiple scales, like those created by the steerable pyramid implementation.

## Installation

To use `fenestration` yourself, you will first need to clone the GitHub repository using the lines below, using SSH:

```bash
git clone git@github.com:plenoptic-org/fenestration.git
```

or HTTPS:

```bash
git clone https://github.com/plenoptic-org/fenestration.git
```

Then, navigate to the directory and run the following to create a virtual environment and download all dependencies:

```bash
cd fenestration
# create virtual environment
python -m venv .venv
# activate environment
.venv/Scripts/activate
# install all dependencies
pip install -e .
```

This code works with Python 3.10, 3.11, 3.12, 3.13, and 3.14 in order to match [PyTorch's compatibility](https://pytorch.org/get-started/locally/).

## Related packages

- [plenoptic](https://docs.plenoptic.org/docs/tags/2.1.0/index.html): This package is largely built to be compatible with plenoptic, particularly for metamer generation and for steerable pyramids. We recommend you check out plenoptic for many other stimulus processing tools!
- [pytorch](https://pytorch.org): Optimized tensor library for deep learning using GPUs and CPUs. We require many stimulus inputs to `fenestration` to be pytorch tensors in order to support GPU and CPU processing.

## Support

This package is supported by the [Simons Foundation Flatiron Institute's Center
for Computational
Neuroscience](https://www.simonsfoundation.org/flatiron/center-for-computational-neuroscience/).

```{toctree}
:hidden:

api/index
tutorials/index

```
