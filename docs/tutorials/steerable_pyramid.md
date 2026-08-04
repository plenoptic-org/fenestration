---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: fenestration
  language: python
  name: python3
---

```{admonition} Run this notebook yourself!
:class: important

Download the executed notebook: **{nb-download}`steerable_pyramid.ipynb`**!

```

# Introduction

In this tutorial, we will show how `fenestration` can be used on dictionaries of 4d tensors, like the output of steerable pyramids. Here, we will use the [SteerablePyramidFreq](https://docs.plenoptic.org/docs/pulls/467/api/generated/plenoptic.process.SteerablePyramidFreq.html#plenoptic.process.SteerablePyramidFreq) process from [plenoptic](https://docs.plenoptic.org/docs/pulls/467/index.html). For an introduction to steerable pyramids, we recommend [plenoptic's user guide](https://docs.plenoptic.org/docs/pulls/467/user_guide/models_and_metrics/Steerable_Pyramid.html) and the [tutorial from pyrtools](https://pyrtools.readthedocs.io/en/latest/tutorials/03_steerable_pyramids.html).

```{code-cell} ipython3
import matplotlib.pyplot as plt
import torch

import fenestration as fen

%load_ext autoreload
%autoreload 2
%matplotlib inline
```

First, we will load in an example image. For plenoptic images, these must be 4d tensors with shape batch by channel by height by width.

```{code-cell} ipython3
img = torch.from_numpy(plt.imread('../_static/images/parrot.png').astype(np.float32)) / 255
while img.ndim < 4:
    img = img.unsqueeze(0)
```
