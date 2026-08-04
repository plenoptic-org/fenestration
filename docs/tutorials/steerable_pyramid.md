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

```{code-cell} ipython3
:tags: [hide-input]

import warnings

warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release,",
    category=UserWarning,
)
```

```{admonition} Run this notebook yourself!
:class: important

Download the executed notebook: **{nb-download}`steerable_pyramid.ipynb`**!

```

# Steerable Pyramids

In this tutorial, we will show how `fenestration` can be used on dictionaries of 4d tensors, like the output of steerable pyramids. Here, we will use the [SteerablePyramidFreq](https://docs.plenoptic.org/docs/pulls/467/api/generated/plenoptic.process.SteerablePyramidFreq.html#plenoptic.process.SteerablePyramidFreq) process from [plenoptic](https://docs.plenoptic.org/docs/pulls/467/index.html). For an introduction to steerable pyramids, we recommend [plenoptic's user guide](https://docs.plenoptic.org/docs/pulls/467/user_guide/models_and_metrics/Steerable_Pyramid.html) and the [tutorial from pyrtools](https://pyrtools.readthedocs.io/en/latest/tutorials/03_steerable_pyramids.html).

## Setting Up Figures

First, we will load in an example image. For plenoptic images, these must be 4d tensors with shape batch by channel by height by width. We will also take advantage of plenoptic's plotting functions.

```{code-cell} ipython3
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

import fenestration as fen
import plenoptic as po

%load_ext autoreload
%autoreload 2
%matplotlib inline
```

```{code-cell} ipython3
img = torch.from_numpy(plt.imread('../_static/images/parrot.png').astype(np.float32)) / 255
while img.ndim < 4:
    img = img.unsqueeze(0)
print(img.shape)
po.plot.imshow(img);
```

## Creating the Steerable Pyramid

Now we will use the SteerablePyramidFreq process to create the pyramid and extract the coefficients. The parameter `height` refers to the height of the pyramid, which should be the same as the number of `scales` we create with `PoolingWindows`. Here we see that `pyr_coeffs` is a dictionary with keys 0-3, corresponding to `height=4` in addition to `residual_lowpass` and `residual_highpass`. The shape of each of the dictionary elements now has an additional dimension corresponding to each of the four orientations.

```{code-cell} ipython3
# create the pyramid
pyr = po.process.SteerablePyramidFreq(img.shape[-2:], height=4, downsample=False)
# get the pyramid coefficients; this is equivalent to pyr.forward(img)
pyr_coeffs = pyr(img)
print(pyr_coeffs.keys())
print(pyr_coeffs[0].shape)
```

We can manipulate these coefficient outputs to display each of the image filters.

```{code-cell} ipython3
# convert coeffs to tensor
coeffs_tensor, _ = pyr.convert_pyr_to_tensor(pyr_coeffs)
# rearrange so that the residuals are at the end
coeffs_tensor = [
    coeffs_tensor[:, 1:-1],
    coeffs_tensor[:, :1],
    coeffs_tensor[:, -1:],
]
po.plot.imshow(coeffs_tensor, col_wrap=pyr.num_orientations);
```

## Applying PoolingWindows to Steerable Pyramid

In the previous example, we set the argument `downsample=False` in `SteerablePyramidFreq` so all coefficients would be the same size for easier visualization. However, when calculating the pooled coefficients at each scale, we want to use the default downsampling. Here we see that as we increase the height, the image size decreases.

```{code-cell} ipython3
# create the pyramid
pyr = po.process.SteerablePyramidFreq(img.shape[-2:], height=4)
# get the pyramid coefficients; this is equivalent to pyr.forward(img)
pyr_coeffs = pyr(img)
print(pyr_coeffs.keys())
print(pyr_coeffs[0].shape)
print(pyr_coeffs[1].shape)
```

The input to PoolingWindow's `forward` method must be a dictionary of 4d tensors (or a single 4d tensor) in which the keys are `(scale, orientation)` tuples. Therefore, we remove `residual_highpass` and `residual_lowpass` and rearrange these values into a new dictionary `new_pyr`.

```{code-cell} ipython3
for k in ['residual_highpass', 'residual_lowpass']:
    pyr_coeffs.pop(k)

new_pyr = {}
for k, v in pyr_coeffs.items():
    for ori in range(v.shape[2]):
        new_pyr.update({(k,ori): v[:,:,ori,:,:]})
print(new_pyr.keys())
```

We can now instantiate our `PoolingWindows` object (remember to use 4 scales!) and pass it our new dictionary. The `pooled_coeffs` will have the same keys as our input dictionary and its values will be pooled versions of the corresponding coefficients. Note the warnings about some windows being too small.

```{code-cell} ipython3
pw = fen.PoolingWindows(0.5, img.shape[-2:], num_scales=4)
pooled_coeffs = pw(new_pyr)
for k, v in pooled_coeffs.items():
    print(f'scale {k[0]}, orientation band {k[1]}: {v.shape}')
```

Finally, we can use the `project` method of `PoolingWindows` to project the pooled values back onto an image. Here we can visualize these projections at each scale (rows) and orientation (columns).

```{code-cell} ipython3
mpl.rcParams['xtick.bottom'] = False
mpl.rcParams['xtick.labelbottom'] = False
mpl.rcParams['ytick.left'] = False
mpl.rcParams['ytick.labelleft'] = False

proj = pw.project(pooled_coeffs)
fig, ax = plt.subplots(4,4, layout="constrained")
for sc in range(4):
    for ori in range(4):
        ax[sc,ori].imshow(torch.squeeze(proj[(sc,ori)]), cmap="gray")
```
