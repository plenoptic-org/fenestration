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

```{admonition} Warning
:class: important

This notebook requires the optional dependency `plenoptic`, which can be installed with `pip`.

```

In this tutorial, we will show how `fenestration` can be used on dictionaries of 4d tensors, like the output of steerable pyramids. Here, we will use the [SteerablePyramidFreq](https://docs.plenoptic.org/docs/tags/2.1.0/api/generated/plenoptic.process.SteerablePyramidFreq.html#plenoptic.process.SteerablePyramidFreq) process from [plenoptic](https://docs.plenoptic.org/docs/tags/2.1.0/index.html).

The steerable pyramid can be thought of as a bank of oriented bandpass convolutional filters which span all orientations and frequencies. In this way, it is often thought of as having a representation similar to that of the primary visual cortex (V1). For an introduction to steerable pyramids, we recommend [plenoptic's user guide](https://docs.plenoptic.org/docs/tags/2.1.0/user_guide/models_and_metrics/Steerable_Pyramid.html) and the [tutorial from pyrtools](https://pyrtools.readthedocs.io/en/latest/tutorials/03_steerable_pyramids.html).

```{code-cell} ipython3
import fenestration as fen
import plenoptic as po

%load_ext autoreload
%autoreload 2
%matplotlib inline
```

## Setting Up Figures

First, we will load in an example image. These must be 4d tensors with shape batch by channel by height by width, which is the default for plenoptic's images. We will also take advantage of plenoptic's plotting functions in this tutorial.

```{code-cell} ipython3
img = po.data.einstein()
po.plot.imshow(img);
```

## Creating the Steerable Pyramid

Now we will initialize the `SteerablePyramidFreq` object and extract its coefficients for this image. The parameter `height` refers to the height of the pyramid (i.e., the number of spatial scales), which should be the same as the `scales` parameter we pass to {class}`~fenestration.PoolingWindows`. Here we see that `pyr_coeffs` is a dictionary with keys 0 to 3, corresponding to `height=4` in addition to `residual_lowpass` and `residual_highpass`. The shape of each of the dictionary elements now has an additional dimension corresponding to each of the orientations (four orientations is the default).

```{code-cell} ipython3
# create the pyramid
pyr = po.process.SteerablePyramidFreq(img.shape[-2:], height=4)
# get the pyramid coefficients; this is equivalent to pyr.forward(img)
pyr_coeffs = pyr(img)
print(pyr_coeffs.keys())
print(pyr_coeffs[0].shape)
```

We can visualize these coefficient outputs using `po.plot.pyrshow`. Each plot shows the coefficients for a given scale and orientation band. Height 00 shows the finest scales with high spatial frequencies and we can see the spatial frequencies decrease as we move down the rows. Each column shows a given orientation ("band"), with vertical in the first column, horizontal in the third, and the diagonals in the second and fourth. Finally, the residuals at the very bottom of the plot show the high and low frequencies, respectively, which are not captured by the pyramid.

```{code-cell} ipython3
po.plot.pyrshow(pyr_coeffs);
```

## Applying PoolingWindows to Steerable Pyramid

Now we can pass the pyramid coefficients into {class}`~fenestration.PoolingWindows` in order to get the pooled windows at each scale and orientation. However, the input to PoolingWindow's {meth}`~fenestration.PoolingWindows.forward` method must be a dictionary of 4d tensors (or a single 4d tensor) in which the keys are `(scale, orientation)` tuples. Therefore, we remove `residual_highpass` and `residual_lowpass` and rearrange these values into a new dictionary `new_pyr`.

```{code-cell} ipython3
new_pyr = {}
for k, v in pyr_coeffs.items():
    # skip residuals
    if isinstance(k, str):
        continue
    for ori in range(v.shape[2]):
        new_pyr[(k, ori)] = v[:, :, ori, :, :]
print(new_pyr.keys())
```

We can now instantiate our {meth}`~fenestration.PoolingWindows` object (with the same number of scales as the pyramid) and pass it our new dictionary. The `pooled_coeffs` will have the same keys as our input dictionary and its values will be pooled versions of the corresponding coefficients with the 3rd dimension corresponding to each window. Note the warnings about some windows being too small!

```{code-cell} ipython3
pw = fen.PoolingWindows(0.5, img.shape[-2:], num_scales=pyr.num_scales)
pooled_coeffs = pw(new_pyr)
for k, v in pooled_coeffs.items():
    print(f'scale {k[0]}, orientation band {k[1]}: {v.shape}')
```
