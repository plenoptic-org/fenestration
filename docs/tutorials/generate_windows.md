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

warnings.filterwarnings(
    "ignore",
    message="torch.range is deprecated and will be removed",
    category=UserWarning,
)
```

```{admonition} Run this notebook yourself!
:class: important

Download the executed notebook: **{nb-download}`generate_windows.ipynb`**!

```

(generate-windows-nb)=
# Generate Windows

This notebook provides tutorials on the most common ways of initializing and interacting with {class}`~fenestration.PoolingWindows`, which constructs foveated windows and uses them to take weighted averages across an image.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch

import fenestration as fen

mpl.rcParams['xtick.bottom'] = False
mpl.rcParams['xtick.labelbottom'] = False
mpl.rcParams['ytick.left'] = False
mpl.rcParams['ytick.labelleft'] = False

%load_ext autoreload
%autoreload 2
%matplotlib inline
```

## Creating PoolingWindows Objects

Let's begin by creating a {class}`~fenestration.PoolingWindows` object for image size `(256,256)` and visualize the window contours that are created. We must also input a `scaling` value that determines the size of the window (see [](choosing-scaling-values)).

```{code-cell} ipython3
pw = fen.PoolingWindows(0.5, (256,256))
pw.plot_windows(subset=False)
```

We can also change a number of other optional parameters:
- `min_eccentricity` and `max_eccentricity` define the extent of the windows within the image to support conversion between pixels and degrees of visual angle. We do not place windows in the foveal region within the `min_eccentricity` ring.
- `num_scales` which controls the number of window scales generated.
- `cache_dir` for specifying a directory to cache the windows. If windows are already cached there, will load them instead of re-creating them. If they're not present, will save them after creation.
- ` window_type` which can be defined as `gaussian` or `cosine`.

The default for {meth}`~fenestration.PoolingWindows.plot_windows` is to only plot four angle window slices to save time and memory:

```{code-cell} ipython3
pw = fen.PoolingWindows(
  scaling=0.5,
  img_res=(256,256),
  min_eccentricity=1,
  max_eccentricity=10,
  window_type='cosine'
  )
pw.plot_windows();
```

If you want to just generate the eccentricity rings and angular wedges separately, you can also call {meth}`~fenestration.create_pooling_windows`. Here we will use `scaling=2` and and image size of `(256,256)`. We will also take advantage of [plenoptic's](https://plenoptic.org/) plotting function `po.imshow`.

```{code-cell} ipython3
import plenoptic as po

angle_w, ecc_w = fen.create_pooling_windows(2, (256, 256))
# only show first 8 eccentricity rings
fig = po.plot.imshow(ecc_w[:8].unsqueeze(0))
fig = po.plot.imshow(angle_w.unsqueeze(0))
plt.show()
```

It is also possible to reconstruct the full windows from the separate angle and eccentricity tensors, though note this may take up a lot of memory! We can visualize each of the windows by indexing the first dimension of `windows`.

```{code-cell} ipython3
windows = torch.einsum('ahw,ehw->eahw', [angle_w, ecc_w]).flatten(0, 1)
plt.imshow(windows[0,:,:]);
```

However, in order to reproduce the outputs of {class}`~fenestration.PoolingWindows`, you would also need to normalize the windows (using {meth}`~fenestration.pooling.normalize_windows`) so that they have an L1-norm of 1. This ensures that each eccentricity contributes equally, which can be useful when generating model metamers.

## Displaying Window Values

Now let's generate a figure with a noisy gradient across the image. We can then use {meth}`~fenestration.PoolingWindows.plot_window_values` to display the average values within each window.

```{code-cell} ipython3
img = torch.rand((1, 1, 256, 256), dtype=torch.float32) * torch.range(1/256,1,1/256)
plt.imshow(torch.squeeze(img), cmap='gray');
```

Since we are "pooling" the input within each window, we now see a smooth gradient across the windows returned after averaging across the noise.

```{code-cell} ipython3
pw = fen.PoolingWindows(0.8, (256,256))
pw.plot_window_values(img, subset=False);
```

If you would like a summary of the size and values associated with the pooling windows, you can call {meth}`~fenestration.PoolingWindows.summarize_window_sizes` for either pixels or degrees. Here, we use [pprint](https://docs.python.org/3/library/pprint.html) to help us cleanly display the dictionary of sizes.

```{code-cell} ipython3
from pprint import pprint
summary = pw.summarize_window_sizes(units="pixels")
pprint(summary)
summary = pw.summarize_window_sizes(units="degrees")
pprint(summary)
```

## Checking Windows
We also have a few additional visualization functions, including plotting the window widths (left) and areas (right). Both of these figures show the window sizes along the y axes as the eccentricity increases along the x axes, measured in degrees of visual angle (calculated based on `min_eccentricity` and `max_eccentricity`).

The window widths depict two measurements: width of the windows along the radial (long) axis and angular (short) axis. Each individual window's size is defined by three measurements: 'top', 'half', and 'full' widths. Top is the width of the flat-top region of each window where the window's value is 1 (only present for cosine windows); full is the width of the entire window; half is the width at the half-max value. To get the approximate area, we multiply the radial width against the corresponding angular width, then divide by {math}`\frac{\pi}{4}`.

```{code-cell} ipython3
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['xtick.labelbottom'] = True
mpl.rcParams['ytick.left'] = True
mpl.rcParams['ytick.labelleft'] = True

fig, ax = plt.subplots(1,2, figsize=(10, 4))
pw.plot_window_widths(ax=ax[0]);
pw.plot_window_areas(ax=ax[1]);
```

We can also check whether the windows have been normalized properly so that they have an L1-norm of 1 to ensure that each eccentricity contributes equally. The first row shows the L1-norm of the windows, the second shows the sum. Each row will have one plot and, if everything worked correctly, they should each look like a sigmoid function that runs from 1 for small eccentricities to 0 for high eccentricities, measured in degrees of visual angle.

```{code-cell} ipython3
pw.plot_window_checks();
```

(choosing-scaling-values)=
## Choosing Scaling Values

However, the `scaling` values used in previous examples were arbitrary. Let's take a deeper dive into use cases of `scaling`, which is defined as the ratio of a window's radial full-width at half-maximum to eccentricity.

As shown in [Freeman, J., & Simoncelli, E. P. (2011). Metamers of the ventral stream. Nature Neuroscience.](https://www.nature.com/articles/nn.2889#Sec9), `scaling` values were adjusted to match specific behavioral thresholds for discriminating metameric stimuli. Additionally, in [Broderick, W. F., Rufo, G., Winawer, J. & Simoncelli, E. P. (2023). Foveated metamers of the early visual system. eLife.](https://elifesciences.org/reviewed-preprints/90554), the goal was to find the largest `scaling` value for generating metamers at which human and model discrimination performance was matched, and they showed how this critical value is impacted by image statistics, types of discrimination, and metamer synthesis initialization.

Here, we present a tool for calculating a specific `scaling` value based on some input parameters. Say you are displaying images for an experiment and want to build pooling windows ranging from 1-10 degrees of eccentricity, tiling the radial space with 5 windows. You can then find the precise scaling value to generate the corresponding {class}`~fenestration.PoolingWindows` object:

```{code-cell} ipython3
scaling = fen.calculate.scaling(n_windows=5, min_ecc=1, max_ecc=10, std_dev=1)
pw = fen.PoolingWindows(scaling, (256,256), min_eccentricity=1, max_eccentricity=10)
ax = pw.plot_windows()
ax.set_title(f"Scaling = {scaling:.4f}");
```
