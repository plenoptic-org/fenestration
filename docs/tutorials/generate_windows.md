---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: pooling
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

This notebook provides tutorials on the most common ways of generating and interacting with pooling windows. These can be used generally for building foveated windows and interacting with the resulting weights across windows.

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

Let's begin by creating a `PoolingWindows` object for image size `(256,256)` and visualize the window contours that are created. We must also input a `scaling` value that determines the size of the window (see [](choosing-scaling-values)).

```{code-cell} ipython3
pw = fen.PoolingWindows(0.5, (256,256))
pw.plot_windows(subset=False)
```

We can also change a number of other parameters that define the windows: `min_eccentricity` and `max_eccentricity` that define the extent of the windows within the image in degrees of visual angle, `num_scales` which controls the number of window scales generated, `cache_dir` for specifying a directory to cache the windows, and `window_type` which can be defined as `gaussian` or `cosine`.

```{code-cell} ipython3
pw = fen.PoolingWindows(
  scaling=0.5,
  img_res=(256,256),
  min_eccentricity=1,
  max_eccentricity=10,
  window_type='cosine'
  )
pw.plot_windows()
```

If you want to just generate the eccentricity rings and angular wedges separately, you can also call `fen.create_pooling_windows`. Here we will use `scaling=2` and and image size of `(256,256)`. We will also take advantage of [plenoptic's](https://plenoptic.org/) plotting function `po.imshow`.

```{code-cell} ipython3
import plenoptic as po

angle_w, ecc_w = fen.create_pooling_windows(2, (256, 256))
# only show first 8 eccentricity rings
fig = po.plot.imshow(ecc_w[:8].unsqueeze(0))
fig = po.plot.imshow(angle_w.unsqueeze(0))
plt.show()
```

It is also simple to reconstruct the windows from this angle and eccentricity data.

```{code-cell} ipython3
windows = torch.einsum('ahw,ehw->eahw', [angle_w, ecc_w]).flatten(0, 1)
```

However, in order to reproduce the results in `fen.PoolingWindows`, you would also need to normalize the windows so that they have an L1-norm of 1. This ensures that each eccentricity contributes equally, which can be useful when generating model metamers.

## Displaying Window Values

Now let's generate a figure with a noisy gradient across the image. We can then use `plot_window_values` to display the average values within each window.

```{code-cell} ipython3
img = torch.rand((1, 1, 256, 256), dtype=torch.float32) * torch.range(1/256,1,1/256)
plt.imshow(torch.squeeze(img), cmap='gray')
```

```{code-cell} ipython3
pw = fen.PoolingWindows(0.8, (256,256))
pw.plot_window_values(img, subset=False)
```

If you would like a summary of the size and values associated with the pooling windows, you can call `summarize_window_sizes` for either pixels or degrees.

```{code-cell} ipython3
from pprint import pprint
summary = pw.summarize_window_sizes(units="pixels")
pprint(summary)
summary = pw.summarize_window_sizes(units="degrees")
pprint(summary)
```

## Checking Windows
We also have a few additional visualization functions, including plotting the window widths and areas:

```{code-cell} ipython3
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['xtick.labelbottom'] = True
mpl.rcParams['ytick.left'] = True
mpl.rcParams['ytick.labelleft'] = True

fig, ax = plt.subplots(1,2, figsize=(10, 4))
pw.plot_window_widths(ax=ax[0]);
pw.plot_window_areas(ax=ax[1]);
```

and checking whether the windows have been normalized properly:

```{code-cell} ipython3
pw.plot_window_checks();
```

(choosing-scaling-values)=
## Choosing Scaling Values

However, the `scaling` values used in previous examples were arbitrary. Let's say you are displaying images for an experiment and want to build pooling windows ranging from 1-10 degrees of eccentricity, tiling the space with 5 angular windows. You can then find the precise scaling value to generate the corresponding `PoolingWindows` object.

```{code-cell} ipython3
scaling = fen.calculate.scaling(n_windows=5, min_ecc=1, max_ecc=10, std_dev=1)
pw = fen.PoolingWindows(scaling, (256,256), min_eccentricity=1, max_eccentricity=10)
ax = pw.plot_windows()
ax.set_title(f"Scaling = {scaling:.4f}");
```
