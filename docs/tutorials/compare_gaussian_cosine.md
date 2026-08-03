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

Download the executed notebook: **{nb-download}`compare_gaussian_cosine.ipynb`**!

```

(comparing-window-types)=
# Comparing Window Types

In this package, we support two different window type shapes: raised cosine (used for the original implementation in [Freeman and Simoncelli, 2011](https://www.nature.com/articles/nn.2889)) and gaussian (used for a more recent implementation in [Broderick, Rufo, Winawer, & Simoncelli, 2023](https://elifesciences.org/reviewed-preprints/90554)). First, let's see what these functions actually look like, using our built-in functions {meth}`~fenestration.pooling.raised_cosine` and {meth}`~fenestration.pooling.gaussian`.

```{code-cell} ipython3
import fenestration as fen
import matplotlib.pyplot as plt
import torch
import plenoptic as po
import pyrtools as pt
```

```{code-cell} ipython3
x = torch.linspace(-4, 4, 101)
gauss = fen.pooling.gaussian(x, std_dev=1)
cosine = fen.pooling.raised_cosine(x, transition_region_width=0.5)
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
ax[0].plot(x, gauss);
ax[0].set_title("Gaussian function");
ax[1].plot(x, cosine);
ax[1].set_title("Raised cosine function");
```

It is fairly easy to see the differences in the shapes of the functions, such that the raised cosine function is steeper and narrower whereas the gaussian function has larger tails and no flat-top region. Now let's see the actual windows projected onto a `(256,256)` sized image with `scaling=1`. Here we can visualize how the different window types interact with the scaling value.

```{code-cell} ipython3
pw_gauss = fen.PoolingWindows(1, (256,256), window_type="gaussian")
pw_cosine = fen.PoolingWindows(1, (256,256), window_type="cosine")

fig, ax = plt.subplots(1, 2, figsize=(10,4))
pw_gauss.plot_windows(ax=ax[0], subset=False)
pw_cosine.plot_windows(ax=ax[1], subset=False)
```

Conversely, we can calculate the scaling value needed to tile the space with the same number of windows (for example, 5 log-eccentricity windows).

```{code-cell} ipython3
scaling_gauss = fen.calculate.scaling(5, std_dev=1)
scaling_cosine = fen.calculate.scaling(5, std_dev=None)
print(f"Gauss scale value = {scaling_gauss}")
print(f"Cosine scale value = {scaling_cosine}")
```

We can then use these scaling values to build similarly-sized windows for each window type.

```{code-cell} ipython3
pw_gauss = fen.PoolingWindows(scaling_gauss, (256,256), window_type="gaussian")
pw_cosine = fen.PoolingWindows(scaling_cosine, (256,256), window_type="cosine")

fig, ax = plt.subplots(1, 2, figsize=(10,4))
pw_gauss.plot_windows(ax=ax[0], subset=False);
pw_cosine.plot_windows(ax=ax[1], subset=False);
```

Using some tools from [plenoptic](https://docs.plenoptic.org/docs/pulls/467/index.html) and [pyrtools](https://pyrtools.readthedocs.io/en/latest/), we can visualize the differences between two windows at the same location.

```{code-cell} ipython3
win_cosine = po.to_numpy(pw_cosine.ecc_windows[0][3])*po.to_numpy(pw_cosine.angle_windows[0][3])
# normalize so that max value is 1, to match matlab implementation
win_cosine /= win_cosine.max()

win_gauss = po.to_numpy(pw_gauss.ecc_windows[0][3])*po.to_numpy(pw_gauss.angle_windows[0][3])
# normalize so that max value is 1, to match matlab implementation
win_gauss /= win_gauss.max()
pt.imshow([win_cosine, win_gauss, win_cosine-win_gauss], vrange='auto0', );
```

Again, we see that while the raised cosine window has a higher amplitude in the center, gaussian windows have more spread. We can quantify this by printing ``self.window_width_pixels``.

```{code-cell} ipython3
cos_r = pw_cosine.window_width_pixels[0]['radial_half'][3]
cos_a = pw_cosine.window_width_pixels[0]['angular_half'][3]
cos_amp = pw_cosine.window_max_amplitude
gauss_a = pw_gauss.window_width_pixels[0]['angular_half'][3]
gauss_r = pw_gauss.window_width_pixels[0]['radial_half'][3]
gauss_amp = pw_gauss.window_max_amplitude
print(f"Raised-cosine window has widths:\n\tradial: {cos_r:.03f}\n\tangular: {cos_a:.03f}\n\tamplitude: {cos_amp:.03f}")
print(f"Gaussian window has widths:\n\tradial: {gauss_r:.03f}\n\tangular: {gauss_a:.03f}\n\tamplitude: {gauss_amp:.03f}")
```
