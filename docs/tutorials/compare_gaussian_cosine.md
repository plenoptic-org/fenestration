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

Download the executed notebook: **{nb-download}`compare_gaussian_cosine.ipynb`**!

```

(comparing-window-types)=
# Comparing Window Types

In this package, we support two different window type shapes: raised cosine (used for the original implementation in [Freeman and Simoncelli, 2011](https://www.nature.com/articles/nn.2889)) and gaussian (used for a more recent implementation in [Broderick, Rufo, Winawer, & Simoncelli, 2023](https://elifesciences.org/reviewed-preprints/90554)). First, let's see what these functions actually look like, using our built-in functions {meth}`~fenestration.pooling.raised_cosine` and {meth}`~fenestration.pooling.gaussian`.

```{code-cell} ipython3
import fenestration as fen
import matplotlib.pyplot as plt
import torch
x = torch.linspace(-4, 4, 101)
gauss = fen.pooling.gaussian(x, std_dev=1)
cosine = fen.pooling.raised_cosine(x, transition_region_width=0.5)
fig, ax = subplots(1, 2, figsize = (10, 4))
ax[0].plot(x, gauss);
ax[0].set_title("Gaussian function");
ax[1].plot(x, cosine);
ax[1].set_title("Raised cosine function");
```

It is fairly easy to see the differences in the shapes of the functions, such that the raised cosine function is steeper and narrower whereas the gaussian function has larger tails and no flat-top region. Now let's see the actual windows projected onto a `(256,256)` sized image with `scaling=1`. Here we can visualize how the different impacts

```{code-cell} ipython3
pw_gauss = fen.PoolingWindows(1, (256,256), window_type="gaussian")
pw_cosine = fen.PoolingWindows(1, (256,256), window_type="cosine")

fig, ax = plt.subplots(1, 2, figsize=(10,4))
pw_gauss.plot_windows(ax=ax[0], subset=False)
pw_cosine.plot_windows(ax=ax[1], subset=False)
```
