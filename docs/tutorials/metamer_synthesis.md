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

Download the executed notebook: **{nb-download}`metamer_synthesis.ipynb`**!

```

# Synthesizing Metamers

```{code-cell} ipython3
import matplotlib.pyplot as plt
import torch

import fenestration as fen
import plenoptic as po

# so that relative sizes of axes created by po.plot.imshow and others look right
plt.rcParams["figure.dpi"] = 72
# Animation-related settings
plt.rcParams["animation.html"] = "html5"
# use single-threaded ffmpeg for animation writer
plt.rcParams["animation.writer"] = "ffmpeg"
plt.rcParams["animation.ffmpeg_args"] = ["-threads", "1"]

%load_ext autoreload
%autoreload 2
```


## Loading in images

```{code-cell} ipython3
reptile = po.data.reptile_skin()
einstein = po.data.einstein()
imgs = torch.cat([reptile, einstein])
po.plot.imshow(imgs);
```

## Synthesizing image metamers

```{code-cell} ipython3
model = fen.PoolingWindows(0.5,reptile.shape[-2:])
model.eval()
po.remove_grad(model)
met = po.Metamer(imgs, model)
met.synthesize(store_progress=True, max_iter=200);
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 4, figsize=(16,4), layout="tight")
po.plot.imshow(imgs, batch_idx=0, ax=axes[0], title="target image")
model.plot_windows(ax=axes[0])
axes[0].xaxis.set_visible(False)
axes[0].yaxis.set_visible(False)
po.plot.synthesis_status(met, batch_idx = 0, fig=fig, axes_idx={"misc": 0});

fig, axes = plt.subplots(1, 4, figsize=(16,4), layout="tight")
po.plot.imshow(imgs, batch_idx=1, ax=axes[0], title="target image")
model.plot_windows(ax=axes[0])
axes[0].xaxis.set_visible(False)
axes[0].yaxis.set_visible(False)
po.plot.synthesis_status(met, batch_idx = 1, fig=fig, axes_idx={"misc": 0});
```

```{code-cell} ipython3
model = fen.PoolingWindows(0.25,reptile.shape[-2:])
model.eval()
po.remove_grad(model)
met = po.Metamer(imgs, model)
met.synthesize(store_progress=True, max_iter=200);

fig, axes = plt.subplots(1, 4, figsize=(16,4), layout="tight")
po.plot.imshow(imgs, batch_idx=0, ax=axes[0], title="target image")
model.plot_windows(ax=axes[0])
axes[0].xaxis.set_visible(False)
axes[0].yaxis.set_visible(False)
po.plot.synthesis_status(met, batch_idx = 0, fig=fig, axes_idx={"misc": 0});

fig, axes = plt.subplots(1, 4, figsize=(16,4), layout="tight")
po.plot.imshow(imgs, batch_idx=1, ax=axes[0], title="target image")
model.plot_windows(ax=axes[0])
axes[0].xaxis.set_visible(False)
axes[0].yaxis.set_visible(False)
po.plot.synthesis_status(met, batch_idx = 1, fig=fig, axes_idx={"misc": 0});
```

```{code-cell} ipython3
fig = po.plot.imshow(imgs, batch_idx=0, title="target image")
model.plot_windows(ax=fig.axes[0]);
plt.xlim(128,200);
plt.ylim(92,162);
```

## Comparing gaussian and cosine window synthesis

```{code-cell} ipython3
model_gauss = fen.PoolingWindows(0.25, einstein.shape[-2:], window_type="gaussian")
model_gauss.eval()
po.remove_grad(model_gauss)
met_gauss = po.Metamer(einstein, model_gauss)
met_gauss.synthesize(store_progress=True, max_iter=200);

model_cosine = fen.PoolingWindows(0.25, einstein.shape[-2:], window_type="cosine")
model_cosine.eval()
po.remove_grad(model_cosine)
met_cosine = po.Metamer(einstein, model_cosine)
met_cosine.synthesize(store_progress=True, max_iter=200);

fig, axes = plt.subplots(1, 2, figsize=(8,4), layout="tight")
po.plot.imshow(einstein, ax=axes[0], title="target image")
model_gauss.plot_windows(ax=axes[0])
axes[0].xaxis.set_visible(False)
axes[0].yaxis.set_visible(False)

po.plot.imshow(einstein, ax=axes[1], title="target image")
model_cosine.plot_windows(ax=axes[1])
axes[1].xaxis.set_visible(False)
axes[1].yaxis.set_visible(False)
```

```{code-cell} ipython3
po.plot.synthesis_status(met_gauss);
po.plot.synthesis_status(met_cosine);
```
