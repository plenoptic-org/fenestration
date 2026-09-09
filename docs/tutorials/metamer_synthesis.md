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

Here we provide a tutorial for using {class}`~fenestration.PoolingWindows` to create image metamers with [plenoptic](https://plenoptic.org/). By "metamers", we refer to images that are physcially different but have an identical representation for a given model. We will walk through examples of different models and how we can change {class}`~fenestration.PoolingWindows` features to generate different metamers.

```{code-cell} ipython3
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import fenestration as fen
import plenoptic as po

# so that relative sizes of axes created by po.plot.imshow and others look right
plt.rcParams["figure.dpi"] = 72
rcContext = {
    'xtick.major.bottom': False,
    'ytick.major.left': False,
}

%load_ext autoreload
%autoreload 2
```

## Loading in images

We begin by using plenoptic's [data](https://docs.plenoptic.org/docs/tags/2.1.0/api/images.html) and [plotting](https://docs.plenoptic.org/docs/tags/2.1.0/api/plot.html) functions to grab two images. Since these images are very different in terms of spatial frequency and structure, it will help us visualize the resulting metamers.

```{code-cell} ipython3
reptile_orig = po.data.reptile_skin()
einstein_orig = po.data.einstein()
reptile = F.interpolate(reptile_orig, size=(200,200))
einstein = F.interpolate(einstein_orig, size=(200,200))
po.plot.imshow([reptile, einstein]);
```

## Visualizing Pooled Windows

To generate metamers, we first have to define a model which takes in a 4d tensor and performs some computation on the input. The metamer synthesis step starts from a noisy image and tries to minimize the error between the model output of the original image and that of the synthesized image. With this model, the process will try to minimize error between the pooled values of all windows for the original and synthesized images.

Here, we define pooling windows with `scaling=0.8`. In the left figures, we have overlaid a wedge of pooling windows on the original images to visualize the intersection points of each pooling window that is averaging across the image. In the right figures, we use the {meth}`~fenestration.PoolingWindows.plot_window_values` method to displayed the pooled image values in each window that we will matching the metamer values to. You can see that the smaller, more central windows maintain information more similar to the original image, while the more eccentric windows have a more blurred representation.

```{code-cell} ipython3
model = fen.PoolingWindows(0.8,reptile.shape[-2:])

with plt.rc_context(rcContext):
    fig, axes = plt.subplots(2, 2, figsize=(20,20), layout="tight", sharex="all", sharey="all")
    po.plot.imshow(reptile, ax=axes[0,0], title=None)
    model.plot_windows(ax=axes[0,0])
    model.plot_window_values(reptile, ax=axes[0,1], subset=False)
    po.plot.imshow(einstein, ax=axes[1,0], title=None)
    model.plot_windows(ax=axes[1,0])
    model.plot_window_values(einstein, ax=axes[1,1], subset=False)
```

## Synthesizing image metamers

Now that we have reviewed how the {class}`~fenestration.PoolingWindows` model works, let's get to synthesizing some metamers. Here we set our model to evaluation mode, remove gradients from its parameters, initialize the metamer object, and run synthesis.

```{code-cell} ipython3
model.eval()
po.remove_grad(model)
met_reptile = po.Metamer(reptile, model)
met_reptile.synthesize(max_iter=500, stop_criterion=1e-6);
met_einstein = po.Metamer(einstein, model)
met_einstein.synthesize(max_iter=500, stop_criterion=1e-6);
```

We see that the loss has converged, great! Let's check out our metamers and plot the loss and error across each iteration. Since the model only cares about the average pixel values in each window, the metamer will also maintain the corresponding average pixel values in each window region, but does not "see" the noise in the generated image. Also note how the correspondence to the original image changes based on eccentricity due to the increasing window sizes.

```{code-cell} ipython3
po.plot.synthesis_status(met_reptile);
po.plot.synthesis_status(met_einstein);
```

## Changing Scaling Values

Now let's see how the scaling value impacts our metamers. Here we decrease `scaling` in half and again visualize the pooling windows.

```{code-cell} ipython3
model = fen.PoolingWindows(0.4,reptile.shape[-2:])

with plt.rc_context(rcContext):
    fig, axes = plt.subplots(2, 2, figsize=(20,20), layout="tight", sharex="all", sharey="all")
    po.plot.imshow(reptile, ax=axes[0,0], title=None)
    model.plot_windows(ax=axes[0,0])
    model.plot_window_values(reptile, ax=axes[0,1], subset=False)
    po.plot.imshow(einstein, ax=axes[1,0], title=None)
    model.plot_windows(ax=axes[1,0])
    model.plot_window_values(einstein, ax=axes[1,1], subset=False)
```

Also note the warning that we get now about some windows being smaller than a pixel! If the scaling value is small, it's possible for the smallest windows to be smaller than a pixel and not included in the windows, similar to the central region below `min_ecc`. You can access the minimum eccentricity for avoiding this issue with the attribute `self.calculated_min_eccentricity_degrees`. We can zoom in and see the small windows here:

```{code-cell} ipython3
fig = po.plot.imshow(reptile, title=None)
model.plot_windows(ax=fig.axes[0], origin="upper");
plt.xlim(95,125);
plt.ylim(85,115);
```

Despite this, we can still generate our metamer! Since we are using smaller windows, we can see finer-grained details of the original image across a wider range of the metamer.

```{code-cell} ipython3
model.eval()
po.remove_grad(model)
met_reptile = po.Metamer(reptile, model)
met_reptile.synthesize(max_iter=500, stop_criterion=1e-6);
met_einstein = po.Metamer(einstein, model)
met_einstein.synthesize(max_iter=500, stop_criterion=1e-6);

po.plot.synthesis_status(met_reptile);
po.plot.synthesis_status(met_einstein);
```

## Comparing Window Types

Although we have been using gaussian windows for the synthesis thus far, `fenestration` also supports raised-cosine windows. Using `scaling=0.4`, we will generate metamers for the cosine windows and compare to the previous gaussian windows. First, let's visualize the differences in size and shape of each window type for the same scaling value on the Einstein image. See [](compare-window-types) tutorial for additonal information comparing gaussian and cosine windows.

```{code-cell} ipython3
model_cosine = fen.PoolingWindows(0.4, einstein.shape[-2:], window_type="cosine")

fig, axes = plt.subplots(1, 2, figsize=(8,4), layout="tight", sharex="all", sharey="all")
po.plot.imshow(einstein, ax=axes[0], title="target image")
model.plot_windows(ax=axes[0])
axes[0].xaxis.set_visible(False)
axes[0].yaxis.set_visible(False)

po.plot.imshow(einstein, ax=axes[1], title="target image")
model_cosine.plot_windows(ax=axes[1])
axes[1].xaxis.set_visible(False)
axes[1].yaxis.set_visible(False)
```

Now let's perform the synthesis for the cosine windows. Here we can clearly see the benefit of using the smoother gaussian windows (top row) which generates metamers that eliminate the sharper boundaries and ringing effect present from the cosine windows (bottom row).

```{code-cell} ipython3
model_cosine.eval()
po.remove_grad(model_cosine)
met_cosine = po.Metamer(einstein, model_cosine)
met_cosine.synthesize(max_iter=200, stop_criterion=1e-6);

po.plot.synthesis_status(met_einstein);
po.plot.synthesis_status(met_cosine);
```

These are just a few examples of how `fenestration` can interact with plenoptic's metamer synthesis, but there are many more ways these can be used! Feel free to play around with other {class}`~fenestration.PoolingWindows` parameters, other [synthesis methods](https://docs.plenoptic.org/docs/tags/2.1.0/api/synthesis.html) from plenoptic, or combining {class}`~fenestration.PoolingWindows` with other image processing methods, like steerable pyramids.
