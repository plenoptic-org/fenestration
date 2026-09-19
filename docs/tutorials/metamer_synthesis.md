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

Here we provide a tutorial for demonstrating how to use [plenoptic](https://plenoptic.org/) to generate model metamers for {class}`~fenestration.PoolingWindows`. By "model metamers", we refer to images that are physically different but have an identical representation for a given model. We will walk through examples of how different {class}`~fenestration.PoolingWindows` initialization arguments impact the resulting metamers. For additional examples of using this process for perceptual experiments, see [Broderick, W. F., Rufo, G., Winawer, J. & Simoncelli, E. P. (2023). Foveated metamers of the early visual system. eLife.](https://elifesciences.org/reviewed-preprints/90554)

```{code-cell} ipython3
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

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

We begin by using plenoptic's image [data extraction](https://docs.plenoptic.org/docs/tags/2.1.0/api/images.html), [processing](https://docs.plenoptic.org/docs/tags/2.1.0/api/process.html), and [plotting](https://docs.plenoptic.org/docs/tags/2.1.0/api/plot.html) functions to create and visualize two image tensors. Since these images are very different in terms of spatial frequency and structure, it will help us better understand this process. We also crop the images slightly in order to reduce memory consumption with the synthesis processes.

```{code-cell} ipython3
reptile_orig = po.data.reptile_skin()
einstein_orig = po.data.einstein()
reptile = po.process.center_crop(reptile_orig, 200)
einstein = po.process.center_crop(einstein_orig, 200)
po.plot.imshow([reptile_orig, einstein_orig]);
po.plot.imshow([reptile, einstein]);
```

## Visualizing Pooled Windows

To generate metamers, we first have to define a model which takes in a 4d tensor and performs some computation on the input (see plenoptic's [model requirements](https://docs.plenoptic.org/docs/tags/2.1.0/reference/models.html)). Metamer synthesis starts from some image (by default, a patch of uniform noise) and iteratively updates its pixels to minimize the error between the model output of the original image and that of the synthesized image. For {class}`~fenestration.PoolingWindows`, this process will try to minimize error between the pooled values of all windows for the original and synthesized images, resulting in two images whose average pixel values are matched over the spatial scale defined by the model.

Here, we define pooling windows with `scaling=0.8`. In the left figures, we have overlaid a wedge of pooling windows on the original images to visualize the intersection points of each pooling window that is averaging across the image. In the right figures, we use the {meth}`~fenestration.PoolingWindows.plot_window_values` method to displayed the pooled image values in each window that we will matching the metamer values to. You can see that the smaller, more central windows are averaging over a smaller region thus throwing away less information, while the more eccentric windows have a more blurred representation.

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

Now that we have reviewed how the {class}`~fenestration.PoolingWindows` model works, let's get to synthesizing some metamers. As required by plenoptic, we set our model to [evaluation mode](https://docs.pytorch.org/docs/2.14/notes/autograd.html#evaluation-mode-nn-module-eval), [remove gradients](https://docs.plenoptic.org/docs/tags/2.1.0/api/generated/plenoptic.remove_grad.html#plenoptic.remove_grad) from its parameters, initialize the [metamer object](https://docs.plenoptic.org/docs/tags/2.1.0/api/generated/plenoptic.Metamer.html#plenoptic.Metamer), and run the synthesis. We also use the [LBFGS optimizer](https://docs.pytorch.org/docs/2.14/generated/torch.optim.LBFGS.html), which seems to work better on non-deep net metamers compared to the default optimizer, [Adam](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html#torch.optim.Adam).

```{code-cell} ipython3
model.eval()
po.remove_grad(model)
met_reptile = po.Metamer(reptile, model)
met_reptile.setup(optimizer=torch.optim.LBFGS, optimizer_kwargs={"lr": 0.2})
met_reptile.synthesize(max_iter=400, stop_criterion=1e-6);
met_einstein = po.Metamer(einstein, model)
met_einstein.setup(optimizer=torch.optim.LBFGS, optimizer_kwargs={"lr": 0.2})
met_einstein.synthesize(max_iter=400, stop_criterion=1e-6);
```

It looks like the loss is starting to converge! Let's check out our metamers and plot the loss and error across each iteration using a [function from plenoptic](https://docs.plenoptic.org/docs/tags/2.1.0/api/generated/plenoptic.plot.synthesis_status.html#plenoptic.plot.synthesis_status).

```{admonition} Metamer synthesis success
:class: dropdown note

One could run metamer synthesis for longer here by setting `stop_criterion` to a lower value (Broderick et al., 2023 used `1e-9`), but the resulting model metamer doesn't appear to perceptually change beyond this point.

When synthesizing metamers, determining when optimization has successfully concluded is an important decision to make. See Broderick et al., 2023 and the plenoptic documentation ([here](https://docs.plenoptic.org/docs/tags/2.1.0/user_guide/reproduce/feather2023.html) and [here](https://docs.plenoptic.org/docs/tags/2.1.0/user_guide/models_and_metrics/portilla_simoncelli/technical_details/ps_optimization.html#good-enough)) for more information about the topic.

```

```{code-cell} ipython3
po.plot.synthesis_status(met_reptile);
po.plot.synthesis_status(met_einstein);
```

Since the model only cares about the average pixel values in each window, the metamer will have the same average pixel values in each window region as the original image, but the model does not "see" the noise in the generated metamer image. Also note how the similarity to the original image changes based on eccentricity: while the very center of the metamer looks like a noisy version of the original image, as the windows grow larger, more and more details are lost.

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

Also note the warning that we get now about some windows being smaller than a pixel! If the scaling value is small, it's possible for the smallest windows to be smaller than a pixel and not included in the windows, similar to the central region below `min_ecc`. You can access the minimum eccentricity at which windows are consistently larger than a pixel with the attribute `self.calculated_min_eccentricity_degrees`. We can zoom in and see the small windows here:

```{code-cell} ipython3
fig = po.plot.imshow(reptile, title=None)
model.plot_windows(ax=fig.axes[0], origin="upper");
plt.xlim(95,125);
plt.ylim(85,115);
```

Despite this, we can still generate our metamer because all pixels within the radius of `self.calculated_min_eccentricity_degrees` or contained in windows smaller than a pixel will be matched exactly between the metamer and the original image.

Since we are using smaller windows, we can see finer-grained details of the original image across a wider range of the metamer.

```{code-cell} ipython3
model.eval()
po.remove_grad(model)
met_reptile = po.Metamer(reptile, model)
met_reptile.setup(optimizer=torch.optim.LBFGS, optimizer_kwargs={"lr": 1})
met_reptile.synthesize(max_iter=200, stop_criterion=1e-6);
met_einstein = po.Metamer(einstein, model)
met_einstein.setup(optimizer=torch.optim.LBFGS, optimizer_kwargs={"lr": 1})
met_einstein.synthesize(max_iter=200, stop_criterion=1e-6);

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
met_cosine.setup(optimizer=torch.optim.LBFGS, optimizer_kwargs={"lr": 1})
met_cosine.synthesize(max_iter=200, stop_criterion=1e-6);

po.plot.synthesis_status(met_einstein);
po.plot.synthesis_status(met_cosine);
```

## Conclusion

These are just a few examples of how `fenestration` can interact with plenoptic's metamer synthesis, but there are many more ways these can be used! Feel free to play around with other {class}`~fenestration.PoolingWindows` parameters, other [synthesis methods](https://docs.plenoptic.org/docs/tags/2.1.0/api/synthesis.html) from plenoptic, or combining {class}`~fenestration.PoolingWindows` with other image processing methods, like [](steerable-pyramids-nb).
