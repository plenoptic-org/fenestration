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

(compare-window-types)=
# Comparing Gaussian and Cosine Windows

```{admonition} Warning
:class: warning

This notebook requires the optional dependency `plenoptic`, which can be installed with `pip`.
```

In this package, we support two different window types: raised cosine (used for the original implementation in [Freeman and Simoncelli, 2011](https://www.nature.com/articles/nn.2889)) and gaussian (used for a more recent implementation in [Broderick, Rufo, Winawer, & Simoncelli, 2023](https://elifesciences.org/reviewed-preprints/90554)). The authors of Broderick et al. found the Gaussian windows worked better for generating metamers when pooling luminance or spectral energy, whereas the authors of Freeman & Simoncelli used the raised-cosine windows for pooling texture statistics. The gaussian windows also generally support a smoother representation and minimal ringing and blocking artifacts in metamer synthesis, but we will compare the two window types here.

## Comparing Window Functions
First, let's see what these functions look like in 1d, using our built-in functions {meth}`~fenestration.pooling.raised_cosine` and {meth}`~fenestration.pooling.gaussian`.

```{code-cell} ipython3
import fenestration as fen
import matplotlib.pyplot as plt
import torch
import plenoptic as po
```


```{code-cell} ipython3
x = torch.linspace(-4, 4, 101)
cosine = fen.pooling.raised_cosine(x, transition_region_width=0.5)
gauss = fen.pooling.gaussian(x, std_dev=1)
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
ax[0].plot(x, cosine);
ax[0].set_title("Raised cosine function");
ax[1].plot(x, gauss);
ax[1].set_title("Gaussian function");

```

We can see that there are differences in the shapes of the functions, such that the raised cosine function is steeper and narrower with a higher amplitude whereas the gaussian function is wider with a smaller amplitude and no flat-top region.

As we will see below, these properties mean that, all else being equal, more Gaussian windows are needed to tile the image and that they give rise to smoother representations.

## Visualizing Window Types

Now let's see the actual windows projected onto a `(256,256)` sized image with `scaling=1`. Here we can also visualize how the two window types are built differently using the same scaling value.

```{code-cell} ipython3
pw_cosine = fen.PoolingWindows(1, (256,256), window_type="cosine")
pw_gauss = fen.PoolingWindows(1, (256,256), window_type="gaussian")

pw_cosine.plot_windows(subset=False);
pw_gauss.plot_windows(subset=False);
```

At the same scaling value, there are many more gaussian windows! It's important to note that the above contours represent `self.window_intersecting_amplitude` which is equivalent to FWHM for raised cosine windows, but is not the same for gaussian windows. In order to properly tile the input, more gaussian windows are needed in order to prevent sampling issues (see [](sampling-aliasing-nb), especially consider how we might get [](unsuccessful-sampling) with fewer gaussian windows).

## Matching FWHM for Comparison, Scaling=0.5

Let's find a pair of windows that approximately match in terms of FWHM (this requires a bit of trial and error to find corresponding indices) and compare the two window types. Also note the warnings that appear if windows are calculated to be smaller than a pixel at some scales!

```{code-cell} ipython3
pw_cosine = fen.PoolingWindows(0.5, (256, 256), max_ecc=13, num_scales=4, window_type="cosine")
pw_gauss = fen.PoolingWindows(0.5, (256, 256), max_ecc=13, num_scales=5, window_type="gaussian")

win_cosine = pw_cosine.ecc_windows[1][4]*pw_cosine.angle_windows[1][5]
win_gauss = pw_gauss.ecc_windows[1][11]*pw_gauss.angle_windows[1][12]
while win_cosine.ndim<4:
    win_cosine = win_cosine.unsqueeze(0)
    win_gauss = win_gauss.unsqueeze(0)
po.plot.imshow(torch.cat([win_cosine, win_gauss, win_cosine-win_gauss]), channel_idx=0, vrange='auto0', zoom=2);
```

### Computing Window Sizes, Scaling=0.5

Since the shapes are different, it's a bit difficult to decide whether we think they're the same size or not. Here we confirm that the computed angular and radial widths are pretty much the same.

```{code-cell} ipython3
cos_r = pw_cosine.window_width_pixels[1]['radial_half'][4]
cos_a = pw_cosine.window_width_pixels[1]['angular_half'][4]
gauss_a = pw_gauss.window_width_pixels[1]['angular_half'][11]
gauss_r = pw_gauss.window_width_pixels[1]['radial_half'][11]
print(f"Raised-cosine window:\n\tradial: {cos_r:.03f}\n\tangular: {cos_a:.03f}")
print(f"Gaussian window:\n\tradial: {gauss_r:.03f}\n\tangular: {gauss_a:.03f}")
```

The small difference is due to the fact that their centers aren't exactly in the same location.

```{code-cell} ipython3
cos_ctr = pw_cosine.central_eccentricity_pixels[1][4]
gauss_ctr = pw_gauss.central_eccentricity_pixels[1][11]
print(f"Raised-cosine window has center at {cos_ctr:.03f} pixels")
print(f"Gaussian window has center at {gauss_ctr:.03f} pixels")
```

Although these windows are approximately matched based on widths, the number and spacing of windows using these parameters will be different across window types since the relative spread (and therefore window overlap) still differs.

```{code-cell} ipython3
pw_cosine.plot_windows(subset=False);
pw_gauss.plot_windows(subset=False);

```

## Matching FWHM, Scaling=0.25

Now let's decrease the scaling and find another pair of windows that match in terms of FWHM and compare the two window types.

```{code-cell} ipython3
pw_cosine = fen.PoolingWindows(0.25, (256, 256), max_ecc=13, num_scales=4, window_type="cosine")
pw_gauss = fen.PoolingWindows(0.25, (256, 256), max_ecc=13, num_scales=5, window_type="gaussian")

win_cosine = pw_cosine.ecc_windows[1][10]*pw_cosine.angle_windows[1][11]
win_gauss = pw_gauss.ecc_windows[1][25]*pw_gauss.angle_windows[1][25]
while win_cosine.ndim<4:
    win_cosine = win_cosine.unsqueeze(0)
    win_gauss = win_gauss.unsqueeze(0)
po.plot.imshow(torch.cat([win_cosine, win_gauss, win_cosine-win_gauss]), channel_idx=0, vrange='auto0', zoom=2);
```

### Computing Window Sizes, Scaling=0.25

Again, since the centers are offset a bit, it is hard to determine whether the windows are the same size. However, we can see from the calculated widths that they are pretty much the same.

```{code-cell} ipython3
cos_r = pw_cosine.window_width_pixels[1]['radial_half'][4]
cos_a = pw_cosine.window_width_pixels[1]['angular_half'][4]
gauss_a = pw_gauss.window_width_pixels[1]['angular_half'][11]
gauss_r = pw_gauss.window_width_pixels[1]['radial_half'][11]
print(f"Raised-cosine window:\n\tradial: {cos_r:.03f}\n\tangular: {cos_a:.03f}")
print(f"Gaussian window:\n\tradial: {gauss_r:.03f}\n\tangular: {gauss_a:.03f}")

cos_ctr = pw_cosine.central_eccentricity_pixels[1][4]
gauss_ctr = pw_gauss.central_eccentricity_pixels[1][11]
print(f"Raised-cosine window has center at {cos_ctr:.03f} pixels")
print(f"Gaussian window has center at {gauss_ctr:.03f} pixels")
```

Again, since these windows are matched based on FWHM, the number and spacing of windows using these parameters will be different across window types. However, despite gaussian windows overlapping more, leading to smoother representations, the use of the models and scientific inferences made by [Freeman & Simoncelli, 2011](https://www.nature.com/articles/nn.2889) and [Broderick et al., 2023](https://elifesciences.org/reviewed-preprints/90554#tab-content) do not rely on the exact specifications of the windows. In this package, gaussian windows are the default, but you can try them both out yourself!
