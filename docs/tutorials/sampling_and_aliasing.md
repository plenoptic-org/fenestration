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

```{admonition} Run this notebook yourself!
:class: important

Download the executed notebook: **{nb-download}`sampling_and_aliasing.ipynb`**!

```

(sampling-aliasing-nb)=
# Sampling and Aliasing

A key feature of using pooling windows or other image sampling methods (e.g. strided convolutions) is ensuring that the sampling is appropriate. A sampling rate that is too low or that does not effectively match your data can result in aliasing, in which there are incorrect measurements in your reconstructed signal. This notebook shows some examples of the effect of sampling on aliasing. If you want to perform strided convolution, how far apart can your strides be while still avoiding aliasing? We examine this by seeing whether you can accurately interpolate to the function values at the unsampled pixels given different sampling rates.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import torch

import fenestration as fen

%load_ext autoreload
%autoreload 2
%matplotlib inline

# Animation-related settings
plt.rcParams["animation.html"] = "jshtml"
```

## Successful sampling

Here, we'll show an example of good sampling -- it's fine enough that we can successfully use linear interpolation to reconstruct the function centered at each pixel.

We'll use a simple Gaussian with a standard deviation of 1. We'll examine this function on a domain of `[-5, 5]` with values given at increments of `0.1`, and sample it at increments of `0.5`. Because this is a `torch` function, our `x` value should be `torch.linspace` (if it was a `numpy` function, `x` should be `np.linspace` instead).

Under the hood, this function uses `np.linalg.lstsq` to use regression and solve the interpolation.

```{code-cell} ipython3
x = torch.linspace(-5, 5, 101)
sampled, full, interps, coeffs, residuals = fen.sampling.check_sampling(0.5, x=x)
```

This could also be tested using pixels rather than values. For example, we could run the following code scaling all of the values by `10` to make more sense in pixel space. Therefore, we could use an image size of `(100,100)` and sample every `5` pixels. We also need to scale the `std_dev` of the `gaussian` function being used to sample.

```{code-cell} ipython3
x_pix = torch.linspace(-50, 50, 101)
fen.sampling.check_sampling(val_sampling=None, pix_sampling=5, x=x_pix, std_dev=10);
```

However, we will use the previous `val_sampling` method for our analyses moving forward. Let's look at the residuals, the error in each reconstruction. We can see there's problems at the boundaries, but that all the error is on the order of `1e-12`, which is pretty good!

```{code-cell} ipython3
plt.stem(residuals)
```

Let's look at some of the coefficients used to do this interpolation. We can see that some of these just have a `1` and `0` everywhere else -- this corresponds to interpolating the function to one of our sample points. Things get a little more complicated elsewhere.

```{code-cell} ipython3
fig = fen.sampling.plot_coeffs(coeffs[:20], 5)
```

Let's look at one of our interpolated functions, centered at `x=0`. We can see it looks pretty good, and the red dot shows us that the error is low as well.

```{code-cell} ipython3
fen.sampling.interpolation_plot(interps, residuals, None, 0, x);
```

Now let's look at a movie that shows each of these interpolations, with the actual values as a dotted line. We won't be able to tell the difference!

```{code-cell} ipython3
anim = fen.sampling.create_movie(interps, residuals, x, full=full)
plt.close()
anim
```

## Unsuccessful sampling

Alright, that was pretty boring, so let's do this whole thing again, but with sparser sampling and see that the interpolation does not look as good.

Let's double the spacing of our sampling, to every `2`.

```{code-cell} ipython3
x = torch.linspace(-5, 5, 101)
sampled, full, interps, coeffs, residuals = fen.sampling.check_sampling(2, x=x)
```

This looks much worse, but we note that it's periodic, going to down zero at each sample point.

```{code-cell} ipython3
plt.stem(residuals)
```

We won't look at our coefficients or anything here, let's jump right to the movie. We can see that the interpolation gets worse the farther away from the sample points you get, and it gets bad pretty quickly. It doesn't even look Gaussian anymore!

```{code-cell} ipython3
anim = fen.sampling.create_movie(interps, residuals, x, full=full)
plt.close()
anim
```
