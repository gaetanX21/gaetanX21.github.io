---
layout: post
title: "Jeffreys' Prior in Bayesian Inference"
date: 2025-02-07
description: "TL;DR: Bayesian inference requires us to specify a prior distribution. When we're unsure what prior to pick and want to stay as objective as possible, one option is to use Jeffreys' prior, which leverages the Fisher information to provide a reparametrization-invariant prior."
tags: bayesian-ml
thumbnail: assets/img/posts/jeffreys_prior/beta.png
---

$$
\newcommand{\E}{\mathbb{E}}
\newcommand{\Var}{\text{Var}}
\newcommand{\Cov}{\text{Cov}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\mathcalL}{\mathcal{L}}
$$

We first motivate the need for "objective" priors in Bayesian inference by highlighting the limitations of uniform priors. We then introduce Jeffreys' prior, which is invariant under reparametrization and provides a principled way to assign priors in Bayesian inference. We prove its invariance under reparametrization and illustrate its use in a coin flip problem. Note that throughout this post we restrict ourselves to the one-dimensional case for simplicity.

## Introduction

In Bayesian inference, prior distributions encode our initial beliefs about an unknown parameter $\theta$ before observing data $x$. We can then update these beliefs using Bayes' theorem to obtain a posterior distribution. Namely: *posterior = likelihood x prior*, which can be rewritten as $p(\theta \| x) \propto p(x \| \theta) p(\theta)$.

Choosing priors usually involves a trade-off between incorporating prior knowledge and maintaining objectivity. Depending on the context and how much we know about the problem, we might have different beliefs about the parameter, or no beliefs at all. For instance, if we're doing linear regression on standardized data ($y _ i = \beta^T x _ i + \varepsilon _ i$), we may feel like our prior for $\beta$ should be centered around zero. But if we're doing a coin flip experiment ($X_i \sim B(\theta)$), we might not have any strong prior beliefs about the bias of the coin. So how do we choose the prior in this case? One naive approach would be to use a flat prior $p(\theta) \sim U([0, 1])$. This prior seems uninformative but it really isn't. To see why, let's consider the same coin flip experiment but this time we want to estimate the odds ratio $\phi = \frac{\theta}{1 - \theta}$. We may again naively choose a flat prior $p(\phi) \propto 1$[^improper]. But this flat prior on $\phi$ induces a non-flat prior on $\theta$! In fact, since $\phi$ is uniform on $\R_+$[^improper] and as such biased towards arbitrarily large values, $\theta = \frac{\phi}{1 + \phi}$ is highly biased towards $1$, as illustrated on [Figure 1](#fig-1). Thus, choosing a flat prior for $\phi$ is not the same as choosing a flat prior for $\theta$! That is why the seemingly objective choice of a flat prior is not always the best choice.

<div class="row justify-content-center" id="fig-1">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/jeffreys_prior/theta.png" title="theta" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 1. Sketch of the prior distribution on $\theta$ induced by a flat prior on $\phi=\frac{\theta}{1-\theta}$. Clearly, the prior is not flat and is biased towards $\theta=1$.
</div>

Likewise, choosing a flat prior in high-dimensional spaces assigns way too much mass to unimportant regions of the parameter space, so it is informative, but in a bad way!

With this in mind, we see that **uniform priors are no silver bullet**. Ideally, we would like a prior which does not depend on the parameterization of the problem. In other word, **the information we encode in the prior should be invariant under reparametrization**. If we go back to the example of the coin flip, we would like a prior that encodes the same prior information about the bias of the coin, regardless of whether we're working with $\theta$ or $\phi$. Intuitively, such a prior should be based on the **structure of the data model** itself, rather than the parameterization we choose.

Reparametrization invariance is exactly what Jeffreys' prior achieves, as explained below.

*Note that intuitively, reparametrization invariance is a good heuristic for an "objective" prior.*

## Definition

Jeffreys' prior is defined using the Fisher information matrix. Given a likelihood function $\mathcalL(\theta \| x)$ for a parameter $\theta$, the Fisher information is:

$$
I(\theta) = \E \left[ \left( \frac{\partial}{\partial \theta} \log \mathcalL(\theta | x) \right)^2 \bigg| \theta \right].
$$

Jeffreys' prior is then given by:

$$
\pi_J(\theta) \propto \sqrt{I(\theta)}.
$$

The key property of Jeffrey's prior is that it is invariant under reparametrization. In other words, if we try to estimate a different parameter $\phi = g(\theta)$, the Jeffrey's prior for $\phi$ will be:

$$
\pi_J(\phi) \propto \sqrt{I(\phi)} = \pi_J(\theta) \left| \frac{d\theta}{d\phi} \right|
$$

which is consistent with the transformation rule for probability densities.

*Note that Jeffrey's prior is defined using the likelihood function. While this is convenient because it allows us to use the structure of the data model, it also goes against the Bayesian principle of choosing the prior independently of the data. This is a philosophical issue in Bayesian statistics, and different practitioners may have different views on this.*

## Proof of Invariance Under Reparametrization

In this paragraph we demonstrate Jeffreys' prior invariance under reparametrization. Suppose we have a parameter $\theta$ and a reparametrized parameter $\phi = g(\theta)$. We want to show that Jeffrey's prior for $\phi$ is consistent with the transformation rule for probability densities.

To begin with, note that the chain rule gives:

$$
I(\phi) = I(\theta) \left( \frac{d\theta}{d\phi} \right)^2.
$$

Taking the square root, we get:

$$
\sqrt{I(\phi)} = \sqrt{I(\theta)} \left| \frac{d\theta}{d\phi} \right|
$$

i.e., Jeffrey's prior transforms as:

$$
\pi_J(\phi) = \pi_J(\theta) \left| \frac{d\theta}{d\phi} \right|.
$$

We recognize the transformation rule for probability densities, which demonstrates that Jeffrey’s prior correctly transforms to maintain consistency, proving its invariance by reparametrization.

## Coin flip example

Let's compute Jeffreys' prior for a simple coin flip problem to illustrate its use.

Consider a simple example: estimating the bias $\theta$ of a coin, where $X \sim \text{Bin}(n, \theta)$. The likelihood function is:

$$
\mathcal L(\theta | x) = \prod_{i=1}^n \theta^{x_i} (1 - \theta)^{1-x_i} = \theta^{\sum x_i} (1 - \theta)^{n - \sum x_i}.
$$

We compute the Fisher information:

$$
I(\theta) = \E \left[ \left( \frac{\partial}{\partial \theta} \log \mathcalL(\theta | x) \right)^2 \bigg| \theta \right] = \frac{n}{\theta (1 - \theta)}.
$$

Thus, Jeffreys' prior for $\theta$ is:

$$
\pi_J(\theta) \propto \sqrt{\frac{n}{\theta (1 - \theta)}} \propto \frac{1}{\sqrt{\theta (1 - \theta)}}.
$$

We recognize the **Beta(1/2, 1/2)** distribution, which is commonly used as an uninformative prior for bounded parameters. This is a nice result, as it shows that Jeffreys' prior is consistent with our intuition of an uninformative prior in this case.

<div class="row justify-content-center" id="fig-2">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/jeffreys_prior/beta.png" title="beta" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 2. Jeffreys' prior for the bias of a coin flip experiment is the Beta(1/2, 1/2) distribution.
</div>

## Conclusion

Jeffreys' prior provides a principled way to assign priors in Bayesian inference, ensuring invariance under reparametrization. We proved its reparametrization invariance and illustrated its use in a coin flip problem. Jeffreys' prior is useful when no clear subjective prior information is available, for instance in astrophysics. We've limited ourselves to the one-dimensional case for simplicity, but Jeffreys' prior can be extended to higher dimensions naturally by considering the Fisher information matrix and its determinant, such that $\pi_J(\theta) \propto \sqrt{\text{det}(I(\theta))}$. Finally, I want to stress that Jeffreys' prior violates the Bayesian principle of choosing the prior independently of the data, which may be a concern for some practitioners.

---

**Notes**:

[^improper]: The prior $p(\phi) \propto 1$ is called an *improper* prior since it doesn't integrate to 1. This is a common pitfall when using flat priors. However using unnormalized priors is okay as long as we normalize the posterior distribution.