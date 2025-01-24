---
layout: page
title: Score-Based Generative Modeling
description: Theoretical study of Score-Based Generative Modeling & PyTorch implementation to compare Langevin, SDE and ODE sampling methods. Also explored controlled generation techniques, including conditional generation and inpainting.
img: assets/video/sde/pc.gif
importance: 1
category: work
related_publications: false
---
<div class="row justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/video/sde/pc.gif" title="PC sampling" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Realistic generated MNIST samples using Predictor-Corrector Sampling with Reverse SDE + Langevin Dynamics.
</div>

NB: This post is just a recap of my work, but you can get my full report <a href="https://github.com/gaetanX21/generative-sde/blob/main/report/report.pdf">here</a>.

NBB: For an awesome intro to the topic by the father of score-based generative modeling, check out <a href="https://yang-song.net/blog/2021/score/">this blog</a>.

## Project Overview

In 2020, Song et al. introduced a novel generative modeling framework[^score] in which samples are produced via Langevin dynamics using gradients from the data distribution. The gradients themselves are estimated using a technique known as denoising score matching, which was introduced in back in 2011 by Vincent[^score-denoising]. Shortly after introducing this new generative model, Song et al. proposed a generalization under the lens of stochastic differential equations[^sde].

Our contribution begins by summarizing and connecting the three aforementioned papers. Building on the work of Song et al., we construct a neural network (the score network $s_\theta(\mathbf{x},t)$) from scratch and train it on the MNIST dataset. Using this score network, we implement various sampling methods and compare them. Finally, we extend these methods to controlled generation, focusing on two types: conditional generation and inpainting.


## Why "score"?

Generative modeling is the task of learning an unknown distribution $p_\text{data}(\mathbf{x})$ from a dataset $\mathcal{D}$ of i.i.d. samples $\mathbf{x}_i$. The goal is to learn a generative model $p_\theta(\mathbf{x})$ such that $p_\theta(\mathbf{x}) \approx p_\text{data}(\mathbf{x})$. There are many ways to approach this problem. Perhaps the most natural way is to find $\theta$ that minimizes the Kullback-Leibler divergence between $p_\theta(\mathbf{x})$ and $p_\text{data}(\mathbf{x})$, which is equivalent to maximum likelihood estimation. However, this objective is often intractable for various reasons, the core one being that the KL divergence is too strong of a constraint. We need to relax it somehow.

This is where the score function comes in. The score function is defined as the gradient of the log-likelihood of the data distribution **w.r.t. the data itself**: $\nabla_\mathbf{x} \log p_\text{data}(\mathbf{x})$. Instead of minimizing the KL divergence, one can try to match the score functions of the data and model distributions. This is the approach taken by Song et al. and yields score-based generative modeling.

Per usual, we'll be using a neural network $s_\theta(\mathbf{x})$ to approximate the score function. The score network is trained to minimize the score matching loss

$$J^\text{naive}(\theta) = \frac{1}{2}\mathbb{E}_{p_\text{data}(\mathbf{x})} \left[ \left\| \mathbf{s}_\theta(\mathbf{x}) - \nabla_\mathbf{x} \log p_\text{data}(\mathbf{x}) \right\|^2 \right]$$

However, this objective is obviously intractable since it involves the score function of the data distribution. To circumvent this issue, we can use denoising score matching, which replaces the score function of the data distribution with the score function of a noisy version of the data. The loss becomes

$$J^\text{denoising}(\theta) = \frac{1}{2} \mathbb{E}_{\mathbf{x}\sim p_\text{data}(\mathbf{x}),\tilde {\mathbf{x}}\sim q_\sigma(\tilde  {\mathbf{x}}|\mathbf{x})} \left[ \left\| \mathbf{s}_\theta(\tilde {\mathbf{x}}) - \nabla_{\tilde {\mathbf{x}}} \log q_\sigma(\tilde {\mathbf{x}}|\mathbf{x}) \right\|^2 \right]$$

The intuition behind this objective is that, given a noisy version of the data, the score network should point towards 
the original data point. Indeed, if we take an isotropic Gaussian noise distribution for $q_\sigma(\tilde {\mathbf{x}}|\mathbf{x})$,
we find that $\nabla_{\tilde {\mathbf{x}}} \log q_\sigma(\tilde {\mathbf{x}}|\mathbf{x}) = \frac{\mathbf{x} - \tilde {\mathbf{x}}}{\sigma^2}$, such that the score network is trained to point towards the original data point, i.e., to denoise the data.

We can then use gradient descent to optimize the empirical version of $J^\text{denoising}$ (over $\mathcal{D}$), which gives us $\theta^\star$ and thus the score network $s_{\theta^\star}(\mathbf{x})$. Let's now see how we can use this score network to generate samples.

## Discrete-time sampling: Langevin Dynamics



## Continuous-time sampling: SDEs (and ODEs!)

## Predictor-Corrector Sampling

## Controlled Generation



**References**:

[^sde]: *Score-Based Generative Modeling through Stochastic Differential Equations*. Yang Song et al. [arXiv](https://arxiv.org/abs/2011.13456)
[^score]: *Generative Modeling by Estimating Gradients of the Data Distribution*. Yang Song & Stefano Ermon. [arXiv](https://arxiv.org/abs/1907.05600)
[^score-denoising]: *A Connection Between Score Matching and Denoising Autoencoders*. Pascal Vincent. [iro.umontreal.ca](https://www.iro.umontreal.ca/~vincentp/Publications/DenoisingScoreMatching_NeuralComp2011.pdf)
[^anderson]: *Reverse-time diffusion equation models*. O. G. Anderson. [sciencedirect.com](https://www.sciencedirect.com/science/article/pii/0304414982900515)
[^ddpm]: *Denoising Diffusion Probabilistic Models*. Ho et al. [arXiv] (https://arxiv.org/abs/2006.11239)