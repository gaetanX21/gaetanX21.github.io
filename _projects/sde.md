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

## Project Overview

In 2020, Song et al. introduced a novel generative modeling framework[^score] in which samples are produced via Langevin dynamics using gradients from the data distribution. The gradients themselves are estimated using a technique known as denoising score matching, which was introduced in back in 2011 by Vincent[^score-denoising]. Shortly after introducing this new generative model, Song et al. proposed a generalization under the lens of stochastic differential equations[^sde].

Our contribution begins by summarizing and connecting the three aforementioned papers. Building on the work of Song et al., we construct a neural network (the score network $s_\theta(\mathbf{x},t)$) from scratch and train it on the MNIST dataset. Using this score network, we implement various sampling methods and compare them. Finally, we extend these methods to controlled generation, focusing on two types: conditional generation and inpainting.



**References**:

[^sde]: *Score-Based Generative Modeling through Stochastic Differential Equations*. Yang Song et al. [arXiv](https://arxiv.org/abs/2011.13456)
[^score]: *Generative Modeling by Estimating Gradients of the Data Distribution*. Yang Song & Stefano Ermon. [arXiv](https://arxiv.org/abs/1907.05600)
[^score-denoising]: *A Connection Between Score Matching and Denoising Autoencoders*. Pascal Vincent. [iro.umontreal.ca](https://www.iro.umontreal.ca/~vincentp/Publications/DenoisingScoreMatching_NeuralComp2011.pdf)
[^anderson]: *Reverse-time diffusion equation models*. O. G. Anderson. [sciencedirect.com](https://www.sciencedirect.com/science/article/pii/0304414982900515)
[^ddpm]: *Denoising Diffusion Probabilistic Models*. Ho et al. [arXiv] (https://arxiv.org/abs/2006.11239)