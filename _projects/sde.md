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
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/sde/diffusion_schematic.png" title="Diffusion" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Score-based generative modeling is the reversal of a diffusion SDE. Source: Song et al. paper.
</div>

NB: This post is just a recap of my work, but you can get my full report <a href="https://github.com/gaetanX21/generative-sde/blob/main/report/report.pdf">here</a>.

NBB: For an awesome intro to the topic by the father of score-based generative modeling, check out <a href="https://yang-song.net/blog/2021/score/">this blog</a>.


## Project Overview

In 2020, Song et al. introduced a novel generative modeling framework[^score] in which samples are produced via Langevin dynamics using gradients from the data distribution. The gradients themselves are estimated using a technique known as denoising score matching, which was introduced in back in 2011 by Vincent[^score-denoising]. Shortly after introducing this new generative model, Song et al. proposed a generalization under the lens of stochastic differential equations[^sde].

Our contribution begins by summarizing and connecting the three aforementioned papers. Building on the work of Song et al., we construct a neural network (the score network $s_\theta(\mathbf{x},t)$) from scratch and train it on the MNIST dataset. Using this score network, we implement various sampling methods and compare them. Finally, we extend these methods to controlled generation, focusing on two types: conditional generation and inpainting.


## Why "score"?

Generative modeling is the task of learning an unknown distribution $p _ \text{data}(\mathbf{x})$ from a dataset $\mathcal{D} = \lbrace \mathbf{x} _ i \rbrace_{1\leq i \leq N}$ of i.i.d. samples. The goal is to learn a generative model $p_\theta(\mathbf{x})$ such that $p _ \theta(\mathbf{x}) \approx p _ \text{data}(\mathbf{x})$. There are many ways to approach this problem. Perhaps the most natural way is to find $\theta$ that minimizes the Kullback-Leibler divergence between $p_\theta(\mathbf{x})$ and $p_\text{data}(\mathbf{x})$, which is equivalent to maximum likelihood estimation. However, this objective is often intractable for various reasons, the core one being that the KL divergence is too strong of a constraint. We need to relax it somehow.

This is where the score function comes in. The score function is defined as the gradient of the log-likelihood of the data distribution **w.r.t. the data itself**: $\nabla_\mathbf{x} \log p_\text{data}(\mathbf{x})$. Instead of minimizing the KL divergence, one can try to match the score functions of the data and model distributions. This is the approach taken by Song et al. and yields score-based generative modeling.

Per usual, we'll be using a neural network $s_\theta(\mathbf{x})$ to approximate the score function. The score network is trained to minimize the score matching loss

$$J^\text{naive}(\theta) = \frac{1}{2}\mathbb{E}_{p_\text{data}(\mathbf{x})} \left[ \left\| \mathbf{s}_\theta(\mathbf{x}) - \nabla_\mathbf{x} \log p_\text{data}(\mathbf{x}) \right\|^2 \right]$$

However, this objective is obviously intractable since it involves the score function of the data distribution. To circumvent this issue, we can use denoising score matching, which replaces the score function of the data distribution with the score function of a noisy version of the data. The loss becomes

$$J^\text{denoising}(\theta) = \frac{1}{2} \mathbb{E}_{\mathbf{x}\sim p_\text{data}(\mathbf{x}),\tilde {\mathbf{x} }\sim q_\sigma(\tilde  {\mathbf{x} }|\mathbf{x})} \left[ \left\| \mathbf{s}_\theta(\tilde {\mathbf{x} }) - \nabla_{\tilde {\mathbf{x} } } \log q_\sigma(\tilde {\mathbf{x} }|\mathbf{x}) \right\|^2 \right]$$

The intuition behind this objective is that, given a noisy version of the data, the score network should point towards 
the original data point. Indeed, if we take an isotropic Gaussian noise distribution for $q_\sigma(\tilde {\mathbf{x} }|\mathbf{x})$,
we find that $\nabla_{\tilde {\mathbf{x} } } \log q_\sigma(\tilde {\mathbf{x} }|\mathbf{x}) = \frac{\mathbf{x} - \tilde {\mathbf{x} } }{\sigma^2}$, such that the score network is trained to point towards the original data point, i.e., to denoise the data.

We can then use gradient descent to optimize the empirical expression of $J^\text{denoising}$ over $\mathcal{D}$, which gives us $\theta^\star$ and thus the score network $s_{\theta^\star}(\mathbf{x})$.

However, one question has been left unanswered: how do we pick the noise $\sigma$? If too small, $s_{\theta}(\mathbf{x})$ will be poorly approximated in low-density regions and thus worthless for sampling. If too large, $s_{\theta}(\mathbf{x})$ will be too far from the true score function. The trick is to use a schedule for $\sigma$ that starts large and decreases over time. The only change is to make the score network noise-conditional and train it across various noise levels $\lbrace \sigma_t \rbrace_{t=1}^T$.

## Discrete-time sampling: Langevin Dynamics

Langevin Dynamics is a Markov Chain Monte Carlo (MCMC) procedure to sample from the data distribution.

Given a fixed step size $\epsilon>0$ and an initial value $\tilde {\mathbf{x} }_0 \sim \pi(\mathbf{x})$ where $\pi$ is a tractable prior distribution (e.g. a Gaussian), the Langevin method recursively computes

$$\tilde {\mathbf{x} }_{t} = \tilde {\mathbf{x} }_{t-1} + \frac{\epsilon}{2} \nabla_\mathbf{x} \log p(\tilde {\mathbf{x} }_{t-1}) + \sqrt{\epsilon}\mathbf{z}_t$$

where $\mathbf{z}_t \sim \mathcal{N}(0, \mathbf{I})$.

The distribution of $\tilde {\mathbf{x} }_T$ converges to the data distribution $p(\mathbf{x})$ as $\epsilon \to 0$ and $T \to \infty$. In practice, $\epsilon>0$ and $T<\infty$, which creates an error in the sampling process, but we can safely assume that this error is small enough to be ignored.

Note that since the noise is scheduled, we are in fact doing **annealed** Langevin dynamics.


<div class="row justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/video/sde/langevin.gif" title="langevin sampling" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Langevin sampling on MNIST.
</div>

## Continuous-time sampling: Stochastic Differential Equations

The Langevin dynamics method presented above works well in practice and scales up nicely to higher dimensions. In effect, we have learned how to gradually noise and gradually denoise our data, through discrete noise levels $\sigma_t$ in the schedule.
However, Song et al. found shortly after their first publication that this discrete noising process could be subsumed into a continous-time stochastic differential equation (SDE) framework. This is very interesting because we know from Anderson[^anderson] that reversing the time in a diffusion process yields another diffusion process! Indeed, the diffusion process

$$d\mathbf{x}_t = \mathbf{f}(\mathbf{x}_t,t)dt + g(t)d\mathbf{w}_t$$

can be reversed into

$$d\mathbf{x}_t = \left[ \mathbf{f}(\mathbf{x}_t,t) - g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x}_t) \right]dt + d\mathbf{\bar{w} }_t$$

where $p_t(\mathbf{x})$ is the density of the process at time $t$ (which is given by the Fokker-Planck equation) and $\mathbf{\bar{w} }_t$ is a Wiener process when time flows backwards from $T$ to $0$. (crucially, $dt>0$ in the above equation)

Thus, by starting from $\mathbf{x} _ T \sim p _ T (\mathbf{x})$ and running the reverse process, we can obtains samples $\mathbf{x} _ 0 \simeq p _ 0(\mathbf{x})$. This is the idea behind SDE-based generative modeling. Importantly, the only ingredient needed to reverse the process is the score function $\nabla_\mathbf{x} \log p _ t(\mathbf{x})$. We can approximate this score function using a time-dependent neural network $s_\theta(\mathbf{x},t)$, which is the continuous-time equivalent of the noise-conditional score network we used in the discrete-time case.

Once the time-dependent score network is trained, we can sample from the SDE using the any black-box SDE solver.


<div class="row justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/video/sde/reverse_sde.gif" title="sde sampling" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Reverse SDE sampling on MNIST.
</div>

## Predictor-Corrector Sampling

In a nutshell, the idea behing Predictor-Corrector (PC) sampling is to use both the Langevin and SDE sampling methods to generate samples. Each step of the SDE is followed by a step of Langevin dynamics to "correct" the sample $\mathbf{x}_t$ obtained from the SDE. In practice, this combination of approaches yields better samples than either method alone. Song et. al achieved state-of-the-art results on the CIFAR-10 dataset using PC sampling.[^sde]

<div class="row justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/video/sde/pc.gif" title="pc sampling" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    PC sampling yields superior samples compared to Langevin or SDE sampling alone.
</div>


## Ordinary Differential Equation (ODE) Sampling

In [^score], Song et al. demonstrate that any SDE can be converted into an ODE with the same marginal distributions $\{ p _ t(\mathbf{x}) \}_{t\in[0,T]}$ (albeit not the same joint distributions $p(\mathbf{x _ {0:T}})$). The ODE of an SDE is called **probability flow ODE** and is given by

$$d\mathbf{x}(t) = \left[ \mathbf{f}(\mathbf{x}(t), t) - \frac{1}{2}g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x}(t)) \right]dt$$

Once again, we only need to be able to compute the score function $\nabla_\mathbf{x} \log p_t(\mathbf{x})$ to sample from the ODE. This is done using the time-dependent score network $s_\theta(\mathbf{x},t)$. The ODE is then solved using a black-box ODE solver. Importantly, probability flow ODEs are a special case of neural ODEs and as such they allow for exact log-likelihood computation, which is a major advantage over Langevin and SDE sampling methods.

<div class="row justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/video/sde/ode.gif" title="ode sampling" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    ODE sampling is notably smoother than Langevin or SDE sampling since it is a deterministic process.
</div>

## Controlled Generation

So far, we have described several methods for generating random samples from the data distribution. However, the diffusion approach is flexible and can easily be tweaked to control the generation process. The idea is the following: if we denote by $\mathbf{y}$ the conditions we want to impose on the generated samples, we aim to sample from the conditional distribution $p _ \text{data} ( \mathbf{x} \mid \mathbf{y} )$.

Starting with random noise, we reverse through time using either the Langevin, SDE or ODE method, adjusting the process to use $\nabla_{\mathbf{x} } \log p _ t ( \mathbf{y} \mid \mathbf{x} )$ instead of $\nabla_{\mathbf{x} } \log p _ t ( \mathbf{x} )$. Intuitively, this approach reverts the diffusion of $\lbrace \mathbf{x} _ t \mid \mathbf{y} \rbrace _ {t \in [0,T]}$ instead of $\lbrace \mathbf{x} _ t \rbrace _ {t \in [0,T]}$, meaning our final sample will satisfy $\mathbf{x} _ 0 \sim p _ 0 ( \mathbf{x} \mid \mathbf{y} )$ as desired.

To illustrate, we implemented this idea in two key applications: class-conditional generation and inpainting. The details are in the full report, but the main idea is to condition the score network on the class label or the masked pixels, respectively. The results are quite satisfying, as shown below.

<div class="row justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/video/sde/conditional.gif" title="conditional sampling" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Class-conditional generation using PC sampling and setting $y=4$.
</div>

<div class="row justify-content-center">
    <div class="col-sm-3 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/sde/inpainting_masked.png" title="inpainting masked" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-3 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/sde/inpainting_recovered.png" title="inpainting recovered" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Left: masked MNIST samples. Right: after inpainting.
</div>


**References**:

[^sde]: *Score-Based Generative Modeling through Stochastic Differential Equations*. Yang Song et al. [arXiv](https://arxiv.org/abs/2011.13456)
[^score]: *Generative Modeling by Estimating Gradients of the Data Distribution*. Yang Song & Stefano Ermon. [arXiv](https://arxiv.org/abs/1907.05600)
[^score-denoising]: *A Connection Between Score Matching and Denoising Autoencoders*. Pascal Vincent. [iro.umontreal.ca](https://www.iro.umontreal.ca/~vincentp/Publications/DenoisingScoreMatching_NeuralComp2011.pdf)
[^anderson]: *Reverse-time diffusion equation models*. O. G. Anderson. [sciencedirect.com](https://www.sciencedirect.com/science/article/pii/0304414982900515)
[^ddpm]: *Denoising Diffusion Probabilistic Models*. Ho et al. [arXiv] (https://arxiv.org/abs/2006.11239)