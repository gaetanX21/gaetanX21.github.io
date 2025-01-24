---
layout: page
title: Diffusion Schrödinger Bridge
description: Theoretical study of the Schrödinger Bridge problem & PyTorch implementation of the Diffusion Schrödinger Bridge algorithm to study convergence properties in the Gaussian case.
img: assets/video/dsb/troll2torch_all.gif
importance: 1
category: work
related_publications: false
toc:
  sidebar: left
---


<div class="row justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/video/dsb/troll2torch_all.gif" title="DSB animation" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Gradual convergence of the IPF algorithm to find the schrödinger bridge from $p_\textnormal{pytorch}$ to $p_\textnormal{trollface}$ under Gaussian reference dynamics. Computed using DSB.
</div>

NB: This post is just a recap of my work, but you can get my full report <a href="https://github.com/gaetanX21/dsb-gaussian/blob/main/report/report.pdf">here</a>.

## Project Overview

Schrödinger Bridges (SB) generalize Optimal Transport by specifying not *where* but *how* to transport mass from one distribution to another, given a reference dynamic. Generative modeling can be achieved by finding SBs to go from $p_\textnormal{prior}$ to $p_\textnormal{data}$, which amounts to solving a *dynamic* SB problem. De Bortoli et al.[^dsb] introduced the Diffusion Schrödinger Bridge (DSB) model, a variational approach to the Iterative Proportional Fitting (IPF) algorithm to solve the *discrete dynamic* SB problem. DSB generalizes score-based generative modeling (SGM) introduced by Song et al[^sde]., and has stronger theoretical guarantees, in particular $p_T=p_\textnormal{prior}$.

This project constitutes a theoretical and practical introduction to DSB. Our contribution is to explicit the closed-form solution of the *discrete dynamic* SB problem in the Gaussian case, and leverage this closed-form expression to assess the performance of the DSB model in various settings by varying the dimension and complexity of $p_\textnormal{data}$ and $p_\textnormal{prior}$. In particular, we demonstrate that setting $L=20$ DSB iterations as in the original paper amounts to *under-training* the DSB model.


## The Schrödinger Bridge Problem

Let's first present the Schrödinger Bridge problem, which is the theoretical framework behind the DSB model. We will limit ourselves to the intuitive formulation of the SB problem from statistical physics.

The Schrödinger Bridge (SB) problem was introduced in 1932 by Schrödinger [^schrodinger] and asks the following question:
> Let $S$ be a particle system composed of a large number of independent particles following a Brownian motion. We observe $S\sim \nu_0$ at time $0$ and $S\sim \nu_1$ at time $T$. What is the most likely dynamical behavior of $S$ between $0$ and $T$ ?

<div class="row justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dsb/sb-statphy.png" title="SB as a statistical physics problem." class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    SB as a statistical physics problem.
</div>

In other words, the SB problem imposes an initial distribution $\nu_0$, a final distribution $\nu_1$ and a *reference dynamic* (here the Brownian motion i.e. diffusion according to the heat equation, but the SB problem can be generalized to any specified dynamics) and looks for the *distribution on paths* $P$ which matches $\nu_0$ at time $0$ and $\nu_1$ at time $T$ while staying as close as possible to the reference dynamics between $0$ and $T$.

Note that by setting $\nu_0=p_\textnormal{data}$ and $\nu_1=p_\textnormal{prior}$ we recover generative modeling.


## The Iterative Proportional Fitting Algorithm
To solve the SB problem, we use the Iterative Proportional Fitting (IPF) algorithm. It is a *discrete* algorithm, meaning that it approximates the *continuous* SB problem by discretizing the time interval $[0,T]$ into $N$ steps. The IPF algorithm is an alternative projection scheme w.r.t. to the Kullback-Leibler divergence which starts from an initial bridge $P_0$ and iteratively refines it according to the following two-step procedure:
1. **Forward step**: Given the bridge $P^{2n}: \mathbf{x} _ 0 \sim p _ \textnormal{data} \rightarrow \mathbf{x} _ T \sim p _ \textnormal{prior}$, compute the bridge $P^{2n+1}: \mathbf{x} _ T \sim p _ \textnormal{prior} \rightarrow \mathbf{x} _ 0 \sim p _ \textnormal{data}$ by starting from $\mathbf{x} _ T \sim p _ \textnormal{prior}$ and using the reverse transitions of $P^{2n}$.
2. **Backward step**: Given the bridge $P^{2n+1}: \mathbf{x} _ T \sim p _ \textnormal{prior} \rightarrow \mathbf{x} _ 0 \sim p _ \textnormal{data}$, compute the bridge $P^{2n+2}: \mathbf{x} _ 0 \sim p _ \textnormal{data} \rightarrow \mathbf{x} _ T \sim p _ \textnormal{prior}$ by starting from $\mathbf{x} _ 0 \sim p _ \textnormal{data}$ and using the reverse transitions of $P^{2n+1}$.
Under mild conditions on $p _ \textnormal{data}$ and $p _ \textnormal{prior}$, the IPF algorithm converges to a fixed point $S$ which is the (unique) solution to the SB problem.
Note that $P^0$ is obtained by starting from $p _ \textnormal{data}$ and using the transitions of the reference dynamic.

<div class="row justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dsb/ipfp_illustration.png" title="IPF" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The IPF algorithm amounts to iteratively solving half-bridges (i.e. projecting with respect to the KL divergence) until convergence to a fixed point denoted by $S$ here
</div>

The only difficulty in implementing IPF is that the reverse transitions are not known in closed-form. This is where the DSB model comes into play. Basically, DSB models all transitions as Gaussian kernels in such a way that the reverse transitions can be approximated in closed-form. However, this closed-form approximation still requires computing Stein scores (i.e. expressions of the form $\nabla _ \mathbf{x} \log p(\mathbf{x})$). We use score-matching to estimate these scores, yielding the DSB model.


## The Gaussian Case

In the Gaussian case (i.e. when both $p_\textnormal{data}$ and $p_\textnormal{prior}$ are Gaussian), there is a closed-form solution to the SB problem. This is useful because it means that we can apply DSB in the Gaussian case and compare the experimental results to the closed-form solution, so as to assess the performance of the DSB model.

Let $p_\textnormal{data} \sim N(0, \Sigma)$ and $p_\textnormal{prior} \sim N(0, \Sigma')$. Note that for conciseness we center on the origin, but it doesn't remove much generality since the hard part is learning the covariance.

The metric we use to assess the performance of the DSB model is the Frobenius norm of the difference $\hat{\Sigma} - \Sigma$ between the estimated covariance matrix $\hat{\Sigma}$ and the ground truth $\Sigma$. Note that the covariance matrix is estimated empirically from $M=250,000$ samples once the DSB model has converged.

To measure performance in different settings, we will vary the **dimension** of the Gaussian distributions ($d\in \lbrace 1,5, 10 \rbrace$), the **complexity** of the covariance matrices (spherical, diagonal or general), and the number of iterations of the DSB model (i.e. the number $L$ of IPF steps). In particular, we will show that setting $L=20$ DSB iterations as in the original paper amounts to *under-training* the DSB model. Note that for each setting we run $n_\textnormal{exp}=25$ experiments to average out the noise.

## Numerical findings

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dsb/sph_sigma.png" title="spherical" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dsb/dia_sigma.png" title="diagonal" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dsb/gen_sigma.png" title="general" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Numerical results obtained by training DSB in Gaussian case and comparing to ground truth.
</div>

The solid lines represent the mean error (computed over the $n_\textnormal{exp}$ experiments) as a function of the DSB iteration $n$, and the shaded areas depict the uncertainty over the estimated errors ($\pm1$std computed over the $n_\textnormal{exp}$ experiments). In all plots, the error increases with $d$ and decreases with $n$, as expected. Likewise, the variance of the error tends to diminish with $n$, which makes sense because each IPF step brings us closer to convergence.

We observe that the error follows the same decreasing pattern and the overall error levels follow the expected order of difficulty $spherical < diagonal < general$. In addition, despite the cogent shape of the error curve, the final error values (i.e. for $n=L$) are still one order of magnitude higher than the noise threshold, which implies that the IPF algorithm used by DSB is still *far from convergence*.

## Critics & Conclusion

There are a few critics regarding our results:
1. We limited ourselves to the Gaussian case, which is a strong assumption. However, it is a good starting point to understand the behavior of the DSB model.
2. For lack of sufficient computational resources, we had to limit ourselves to low dimensions. However DSB is meant to be applied in high dimensional settings such as image generation.
3. *A priori*, the error term $\| \hat{\Sigma}-\Sigma \|_F$ is contaminated with noise from the estimation of $\hat{\Sigma}$ and as such it overestimates the actual error, by an amount which is not trivial to determine, especially in higher dimensions.

Nonetheless, our results are consistent with the theoretical guarantees of the DSB model, and we have shown that setting $L=20$ DSB iterations as in the original paper amounts to *under-training* the DSB model. This is a first step towards understanding the convergence properties of the DSB model, and we hope that future work will address the critics we raised.


**References**:

[^dsb]: *Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling*. De Bortoli et al. [arXiv](https://arxiv.org/abs/2106.01357)
[^sde]: *Score-Based Generative Modeling through Stochastic Differential Equations*. Yang Song et al. [arXiv](https://arxiv.org/abs/2011.13456)
[^schrodinger]: *Sur la théorie relativiste de l’électron et l’interprétation de la mécanique quantique*. Erwin Schrödinger.