---
layout: post
title: "A geodesic from cat to dog"
date: 2024-11-16
description: "TL;DR: Entropic regularization relaxes the Kantorovitch problem into a strictly convex problem which can be solved efficiently with the Sinkhorn algorithm. We can use this to efficiently compute Wasserstein distances, barycenters, and finally geodesics between distributions."
tags: optimal-transport
thumbnail: assets/img/posts/ot_geodesic/cat2dog.png
---

$$
\newcommand{\R}{\mathbb{R}}
\newcommand{\tn}[1]{\textnormal{#1}}
$$

We first introduce the discrete entropy-regularized Kantorovich problem and show how it can be solved efficiently using the Sinkhorn algorithm. We then illustrate the usefulness of the Sinkhorn algorithm to compute Wasserstein distances, barycenters, and geodesics between probability distributions. We finally apply this to interpolate between grayscale images of a cat and a dog, effectively computing a geodesic in the space of grayscale $N\times N$ images for the 2-Wasserstein metric.

# Discrete Entropy-Regularized Kantorovich problem
The discrete entropy-regularized Kantorovich problem formulates as:
$$
\begin{equation}
    \label{eq:Kreg}
    \tag{$\tn{K}^\tn{reg}$}
    P^{\epsilon,\star}= \arg \min _ {P\in U(\alpha,\beta)} \langle C,P\rangle - \epsilon H(P)
\end{equation}
$$
where $H(P)=\sum _ {i=1}^nP_{ij}(\log P _ {ij}-1)$ is the discrete entropy. Note that when $\epsilon=0$ one recovers classical discrete OT. Crucially, (\ref{eq:Kreg}) is strictly convex as soon as $\epsilon>0$ and thus has a unique solution $P^{\epsilon,\star}$.

Additionally, One can easily show that $\langle C,P \rangle - \epsilon H(P)= \tn{KL} (P\|K)$, where $K=\exp(-\frac{C}{\epsilon})$ is called a Gibbs kernel. Thus (\ref{eq:Kreg}) can be seen as a projection problem w.r.t. to the KL divergence: 
(\ref{eq:Kreg}) rewrites as $P^{\epsilon,\star}= \arg \min _ {P\in U(\alpha,\beta)} \tn{KL}(P\|K)$ i.e. $P^{\epsilon,\star}=\tn{Proj} _ {U(\alpha,\beta)}^\tn{KL}(K)$.

The whole point of introducing the entropy is to relax the Kantorovich problem into a strictly convex problem which can be solved efficiently using the Sinkhorn algorithm, which we now introduce.

## Sinkhorn's algorithm
The most well-known method to solve (\ref{eq:Kreg}) is Sinkhorn's algorithm, which uses the fact that $P^{\epsilon,\star}$ necessarily has the form $P^{\epsilon,\star}=\tn{Diag}(u)K\tn{Diag}(v)$ where $K$ is the Gibbs kernel. The conditions $P\mathbb{1} _ m=a$ and $P^T\mathbb{1} _ n=b$ thus rewrite as $u * (Kv) = a$ and $v * (K^Tu) = b$ respectively, where $*$ denotes the Hadamard product. One can thus iteratively solve these two equations until $u$ and $v$ converge, yielding the Sinkhorn algorithm:

$$
\begin{align*}
    u^{l+1}&\leftarrow\frac{a}{Kv^l}\\
    v^{l+1}&\leftarrow\frac{b}{K^Tu^{l+1}}
\end{align*}
$$

where we use the initialization $v^0=\mathbb{I} _ m$.

In practice, Sinkhorn's algorithm allows us to compute Wasserstein distances efficiently. In turn, we can use these distances to compute barycenters and geodesics between probability distributions.

## Wasserstein barycenters and geodesics on probability spaces
Using OT, one can define *distances* between probability distributions defined on the same space $X$. The most common is the $p$-Wasserstein distance $W_p$ defined for any real number $p>0$:
$$
\begin{equation}
    \label{eq:Wp}
    \tag{$W_p$}
    W_p(\alpha,\beta)=\min_{P\in U(\alpha,\beta)} \langle P,C^p \rangle^{1/p} = \bigg(\sum_{1\leq i,j\leq n} d(x_i,y_j)^p P_{ij}\bigg)^{1/p}
\end{equation}
$$
where $d$ is a distance on $X$. For instance if $X=\R^d$ one can use $d(x,y)=||x-y||$.

Now that we have a distance on probability measures, we can use it to compute barycenters. For a fixed $p>1$ and $R$ probability distributions $\alpha_1,\dots,\alpha_R \in \tn{P}(X)$, their $p$-Wasserstein barycenter with coefficients $(\lambda_r)_r$ is defined as:
$$
\begin{equation}
    \label{eq:barycenter}
    \tag{B}
    \beta = \arg \min_{\beta \in \tn{P}(X)} \sum_{r=1}^{R} \lambda_k W_p(\alpha_r,\beta)
\end{equation}
$$

$p$-Wasserstein barycenters can in particular be used to compute geodesics for the $p$-Wasserstein metric of the form $t\in[0,1]\mapsto\mu_t\in\tn{P}(X)$ from $\alpha$ to $\beta$ as:
$$
\begin{equation}
    \mu_t = \arg \min_{\mu\in\tn{P}(X)} (1-t)W_p^p(\alpha,\mu_t) + tW_p^p(\beta,\mu_t)
\end{equation}
$$
When $p=2$, we in fact have $\mu_t=\sum_{1\leq i,j\leq n}P_{ij}^\star \delta_{(1-t)x_i+ty_j}$ using the notations from my [previous post](/blog/2024/ot-assignement-problem/).

## A geodesic from cat to dog
Wasserstein barycenters can be used to interpolate between (grayscale) images using the following formalism: a grayscale image of dimension $N\times N$ can be seen as a distribution of "light" $\alpha\in \tn{P}(\R^{N\times N})$. Then, one can go from an image of a cat $\alpha$ to that of a dog $\beta$ using OT. This has little value in itself, but one can also consider the geodesic $\mu^{\tn{cat}\rightarrow \tn{dog}}$ from $\alpha$ to $\beta$ and thus see the gradual fade from the cat image to the dog image.

For $0\leq i \leq 8$ we consider the barycenter coefficients $\lambda=(1-t_i,t_i)$ where $t_i=\frac{i}{8}$ and we plot the 9 corresponding 2-Wasserstein barycenters $b^i\in\tn{P}(\R^{N\times N})$ which intuitively interpolate between the cat and the dog. The pictures were found online, turned to grayscale, resized to $N\times N$ with $N=128$, smooth with a Gaussian kernel, and then each Wasserstein barycenter is computed using the `ot` Python library. The results are presented in [Figure 1](#fig-1) and quite satisfying for such a simple approach!

<div class="row justify-content-center" id="fig-1">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/ot_geodesic/cat2dog.png" title="cat2dog" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 1. Geodesic $\mu^{\tn{cat}\rightarrow \tn{dog}}$ in $\R^{N\times N}$ computed using 2-Wasserstein barycenters.
</div>

## Conclusion

We have shown that the entropy-regularized Kantorovich problem can be solved efficiently using the Sinkhorn algorithm. This allows us to efficiently compute Wasserstein distances, barycenters, and geodesics between probability distributions. We have illustrated this by interpolating between grayscale images of a cat and a dog, effectively computing a geodesic in the space of grayscale $N\times N$ images for the $W_2$ metric.
