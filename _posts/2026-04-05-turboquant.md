---
layout: post
title: "TurboQuant: KV cache hyper compression"
date: 2026-04-05
description: "TL;DR: As LLM trajectories get longer with the rise of agentic workflows, the KV cache becomes a massive memory bottleneck. Google's TurboQuant solves this via a lightweight, online, accelerator-friendly algorithm. We explore the beautiful high-dimensional geometry that makes it work: from random rotations on the $d$-sphere to minimize MSE distortion to unbiased inner products via QJL."
tags: llm, quantization, linear-algebra
thumbnail: assets/img/posts/turboquant/thumbnail1.png
---

$$\newcommand{\R}{\mathbb{R}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\r}{\mathbf{r}}
\newcommand{\u}{\mathbf{u}}
\newcommand{\w}{\mathbf{w}}
\newcommand{\x}{\mathbf{x}}
\newcommand{\y}{\mathbf{y}}
\newcommand{\z}{\mathbf{z}}
\newcommand{\q}{\mathbf{q}}
\newcommand{\k}{\mathbf{k}}
\newcommand{\v}{\mathbf{v}}
\newcommand{\e}{\mathbf{e}}
\newcommand{\s}{\mathbf{s}}
\newcommand{\I}{\mathbf{I}}
\newcommand{\S}{\mathcal{S}}
\newcommand{\U}{\text{U}}

\newcommand{\norm}[1]{\left\|#1\right\|}
\newcommand{\inner}[2]{\langle #1, #2 \rangle}
\newcommand{\QJL}{\text{QJL}}$$

<div class="row justify-content-center" id="fig-1">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/turboquant/thumbnail1.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <b>Figure 1. </b>Artist's view of TurboQuant. The first stage uses random rotations to redistribute the energy of the input vector across all coordinates, allowing for more efficient quantization. The second stage applies a Quantized Johnson-Lindenstrauss transform on the residual error to guarantee unbiased inner products, which is crucial for KV cache quantization since attention requires accurate estimations of key-query inner products.
</div>

In the era of agentic workflows and unbounded-context Large Language Models (LLMs), we are hitting a severe wall: **KV cache**. As generation trajectories get longer, storing the *key* and *value* tensors for every token consumes an exorbitant amount of memory[^kv-cache-usage], hindering both throughput (lower concurrency due to smaller batch sizes) and latency (more HBM<->SRAM traffic, even slower if we're offloading KV cache to CPU RAM).

There's been a lot of work to address KV cache bloat, which can be categorized in three main approaches:
1. **architectural**: the standard multi-head attention (MHA) was replaced with multi-query attention (MQA) and then grouped query attention (GQA) to reduce the number of KV heads. DeepSeek introduced multi-head latent attention (MLA) to further compress the KV cache.
2. **sparsity**: pruning unimportant tokens from the KV cache via various heuristics
3. **quantization**: aggressively quantizing the KV cache to reduce the per-token memory footprint.

In this post, we will discuss **TurboQuant**, a recent promising quantization approach developed by Google Research. TurboQuant is a lightweight, online, accelerator-friendly algorithm that achieves extreme compression of the KV cache *while* maintaining model performance. While the official announcements[^turboquant-post] and the paper itself[^turboquant-paper] are heavily geared toward empirical results and hardware efficiency, they gloss over something incredibly elegant: **the geometry of why this works.**

My goal in this post is to bridge that gap. We'll explore why TurboQuant is so effective, rederive the underlying math from first principles, and build a geometric intuition for its two main stages:
1. Random rotations on the hypersphere, to compress the signal.
2. Quantized Johnson-Lindenstrauss (QJL)[^qjl] transform on the residual error, to guarantee unbiased inner products

*If you want to try TurboQuant yourself, you can find my implementation [here](https://github.com/gaetanX21/turboquant).*

---

## Table of Contents
{:.no_toc}

* Table of Contents
{:toc}

---

## I. Motivation: the KV cache bottleneck

Before diving into the math, let's briefly review the problem. In autoregressive transformer architectures modeling $p(y_t \| y_{<t})$, the model attends to all previously generated tokens $y_{<t}$ when computing causal attention. If $\q\in\R^{d_k}$ is the query vector for the token $t$ and $K\in\R^{t\times d_k}$, $V\in\R^{t\times d_v}$ are the key and value matrices for tokens $y_{\leq t}$, the attention output for token $t$ is computed as:

$$
\text{Attention}(\q, K, V) = \text{softmax}\left(\frac{\q K^T}{\sqrt{d_k}}\right)V
$$

*Note: here we take the perspective of the decoded token only, hence $\q$ is a vector and not a matrix.*

To avoid recomputing $K$ and $V$ at each new decode step, we cache these two tensors. This is the KV cache. However, as the sequence length $t$ grows, the memory required to store $K$ and $V$ grows linearly, which becomes unsustainable for long trajectories.

One way to mitigate this is to quantize the KV cache, reducing the number of bits used to represent each element. However, LLM activations are known to exhibit emergent large magnitude features[^llmint8]. If we apply standard quantization techniques directly, these massive outliers dominate the quantization range, causing most of the "normal" values to be quantized to zero or a single bin, which destroys the signal.

Furthermore, because the KV cache is generated dynamically during inference, we can't rely on offline calibration. We need an **online** algorithm that is lightweight and accelerator-friendly (i.e. highly vectorized). This is exactly what TurboQuant delivers.

## II. Framing the problem

### Defining quantization

We are looking for a quantization procedure

$$
\begin{aligned}
Q: \mathbb{R}^d &\to \{0, 1\}^B \\
\x &\mapsto \z = Q(\x)
\end{aligned}
$$

that maps a continuous vector $\x$ to a $B$-bit binary vector $\z$, where $B=bd$ is the total number of bits we want to use. Note that this implies a fractional bitwidth of $B/d$ bits per coordinate. The reverse mapping $Q^{-1}: \lbrace 0,1 \rbrace^B \to \R^d$ takes the binary representation back to a continuous vector. Importantly, we make no assumptions about the distribution of $\x$; it can be arbitrarily pathological, e.g. heavy-tailed with outliers.

We let our quantizer $Q$ be **stochastic**, meaning that it can randomly map $\x$ to different binary vectors $\z$ on different runs. This is required for achieving conditional unbiasedness[^global-vs-conditional-unbiasedness]. Since we're data-agnostic, we need to design $Q$ to work well for all possible input vectors. So we fix $\x, \y \in \R^d$ as (worst-case) arbitrary vectors.

We note $\tilde{\x} = Q^{-1}(Q(\x))$ for the reconstructed vector. The key objectives for designing our quantizer $Q$ are:
1. Minimizing MSE distortion: $\E_Q[\norm{\x - \tilde{\x}}^2]$
2. Minimizing inner product distortions: $\E_Q[\|\inner{\y}{\x} - \inner{\y}{\tilde{\x}}\|^2]$ for any query vector $\y\in\R^d$
3. Unbiasedness of inner products: $\E_Q[\inner{\y}{\tilde{\x}}] = \inner{\y}{\x}$ for any query vector $\y\in\R^d$. Note that this implies $\E[\tilde{\x}] = \x$ by setting $\y = \e_i$ for each standard basis vector.

In practice, because the attention mechanism relies on inner products $\inner{\q}{\k}$, KV cache quantization is more about preserving the geometry of inner products than minimizing MSE. More on this later.

### The TurboQuant approach

TurboQuant is a two-stage algorithm that achieves these objectives. In a nutshell:
1. Stage 1: Apply a random rotation to the input vector $\x$ to smear outliers and make the coordinates well-behaved, then apply a Lloyd-Max quantizer to minimize MSE distortion.
2. Stage 2: Apply a Quantized Johnson-Lindenstrauss (QJL) on the residual quantization error to ensure that the inner products are unbiased.

We will cover both stages in the next sections. What you need to know for now is that
1. Stage 1 targets the MSE but does not guarantee unbiased inner products, which is something we really want for attention.
2. Stage 2 is designed to fix the inner product bias introduced by Stage 1.

What's interesting about TurboQuant is that the techniques behind Stage 1 and Stage 2 were already known individually, but combining them serially is novel!

## III. Stage 1: Random rotation then Lloyd-Max quantization

### Intuition

To quantize a vector effectively, we would like its coordinates to follow a predictable, well-behaved distribution without extreme outliers. Once again, that's because outliers will dominate the quantization range, causing most values to be quantized to a single bin, resulting in excessively lossy quantization.

TurboQuant achieves this by relying on a classic technique from high-dimensional geometry: mixing up the coordinates to smear the energy across all dimensions, effectively "destroying" outliers. This is done by applying a **random rotation** to the activation vectors $\k\in\R^{d_k}$ and $\v\in\R^{d_v}$.

In fact, the rotation does three very important things:
1. It redistributes the energy of the vector across all coordinates, so that no single coordinate dominates the quantization range at the expense of the others.
2. It transforms the distribution of each coordinate into a well-known distribution (specifically, a Beta distribution), allowing us to design an optimal quantizer before even seeing the data!
3. It makes the coordinates asymptotically independent as the dimension grows, which allows us to apply a simple scalar quantizer (Lloyd-Max) to each coordinate without worrying about complex joint distributions.

We'll now go over the math behind each of these points, but the key takeaway is that **random rotations are a powerful tool for taming the wild distributions of LLM activations, making them amenable to aggressive quantization**.

*In addition, note that applying a random rotation is an accelerator-friendly operation.*

### 1. Redistributing energy

#### Lemma 1 (Random rotations yield the uniform distribution on the $d$-sphere)

Let $\x \in \S^{d-1}$ be a fixed point on the $d$-sphere. Consider $\Pi \sim \U(O(d))$, a random rotation matrix sampled uniformly from the orthogonal group. Then, the random variable $\y=\Pi\x$ is uniformly distributed on $\S^{d-1}$, i.e. $\y\sim\U(\S^{d-1})$.

#### Proof sketch
*Let $\mu$ denote the probability measure on $\S^{d-1}$ induced by the random variable $\y=\Pi\x$. This is the pushforward of the normalized Haar measure $\nu$ on $O(d)$ under the evaluation map $T_{\x}:\Pi \mapsto\Pi\x$, i.e. $\mu=T_{\x} \\# \nu$. Because the Haar measure is invariant under left multiplication by any fixed rotation, $\mu$ is rotationally invariant. Since the uniform distribution is the unique rotationally invariant probability measure on $\S^{d-1}$, it follows that $\mu$ must be the uniform distribution on $\S^{d-1}$. Therefore, $\y \sim \U(\S^{d-1})$.*

#### Consequence

Applying a random rotation to our input vector guarantees that it will be uniformly distributed on the surface of a high-dimensional sphere. This is a powerful result because it means that no matter how pathological the original distribution of $\x$ was (e.g. heavy-tailed with outliers), after rotation, the coordinates of $\y$ will follow a well-known distribution with no extreme outliers, which means we can design an effective quantizer for $\y$ without worrying about the original distribution of $\x$.

### 2. The distribution of coordinates after rotation

#### Lemma 2 (Coordinate distribution of a random point on the $d$-sphere)

Let $\x \sim \U(\S^{d-1})$ be a random variable uniformly distributed over the $d$-sphere. Then for any $j \in [d]$ the coordinate $\x_j$ follows the following (scaled/shifted) Beta distribution:

$$f_X(x) = \frac{\Gamma(d/2)}{\sqrt{\pi} \cdot \Gamma((d - 1)/2)} (1 - x^2)^{(d-3)/2}$$

In addition, we asymptotically have

$$f_X \xrightarrow[d \to \infty]{\mathcal{D}}\mathcal{N}(0, 1/d)$$

#### Proof sketch

*Let $S_{d-1}(r)=\frac{2\pi^{d/2}}{\Gamma(d/2)}r^{d-1}$ denote the surface area of a $d$-sphere of radius $r$. Geometrically, the condition $\x_j=x$ defines a slice of the $d$-sphere at height $x$. The cross-section of this slice is a $(d-1)$-sphere with radius $\sqrt{1-x^2}$. The probability density of $\x_j=x$ is proportional to the surface area of this cross-sectional sphere, which is given by $S_{d-2}(\sqrt{1-x^2})$, divided by the total surface area of the unit sphere $S_{d-1}(1)$. Thus, we have $f_X(x) dx = \frac{S_{d-1}(\sqrt{1-x^2})}{S_d(1)} ds = \frac{\frac{2\pi^{(d-1)/2}}{\Gamma((d-1)/2)} (1-x^2)^{(d-2)/2}}{\frac{2\pi^{d/2}}{\Gamma(d/2)}} \frac{1}{\sqrt{1-x^2}} dx$, which simplifies to the stated Beta distribution. The relationship $ds=\frac{1}{\sqrt{1-x^2}}dx$ comes from the Pythagorean theorem.*

*The asymptotic normality can be derived analytically from $f_X$ using Stirling's approximation for the Gamma function. Intuitively, it can just be seen as a consequence of the concentration of measure in high dimensions.*

#### Consequence
After applying a random rotation to $\x$, each coordinate of the resulting vector $\y$ follows a well-known distribution (a scaled Beta distribution that converges to a normal distribution as dimension grows). This allows us to totally ignore the original distribution of $\x$ when designing our quantizer. We can simply design an optimal quantizer for the Beta distribution, and it will work well for all input vectors after random rotation.

### 3. Asymptotic independence

#### Lemma 3 (Asymptotic independence of coordinates on the $d$-sphere)

Let $\x \sim \U(\S^{d-1})$ be a random point sampled uniformly on the $d$-sphere. As $d \to \infty$, the coordinates of $\x$ become asymptotically independent.

#### Proof sketch

*We use the probabilistic representation method. Let $\z\sim \mathcal{N}(0, \I_d)$ be a vector of i.i.d. standard normal variables. Consider $\y = \frac{\z}{\norm{\z}}$. Because the multivariate normal distribution is spherically symmetric, $\y$ has no preferred direction and lies on the unit $d$-sphere. Thus, $\y \sim \U(\S^{d-1})$. By the law of large numbers, $\norm{\z} \xrightarrow[d \to \infty]{a.s.} \sqrt{d}$, so $\y_i \approx \frac{\z_i}{\sqrt{d}}$. Since the $\z_i$'s are independent, the coordinates $\y_i$ become asymptotically independent as $d \to \infty$.*

<div class="row justify-content-center" id="fig-2">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/turboquant/energy_redistribution.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <b>Figure 2. </b>Redistribution of energy after random rotation. Starting from a well-behaved normal distribution (left) or a uniform distribution of coordinates (right), applying a random rotation results in a distribution of coordinates that follows the same shape (a Beta distribution) regardless of the original distribution. This is what allows us to design a single quantizer for all input vectors. Here we used  $d=4096$.
</div>

#### Consequence
The coordinates of the rotated vector $\y$ become asymptotically independent as the dimension grows. This means that we can apply a simple scalar quantizer (like Lloyd-Max) to each coordinate independently without worrying about complex joint distributions or correlations between coordinates. This is a huge win for simplicity and efficiency in quantization.


### Lloyd-Max quantization

With the coordinates of $\y$ following a well-known distribution and being asymptotically independent, we can determine the optimal quantization bins and centroids using the Lloyd-Max algorithm, which minimizes MSE distortion for a given bit budget $b$.

Formally, the optimization problem we solve is a 1-dimensional $k$-means clustering problem:
$$
\mathcal{C}(f_X, b) = \min_{\substack{-1\leq c_1 \leq c_2 \leq \dots \leq c_{2^b} \leq 1}} \sum_{i=1}^{2^b} \int_{\frac{c_{i-1}+c_i}{2}}^{\frac{c_i+c_{i+1}}{2}} f_X(x) \| x - c_i \|^2 dx
$$

The beauty here is that we can compute these centroids analytically for the Beta distribution, so we don't even need to see the data to design our quantizer! For a given dimension $d$ and bit budget $b$, we can precompute the optimal quantization scheme and store the results for future use during inference.

<div class="row justify-content-center" id="fig-3">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/turboquant/centroids.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <b>Figure 3. </b>Optimal quantization centroids obtained with the Lloyd-Max algorithm for $d=4096$ and various bit budgets $b$. The centroids are symmetric around zero and more densely packed near the center where there is the most probability mass of the Beta distribution.
</div>

### Q-MSE algorithm

We now have all the ingredients to implement the Stage 1 quantization algorithm.

$$
\begin{array}{l}
\textbf{Algorithm 1 } \text{TurboQuant}_{\text{mse}}\text{: optimized for MSE} \\
\hline
\textbf{Input: } \text{dimension } d \text{ and bit-width } b \\
\\
\color{blue}{\texttt{// Global Parameters for Setting up } \text{TurboQuant}_{\text{mse}}} \\
\textbf{1: } \text{Generate a } \color{blue}{\texttt{random rotation matrix }} \boldsymbol{\Pi} \in \mathbb{R}^{d \times d} \\
\textbf{2: } \text{Construct } \color{blue}{\texttt{codebook}} \text{ by finding centroids } c_1, c_2, \dots, c_{2^b} \in [-1, 1] \text{ that minimize MSE cost} \\
\hline
\textbf{Procedure } \text{QUANT}_{\text{mse}} (\x) \\
\textbf{3: } \quad \y \leftarrow \boldsymbol{\Pi} \cdot \x \\
\textbf{4: } \quad \text{idx}_j \leftarrow \arg\min_{k \in [2^b]} |\y_j - c_k| \text{ for every } j \in [d] \qquad \triangleright \color{blue}{\texttt{ idx}_j\texttt{'s are b-bit integers}} \\
\textbf{5: } \quad \textbf{output: } \text{idx} \\
\hline
\textbf{Procedure } \text{DEQUANT}_{\text{mse}} (\text{idx}) \\
\textbf{6: } \quad \tilde{\y}_j \leftarrow c_{\text{idx}_j} \text{ for every } j \in [d] \\
\textbf{7: } \quad \tilde{\x} \leftarrow \boldsymbol{\Pi}^\top \cdot \tilde{\y} \\
\textbf{8: } \quad \textbf{output: } \tilde{\x} \\
\hline
\end{array}
$$

## IV. Stage 2: QJL for unbiasedness

Stage 1 compresses the data efficiently in terms of MSE, but it does not guarantee that the inner products are unbiased, i.e. $\E_Q[\inner{q}{\tilde{k}}] \neq \inner{q}{k}$. This is a problem for attention, which relies heavily on inner products between queries and keys to compute attention weights.

Thus, we need the inner product to be **unbiased**. This brings us to Stage 2: Quantized Johnson-Lindenstrauss (QJL).

### A. The QJL transform

The Quantized Johnson-Lindenstrauss (QJL[^qjl]) transform is a 1-bit vector quantization scheme. Just like the procedure introduced in stage 1, this scheme is randomized and data-agnostic, meaning that it does not rely on any assumptions about the distribution of the input vector.

QJL is defined as follows, where $S\in\R^{d\times d}$ is a random matrix with i.i.d. standard normal entries i.e. $S_{ij}\sim\mathcal{N}(0, 1)$:

$$
\begin{aligned}
\QJL: \R^d &\to \{-1, 1\}^d \\
\x &\mapsto \z=\text{sign}(S\x)
\end{aligned}
$$

$$
\begin{aligned}
\QJL^{-1}: \{-1, 1\}^d &\to \R^d \\
\z &\mapsto \tilde{x}=\frac{\sqrt{\pi/2}}{d} S^T\z
\end{aligned}
$$

Remarkably, the QJL transform has the following property: for any $\x\in\S^{d-1}, \y \in \R^d$, if we note $\tilde{\x} = \QJL^{-1}(\QJL(\x))$ as the reconstructed version of $\x$, we have

$$\E_Q[\inner{\y}{\tilde{\x}}] = \inner{\y}{\x}$$

In other words, the inner product between $\y$ and the reconstructed version of $\x$ is an unbiased estimator of the true inner product between $\y$ and $\x$. This is exactly what we need for attention!

Before proving this, let's build some geometric intuition for why this is the case.


### B. Geometric intuition of QJL

The QJL transform can be seen as a random hyperplane hashing scheme.

If we denote $\s_1^T, \s_2^T, \dots, \s_d^T$ the rows of the random matrix $S$ and $\z=\QJL(\x)$, then we see that $\z_i = \text{sign}(\inner{\s_i}{\x})$ is the sign of the projection of $\x$ onto the random vector $\s_i$. Thus, each row of $S$ defines a random hyperplane in $\R^d$, and the sign of $\z_i$ indicates on which side of the hyperplane $\x$ lies. The collection of signs $\z$ encodes a sort of "binary fingerprint" of the position of $\x$ relative to these random hyperplanes.

The dequantization step can be developed as $\QJL^{-1}(\z)=\frac{\sqrt{\pi/2}}{d} \sum_{i=1}^d  \z_i \s_i$, showing that it reconstructs $\x$ as a linear combination of the random vectors $\s_i$, weighted by the signs $\z_i$. Thus, the underlying heuristic is essentially, for each coordinate $\z_i$:
- if $\z_i=1$, it means that $\x$ is on the positive side of the hyperplane defined by $\s_i$, which means that $\s_i$ somewhat points in the same general direction as $\x$, so we add $\s_i$ to the reconstruction to get closer to $\x$.
- on the contrary, $\z_i=-1$ means that $\s_i$ somewhat points in the opposite direction of $\x$, so we subtract $\s_i$ from the reconstruction to get closer to $\x$.

Note that the $\frac{\sqrt{\pi/2}}{d}$ scaling factor is there to compensate the stretching effect of the random projection, ensuring that the expected length of the reconstructed vector matches that of the original vector. This factor naturally appears in the proof of unbiasedness.

Intuitively, we feel that as $d$ gets larger, the random hyperplanes defined by the $\s_i$ will be distributed more and more uniformly in all directions, such that the "binary fingerprint" $\z$ will capture more and more information about the position of $\x$ in space, and thus the reconstruction should be more and more accurate. Indeed, we can show that not only does QJL yield unbiased inner products estimators, but the variance $\text{Var}(\inner{\y}{\tilde{\x}})$ scales in $\frac{1}{d}$! We will prove this neat result in the next section.

### C. Unbiasedness of inner products via QJL

#### Lemma 4 (Unbiased inner products via QJL + Variance bound)

Let $\x\in\S^{d-1}, \y \in \R^d$, and $\tilde{\x} = \QJL^{-1}(\QJL(\x))$ be the reconstructed version of $\x$ via the QJL transform. Then we have
1. $\E_Q[\inner{\y}{\tilde{\x}}] = \inner{\y}{\x}$
2. $\text{Var}(\inner{\y}{\tilde{\x}})\leq \frac{\pi}{2d} \norm{\y}^2$

#### Proof

*By definition, we have $\inner{\y}{\tilde{\x}} = \y^T \frac{\sqrt{\pi/2}}{d} S^T\ \text{sign}(S\x) = \frac{\sqrt{\pi/2}}{d} \sum_{i=1}^d \inner{\s_i}{\y} \text{sign}(\inner{\s_i}{\x})$. We want to compute the expectation of this quantity over the randomness of $S$.*

*Let's decompose $\y$ as its orthogonal projection on $\x$ and its orthogonal complement, i.e. $\y = \inner{\y}{\x}\x + \y_\perp$. Note that $\y_\perp$ is orthogonal to $\x$ by construction.*

*We can then rewrite $\inner{\s_i}{\y} \text{sign}(\inner{\s_i}{\x})=\inner{\s_i}{\inner{\y}{\x}\x} \text{sign}(\inner{\s_i}{\x}) + \inner{\s_i}{\y_\perp} \text{sign}(\inner{\s_i}{\x})$.*

*Because $\s_i$ is a Gaussian vector and $\y_\perp$ is orthogonal to $\x$, the random variables $\inner{\s_i}{\x}$ and $\inner{\s_i}{\y_\perp}$ are independent. Thus $\E_Q[\inner{\s_i}{\y_\perp} \text{sign}(\inner{\s_i}{\x})]=\E_Q[\inner{\s_i}{\y_\perp}]\E_Q[\text{sign}(\inner{\s_i}{\x})]=0$.*

*We're left with $\E_Q[\inner{\s_i}{\inner{\y}{\x}\x} \text{sign}(\inner{\s_i}{\x})]=\inner{\y}{\x}\E_Q[\|\inner{\s_i}{\x}\|]$. Because $\inner{\s_i}{\x}\sim\mathcal{N}(0, 1)$, we have $\E_Q[\|\inner{\s_i}{\x}\|]=\sqrt{2/\pi}$ by a classic result on the folded normal distribution.*

*Plugging this back into the expression for $\inner{\y}{\tilde{\x}}$ yields $\E_Q[\inner{\y}{\tilde{\x}}] = \frac{\sqrt{\pi/2}}{d} \sum_{i=1}^d \inner{\y}{\x}\sqrt{2/\pi} = \inner{\y}{\x}$, proving unbiasedness.*

---

*To derive the variance bound, note $\u_i = \inner{\s_i}{\y} \text{sign}(\inner{\s_i}{\x})$. Since the $\s_i$'s are independent, the $\u_i$'s are also independent. Thus, $\text{Var}(\inner{\y}{\tilde{\x}}) = \frac{\pi}{2d^2} \sum_{i=1}^d \text{Var}(\u_i)$.*

*We can then bound $\text{Var}(\u_i)$ as follows: $\text{Var}(\u_i) = \E_Q[\u_i^2] - \E_Q[\u_i]^2 \leq \E_Q[\inner{\s_i}{\y}^2] = \norm{\y}^2$, where we used the fact that $\text{sign}(\inner{\s_i}{\x})^2=1$ and that $\E_Q[\inner{\s_i}{\y}^2]=\text{Var}(\mathcal{N}(0,\norm{\y}^2))=\norm{\y}^2$.*

*Plugging this back yields $\text{Var}(\inner{\y}{\tilde{\x}}) \leq \frac{\pi}{2d^2} \sum_{i=1}^d \norm{\y}^2 = \frac{\pi}{2d} \norm{\y}^2$, which is the desired bound.*



### D. Combining Stage 1 and Stage 2 with QJL on the residual

Now that we have understood and demonstrated the unbiasedness of the QJL transform for inner product, we can finally present the full TurboQuant algorithm, which combines the MSE-optimized quantization from Stage 1 with the QJL transform from Stage 2 to achieve both low distortion and unbiased inner products.

#### Description of the algorithm

The idea is simple yet elegant: for a given vector $\x$, we first apply the Stage 1 quantizer to get a compressed representation $\tilde{\x} _ {mse} = Q_{mse}^{-1}(Q_{mse}(\x))$ that minimizes MSE distortion. However, this representation may have biased inner products. To fix this, we consider the residual error $\r = \x - \tilde{\x} _ {mse}$ and apply the QJL transform to this residual to get a quantized version $\z$ that has unbiased inner products with any query vector.

Two details to note:
1. Since $\z$ consumes 1 bit per coordinate, we now spend 1 less bit per coordinate on the MSE quantizer to stay within our overall bit budget $B$ i.e. a bitwidth of $b$ bits per coordinate.
2. Since the $\text{sign}$ function makes the QJL transform scale-agnostic, we need to store the norm of the residual $\norm{\r}$ as an additional scalar to properly scale the QJL reconstruction. We ignore this cost in terms of bits since it's negligible compared to the overall bit budget for large $d$.


With this in mind, we naturally define the following two-stage quantization and dequantization procedures:

$$
\begin{aligned}
Q_{full}: \mathbb{R}^d &\to \{0, 1\}^{(b-1)d} \times \{-1, 1\}^d \times \R_+ \\
\x &\mapsto (\text{idx}, \z, \gamma) = (Q_{mse}(\x), \QJL(\x-Q_{mse}^{-1}(Q_{mse}(\x))), \norm{\x-Q_{mse}^{-1}(Q_{mse}(\x))})
\end{aligned}
$$

$$
\begin{aligned}
Q_{full}^{-1}: \{0, 1\}^{(b-1)d} \times \{-1, 1\}^d \times \R_+ &\to \mathbb{R}^d \\
(\text{idx}, \z, \gamma) &\mapsto \tilde{\x} = Q_{mse}^{-1}(\text{idx}) + \gamma \cdot \QJL^{-1}(\z)
\end{aligned}
$$

We now need to prove that this combined quantizer achieves unbiased inner products, i.e.

$$\E_Q[\inner{\y}{Q_{full}^{-1}(Q_{full}(\x))}] = \inner{\y}{\x}$$

for any query vector $\y\in\R^d$.

#### Proof of unbiasedness

*Let $\x, \y \in \R^d$ be arbitrary vectors. We want to show that $\E_Q[\inner{\y}{\tilde{\x}}] = \inner{\y}{\x}$ where $\tilde{\x}=\tilde{\x}_{mse}+\norm{\r}\QJL^{-1}(\QJL(\r))$ is the reconstructed version of $\x$ via the full TurboQuant algorithm.*

*Let's consider the conditional expectation $\E_Q[\inner{\y}{\tilde{\x}} \| \tilde{\x} _ {mse}]= \E_Q[\inner{\y}{\tilde{\x} _ {mse} + \norm{\r}\QJL^{-1}(\QJL(\r))} \|\tilde{\x} _ {mse}] = \inner{\y}{\tilde{\x} _ {mse}} + \norm{\r}\E_Q[\inner{\y}{\QJL^{-1}(\QJL(\r))} \|\tilde{\x} _ {mse}] = \inner{\y}{\tilde{\x} _ {mse}} + \inner{\y}{\r} = \inner{\y}{\x}$ where we used the unbiasedness of the QJL transform for inner products in the second equality, and the definition of $\r$ for the last equality. Note that conditioned on $\tilde{\x} _ {mse}$, the residual $\r$ is fixed, which is why we can take it out of the expectation.*

*Thus, by the law of total expectation, we have $\E_Q[\inner{\y}{\tilde{\x}}] = \E_Q[\E_Q[\inner{\y}{\tilde{\x}} \| \tilde{\x}_{mse}]] = \E_Q[\inner{\y}{\x}] = \inner{\y}{\x}$, which is the desired result.*

## Conclusion

Arguably, TurboQuant defines a new state-of-the-art in online vector quantization. By combining random rotations and random hyperplane hashing (QJL), two powerful techniques from high-dimensional geometry, it achieves low-bitwidth quantization while maintaining unbiased inner products. This unbiasedness makes TurboQuant relevant for a wide range of applications where inner products are fundamental, such as attention in transformers, but also cosine similarity for vector search.

In addition, TurboQuant is remarkably simple to implement and works naturally with accelerator hardware, making it a practical solution for reducing the KV cache bottleneck in LLM inference. I'm excited to see how this method will be adopted in the industry, and how it will allow for even longer agentic rollouts without running into prohibitive memory overheads[^kv-cache-usage].

---

**References**:

[^kv-cache-usage]: For Grouped Query Attention, KV cache usage per token can be computed as $L \cdot n_{KV} \cdot (d_k + d_v) \cdot \text{precision}$ bytes/token, where $L$ is the number of attention layers, $n_{KV}$ is the number of KV heads per attention block, and $d_k$, $d_v$ are the dimensions of keys and values respectively. For instance, for [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) we have $L=32$, $n_{KV}=8$, $d_k=d_v=128$, resulting in 131kB per token if we use 16-bit precision (BF16) for activations. For a 100k token trajectory, common in agentic rollouts, this amounts to 13GB KV cache overhead, which is prohibitive for most hardware setups.
[^turboquant-post]: Google Research (2025). *TurboQuant: Redefining AI Efficiency with Extreme Compression.* [[post](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)]
[^turboquant-paper]: Zandieh, A. et al. (2025). *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.* [[arXiv](https://arxiv.org/abs/2504.19874)]
[^qjl]: Zandieh, A. et al. (2024). *QJL: 1-Bit Quantized JL Transform.* [[arXiv](https://arxiv.org/abs/2406.03482)]
[^llmint8]: Dettmers, T. et al. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* [[arXiv](https://arxiv.org/abs/2208.07339)]
[^global-vs-conditional-unbiasedness]: A worthy side note on unbiasedness: in the quantization literature, there is a distinction between global unbiasedness and conditional unbiasedness. Global unbiasedness means that *over the distribution of all possible input vectors $\x$*, the expected value of the quantized output matches the expected value of the input, i.e. $\E_{Q,\x}[Q(\x)]=\E_{\x}[\x]$. Conditional unbiasedness, on the other hand, requires that *for every specific input vector $\x$*, the expected value of the quantized output equals $\x$, i.e. $\E_{Q}[Q(\x)\|\x]=\x$. Global unbiasedness is the weaker condition and can be satisfied even if certain inputs are systematically biased, as long as the overall distribution averages out. Conditional unbiasedness trivially implies global unbiasedness by the law of total expectation, but the converse is not true. The key insight is that because quantization always introduces some distortion i.e. $Q(\x)\neq \x$ (unless you happen to be quantizing a vector that lies exactly on a quantization centroid), achieving conditional unbiasedness is possible only if you let $Q$ be *stochastic*.