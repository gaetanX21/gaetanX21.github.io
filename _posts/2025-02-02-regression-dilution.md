---
layout: post
title: "Regression Dilution"
date: 2025-02-01
description: "TL;DR: When covariates in linear regression are subject to noise, the estimated regression coefficients shrink towards zero. We derive this effect mathematically and illustrate it with simulations."
# tags:
thumbnail: assets/img/posts/regression_dilution/snr_shrinkage.png
---

$$
\newcommand{\E}{\mathbb{E}}
\newcommand{\Var}{\text{Var}}
\newcommand{\Cov}{\text{Cov}}
\newcommand{\R}{\mathbb{R}}
$$

We first introduce the concept of **attenuation bias** in linear regression due to measurement error in the covariates. We derive the shrinkage effect in the one-dimensional case and extend it to the multivariate case. We do simulations to visualize the shrinkage effect as the signal-to-noise ratio (SNR) goes to zero.

## Introduction

In classical linear regression, we assume a model of the form:

$$
y = X\beta + \varepsilon
$$

where $X$ is an $n \times p$ matrix of covariates $x_i \in \R^p$, $\beta$ is a $p \times 1$ vector of coefficients, and $\varepsilon$ is noise. **Weak exogeneity** is a key assumption in linear regression, which states that the covariates $X$ are fixed and non-random. In other words, the covariates are assumed to be measured without error. However, in many real-world scenarios, this hypothesis is violated: covariates themselves contain measurement noise:

$$
\tilde{X} = X + U
$$

where $U$ is an $n \times p$ matrix of noise $u_i \in \R^p$. This additional noise leads to a phenomenon known as **attenuation bias**, where the estimated coefficients shrink towards zero. Let's first derive this effect in the one-dimensional case.

Note that in what follows we make the following the classical assumptions:
- $x_i$ i.i.d. centered with variance $\sigma_x^2$ (or covariance $\Sigma_x$ in the multivariate case),
- $u_i$ i.i.d. centered with variance $\sigma_u^2$ (or covariance $\Sigma_u$ in the multivariate case),
- $\varepsilon_i$ i.i.d. centered with variance $\sigma_\varepsilon^2$,
- $x_i, u_i, \varepsilon_i$ are independent of each other


## One-dimensional case

Let's first derive the attenuation bias in the one-dimensional case.

Consider the simple case of a one-dimensional linear regression model:

$$
y = \beta x + \varepsilon.
$$

Now assume that we observe a noisy version of $x$: $\tilde{x} = x + u$, where $u$ is the noise term. The least squares estimator of $\beta$ using the noisy covariate $\tilde{x}$ is:

$$
\hat{\beta} = \frac{\Cov(\tilde{x}, y)}{\Var(\tilde{x})} = \frac{\Cov(x + u, \beta x + \varepsilon)}{\Var(x + u)} = \frac{\beta \Var(x)}{\Var(x) + \Var(u)} = \frac{\beta \sigma_x^2}{\sigma_x^2 + \sigma_u^2} = \lambda \beta
$$

where $\lambda = \frac{1}{1 + \frac{\sigma_u^2}{\sigma_x^2}}<1$ is the attenuation factor or shrinkage factor.

Thus the estimated coefficient $\hat{\beta}$ is a scaled version of the true coefficient $\beta$, with the scaling factor $\lambda$ being less than 1. This implies that the estimated coefficient is biased towards zero due to the noise in the covariate.

In particular, note that when $\sigma_u = 0$, we recover the unbiased estimator $\hat{\beta} = \beta$. Likewise, as $\sigma_u \to \infty$, the estimated coefficient $\hat{\beta} \to 0$ since the SNR goes to zero.


## Multivariate case

The multivariate case can be derived similarly, though the algebra is slightly more involved.

If we use the noisy covariates $\tilde{X}$ instead of the true covariates $X$, the least squares estimator becomes:

$$
\hat{\beta} = (\tilde{X}^T \tilde{X})^{-1} \tilde{X}^T y.
$$

Substituting $\tilde{X} = X + U$ and $y = X\beta + \varepsilon$ gives:

$$
\hat{\beta} = [(X + U)^T (X + U)]^{-1} (X + U)^T (X\beta + \varepsilon)
$$

We rewrite this expression so that the law of large numbers can be applied:

$$
\hat{\beta} = \bigg[\frac{1}{n}(X^T X + X^T U + U^T X + U^T U)\bigg]^{-1} \bigg[\frac{1}{n}(X^T X\beta + X^T \varepsilon + U^T X\beta + U^T \varepsilon)\bigg] 
$$

Using the weak law of large numbers, we have

$$
\begin{align*}
\frac{1}{n}X^T X &\to \E[x x^T] = \Sigma_x \\
\frac{1}{n}X^T U &\to \E[x u^T] = 0 \\
\frac{1}{n}U^T X &\to \E[u x^T] = 0 \\
\frac{1}{n}U^T U &\to \E[u u^T] = \Sigma_u
\end{align*}
$$

and

$$
\begin{align*}
\frac{1}{n}X^T X\beta &\to \E[x x^T]\beta = \Sigma_x \beta \\
\frac{1}{n}X^T \varepsilon &\to \E[x \varepsilon_i^T] = 0 \\
\frac{1}{n}U^T X\beta &\to \E[u x^T]\beta = 0 \\
\frac{1}{n}U^T \varepsilon &\to \E[ \varepsilon_i^T] = 0
\end{align*}
$$

where all the convergences are in probability[^strong].

Combining these results and applying Sluskty's lemma and the continuous mapping theorem, we have:

$$
\hat{\beta} \xrightarrow[]{\mathbb{P}} (\Sigma_x + \Sigma_u)^{-1} \Sigma_x \beta = \left(I + \Sigma_x^{-1} \Sigma_u \right)^{-1} \beta.
$$

Note that in the multi-dimensional case, the shrinkage factor is not a scalar but a matrix $\Lambda = (I + \Sigma_x^{-1} \Sigma_u)^{-1}$. In particular, although $\Sigma_x$ and $\Sigma_u$ are positive definite matrices, $\Sigma_x^{-1} \Sigma_u$ is not positive definite in general. Therefore it is more difficult to interpret the shrinkage effect in the multivariate case.

For simplicity, if we assume spherical noise on both covariates and response, i.e., $\Sigma_x = \sigma_x^2 I$ and $\Sigma_u = \sigma_u^2 I$, we recover the one-dimensional result with $\lambda = \frac{1}{1 + \frac{\sigma_u^2}{\sigma_x^2}}$. This makes sense because assuming spherical noise is like running the one-dimensional case independently for each covariate.

Additionally, we recover the unbiased estimator $\hat{\beta} = \beta$ when $\Sigma_u = 0$, as expected. 


## Visualizing the shrinkage effect as SNR goes to zero

We want to illustrate the gradual shrinkage of the estimated coefficients as the SNR gradually decreases. We stick to the one-dimensional case for simplicity.

We simulate a linear regression model with a single covariate $x$ with $\sigma_x = 1$ and noise $u$ with $\sigma_u$ running from $0$ to $5 \sigma_x$. For each value of $\sigma_u$, we fit a linear regression model using the noisy covariate $x + u$ and record the estimated coefficient $\hat{\beta}$.

We then plot the empirical shrinkage ratio $\frac{\hat{\beta}}{\beta}$ as a function of the SNR $\frac{\sigma_x}{\sigma_u}$. Additionally, we overlay the theoretical shrinkage factor $\lambda = \frac{1}{1 + \frac{\sigma_u^2}{\sigma_x^2}}$.

The results are shown in [Figure 1](#fig-1). As the SNR decreases, the estimated coefficients shrink towards zero, as expected. The empirical shrinkage ratio closely follows the theoretical shrinkage factor $\lambda$.

<div class="row justify-content-center" id="fig-1">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/regression_dilution/snr_shrinkage.png" title="snr shrinkage" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 1. Empirical shrinkage ratio as a function of the SNR. The theoretical shrinkage factor $\lambda = \frac{1}{1 + \frac{\sigma_u^2}{\sigma_x^2}}$ is overlaid.
</div>


## Conclusion

When covariates are measured with noise, the estimated regression coefficients shrink towards zero, leading to bias. This is important in fields where measurement errors are common, such as economics and epidemiology. One way to mitigate this bias is to use **error-in-variables models**, which explicitly model the noise in the covariates. The simplest such model is probably Deming regression, which models a one-dimensional linear regression and assumes the SNR to be known. $\hat{\beta}$ is then found by minimizing a *weighted* sum of squared residual to account for the noise in $x$.

---

**Notes**:

[^strong]: The strong law of large numbers would require additional assumptions on the moments of the random variables involved.
