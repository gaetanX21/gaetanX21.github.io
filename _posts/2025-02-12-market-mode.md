---
layout: post
title: "Listening to the Market Mode"
date: 2025-02-12
description: "TL;DR: Performing PCA on returns amounts to constructing a statistical factor model. The largest eigenvalue corresponds to the market mode and far outweighs the other factors. Thus, one can perform rolling PCA on equities' returns to monitor the market risk over time."
tags: random-matrix-theory, linear-algebra, quant-finance
thumbnail: assets/img/posts/market-mode/TOP420.png
---

$$
\newcommand{\E}{\mathbb{E}}
\newcommand{\N}{\mathcal{N}}
$$

We first motivate the use of Principal Component Analysis (PCA) on returns to extract the market mode in equities. This mode is crucial for understanding the market risk and comparing it against other risks. We then do a (very) quick recap on Random Matrix Theory (RMT), which provides a theoretical framework for understanding the eigenspectrum of random matrices. Finally, we apply PCA on S&P 500 components to extract the market mode and we monitor its evolution over time, drawing a comparison with the VIX index.


## Motivation

Among the various asset classes (e.g., equities, bonds, commodities), equities tend to provide the highest returns in absolute terms (i.e. not adjusted for risk). Equities are exposed to a multitude of risk factors, with **market risk** being the dominant one[^CAPM]. As such, understanding the market risk and how much of the variance it explains is crucial for risk management and portfolio construction.

In a nutshell, the question we want to answer is the following: **can we measure the market risk and compare its weight against other risks?**

Perhaps the simplest approach to gauge market risk is to look at ready-made proxies such as the **VIX index**[^VIX]. However, the method of computing the VIX is debatable and may not capture the market risk accurately. A more data-driven approach is to extract the market mode from market returns using PCA, as explained in the next section.

## PCA on Returns = Statistical Factor Model

In a nutshell, a **factor model** describes the variance observed in a set of correlated variables (in our case, stock returns) using a smaller number of **unobserved factors**, which we hope to be more or less independent & more or less interpretable. The idea is to decompose the observed variables $X_i$ as linear combinations of the factors $F_k$ plus some idiosyncratic noise $\varepsilon_i$:

$$
X_i = \sum_k \beta_k^{(i)} F_k + \varepsilon_i
$$

where $\beta_k^{(i)}$ is the (factor) loading of the $i$-th asset on the $k$-th factor.

Now, the hard part is to find **good factors**. One approach is to simply purchase them from vendors like MSCI (Barra models) who gather a lot of data and knowledge to build these factors. Another (cheaper & more transparent) approach is to extract them via PCA. In this case, the factors are obtained as the eigenvectors of the correlation matrix of returns. This is nice for several reasons:
- eigenvectors are orthogonal[^spectral], meaning our factors are uncorrelated;
- eigenvalues directly give us the amount of variance explained by each factor;
- the factors are interpretable as they are linear combinations of the original variables.

Note that since the factors are obtained in a purely data-driven fashion (without any human/economic prior), we call this approach a **statistical** factor model, as opposed to classical factor models like the CAPM or the Fama-French Three-Factor Model.

*Before trying out PCA on real data, let's quickly brush up on Random Matrix Theory (RMT), which provides a neat theoretical framework for understanding the eigenspectrum of correlation matrices.*

## Brush Up on Random Matrix Theory (RMT)

Consider a $T \times N$ matrix $X$ filled with i.i.d. Gaussian entries $X_{ij} \sim \N(0,\sigma^2)$. Typically, $X$ is called the **design matrix** and each one of the $T$ rows corresponds to one observation of the $N$ variables of interest. In our case, the variables are the daily close-to-close returns of the $N=500$ stocks in the S&P 500 index.

If we want to study the correlation between the variables, we first compute a standardized version of the design matrix $\tilde{X}$ by subtracting the mean and dividing by the standard deviation for each column. The sample correlation matrix is then given by $C = \frac{1}{T} \tilde{X}^T \tilde{X}$.

Finally, $C$ is real symmetric so we know from the spectral theorem that it can be diagonalized in an orthonormal basis of eigenvectors. In fact $C$ is also positive semi-definite, so all its eigenvalues are non-negative.

#### Marchenko-Pastur Theorem

RMT studies the behavior of the eigenvalues of $C$ when $T,N \to \infty$ with $Q = T/N$ fixed. The main result is the **Marchenko-Pastur (MP) theorem**:

$$
L_N(\lambda) = \frac{1}{N} \sum_{i=1}^N \delta(\lambda - \lambda_i) \xrightarrow[N,T \to \infty]{\mathcal{W}} \mathbb{P}_\text{MP}(\lambda) = \frac{Q}{2\pi \sigma^2} \frac{\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}}{\lambda} \mathbb{1}_{[\lambda_-, \lambda_+]}(\lambda)
$$

where $\lambda_{\pm} = \sigma^2(1\pm\sqrt{\frac{1}{Q}})^2$ are the limiting bounds of the spectrum. 

The MP theorem tells us that the empirical spectral density of the correlation matrix $L _ N(\lambda)$ converges (weakly in distribution) toward the MP distribution $\mathbb{P} _ \text{MP}$ as $T,N \to \infty$.

This is quite remarkable: we could have expected the eigenvalues to be unbounded as $T,N \to \infty$, but RMT tells us that they are actually bounded and gives us the exact form of the limiting distribution.

In fact, we can relax some hypotheses and the MP theorem will still hold, though convergence may be (significantly) slower. For example, the entries of $X$ don't have to be Gaussian. This is important in our case because we know that Gaussianity is a strong assumption for financial data. In practice returns have fat tails and are often skewed. A Student-distribution is already a much better model for returns. *Good news, MP still holds for Student-distributed entries!*

#### Link with PCA

Now, what does this have to do with PCA? Well, the MP theorem tells us that the eigenvalues of the correlation matrix are bounded and distributed according to a known law. This is useful because it allows us to detect the presence of **signal** in the data. If the eigenvalues are significantly larger than the MP bounds, then we can say that the data contains some structure that is not due to randomness. On the contrary, eigenvalues inside the "noise band" defined by the MP law are considered to be due to randomness. Thus, **we can use the MP theorem to filter out noise and extract only the significant factors from the data**. In particular, when doing PCA on market returns, one eigenvalue will stand out from all the others: it represents the dominant variance component due to the market, and as such we call it the **market mode**. It is approximately equally distributed between all the stocks and is the most important factor in the data.

In this last section, we illustrate the above ideas through an experiment on US equities. Specifically, we compute rolling PCAs on the correlation matrix of S&P 500 components and analyze the behavior of the market mode over time.

## Experiment on US Equities

To run our experiment, we first need some data. We chose US equities because they are the most liquid and it's easy to get clean data. We fetched the daily close-to-close returns of the S&P 500 components from Yahoo Finance using the `yfinance` Python package. We considered the period 2000–2025[^sp500].

#### PCA on 2020-2024 returns

Let's begin by computing the correlation matrix of the daily returns of the S&P 500 components for the period 2020–2024. We then compute the eigenvalues of the correlation matrix and plot them against the MP bounds. The results are shown in [Figure 1](#fig-1). Importantly, <u>the largest eigenvalues are outside the plot</u> for better visibility. There is only ~10 of them, but they are much larger than the rest. **In particular, the first principal component stands out from all the others: it represents the market mode.** The other significant eigenvalues are due to sectoral correlations, which are also interesting to study but outside the scope of this post.

<div class="row justify-content-center" id="fig-1">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/market-mode/mp_true.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 1. Eigenvalues of the correlation matrix of S&P 500 components for the period 2020–2024. The largest eigenvalues are outside the plot.
</div>

<div class="row justify-content-center" id="fig-2">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/market-mode/dist_eigvec_mode.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 2. Distribution of the eigenvector of the market mode as well as a noisy mode for the period 2020–2024. Note how the market mode is approximately equally distributed between all the stocks whereas the noisy mode follows a normal distribution, which makes sense because it has no information and thus must maximize entropy.
</div>

As an extra step, we can shuffle the returns to destroy the correlation structure and then recompute the eigenvalues. The results are shown in [Figure 3](#fig-3). Notice how all the eigenvalues are now neatly inside the MP bounds, which confirms that the structure in the data is due to correlations and not randomness.

<div class="row justify-content-center" id="fig-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/market-mode/mp_shuffled.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 3. Eigenvalues of the correlation matrix of S&P 500 components for the period 2020–2024 after shuffling the returns. All the eigenvalues are now inside the MP bounds and indeed follow the MP distribution.
</div>


#### Rolling PCA on 2000-2024 returns

Now that we've seen how PCA works on a single sample of market returns, let's apply it to a rolling window of the daily returns of the S&P 500 components for a large period of time. The idea is to monitor the evolution of the ratio $\lambda_\text{max} / \sum_i \lambda_i$ over time, where $\lambda_\text{max}$ is the largest eigenvalue (market mode) and $\lambda_i$ are the other eigenvalues. This ratio gives us an idea of how much of the variance is explained by the market mode. Intuitively, we expect this ratio to be high during times of uncertainty / fear / crisis as the market becomes even more important in driving returns. To check this hypothesis, we compare the ratio to the VIX index.

We'll be looking at the period 2000–2024, which includes the 2008 financial crisis and the 2020 COVID-19 pandemic. We will take 6-month rolling windows with a 1-month step size and compute the ratio $\lambda_\text{max} / \sum_i \lambda_i$ for each window. Note that for technical reasons[^sp500], we only consider the top 420 companies in the S&P 500 index (ranked by daily trading volume) instead of the full 500. However taking the top 420 companies or top 500 companies doesn't change the results significantly[^proof].

The results are shown in [Figure 4](#fig-4). The ratio $\lambda_\text{max} / \sum_i \lambda_i$ is plotted in green and the VIX index is plotted in red. We can see that the two series are quite correlated, which confirms our intuition. In particular, we see that the ratio spikes during the 2008 financial crisis, and the 2020 COVID-19 pandemic. This is a nice result as it shows that PCA can indeed capture the market mode and that it is a good proxy for market risk.

<div class="row justify-content-center" id="fig-4">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/market-mode/TOP420.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 4. Ratio $\lambda_\text{max} / \sum_i \lambda_i$ (green) and VIX index (red) over the period 2000–2024. The two series are quite correlated, which confirms our intuition that the market mode is a good proxy for market risk.
</div>

## Conclusion

In this post, we've seen how PCA can be used to extract the market mode from equities' return data. This mode is crucial for understanding the market risk and weighing it against other risks. We've also seen how Random Matrix Theory provides a theoretical framework for understanding the eigenspectrum of correlation matrices. Finally, we've applied PCA on S&P 500 individual returns to extract the market mode and we've analyzed its behavior over time, drawing a comparison with the VIX index.

Note that the market mode is not the only factor that matters. If we stick to the PCA approach (statistical factor model), there are several other eigenvalues outside the MP noise band. These eigenvalues and the corresponding eigenvectors correspond to sectors of the US economy (e.g., tech, finance, utilities). They too are risk factors, albeit less important than the market mode. In practice, depending on the context, one may want to consider these factors, for instance to build a sector-neutral portfolio.

---

**Notes**:

[^CAPM]: The Capital Asset Pricing Model (CAPM) is the most simple factor model as it relies on the market factor only. In a nutshell, it posits that the expected return of an asset is linearly related to the expected return of the market depending on the asset's correlation with the market, known as the beta coefficient.

[^VIX]: The VIX index is a measure of the market's expectation of volatility over the next 30 days. It is calculated using the implied volatility of S&P 500 options and is often referred to as the "fear gauge" as it tends to spike during market downturns.

[^spectral]: The eigenvectors of a real symmetric matrix are orthogonal. This is a consequence of the spectral theorem, which states that a real symmetric matrix can be diagonalized by an orthonormal basis of eigenvectors.

[^sp500]: Note that the composition of the S&P 500 index changes over time as companies are added or removed. We use the current components of the index for each year. Sometimes too many components change and this causes problems. One simple solution is to only consider the top 420 companies (instead of 500), which are more stable. (NB: we rank the companies by daily trading volume.)

[^proof]: One way to confirm this intuition is to re-run the experiment but looking at the top 100 companies instead of the top 420. The results are very similar, which shows that the market mode is indeed robust to the number of companies considered (as long as it's not too small of course).