---
layout: post
title: "The case against leveraged ETFs"
date: 2024-06-17
description: "TL;DR: Leveraged ETFs amplify daily returns, which is not the same as basic leverage, especially in the long term. Digging into the math reveals that leveraged ETFs are not suitable buy-and-hold investments as they 1) exhibit huge price swings 2) incur a volatility drag."
# tags: 
thumbnail: assets/img/posts/leveraged_etf/tqqq_vs_leverage.png
---

$$
\newcommand{\R}{\mathbb{R}}
\newcommand{\tn}[1]{\textnormal{#1}}
$$

We first introduce leveraged ETFs: their purpose, daily rebalancing, and the common misconception around them. We then delve into the math behind leveraged ETFs, showing that they are not suitable for long-term investments due to 1) their extreme price swings 2) the volatility drag they incur. We illustrate our results on `TQQQ` and `SQQQ`, 3x and -3x leveraged ETFs on the Nasdaq 100 respectively. Our data comes from Yahoo Finance (through python module `yfinance`) and consists of the adjusted close prices from 2010 to 2024.

## ETFs and LETFs

ETFs, or *exchange-traded funds*, have gained prominence over the past decades. In 1993, the S&P 500 Trust ETF or SPY and its mere \\$11M in assets gave birth to the ETF industry. Over the years, the number of ETFs and their assets under management (AUM) have ballooned, with over 12,000 ETFs and \\$8T of AUM as of 2024[^investopedia]. ETFs appeal to investors for their low fees and trading flexibility. Unlike mutual funds which can only be traded at market close, ETFs are traded continuously just like stocks. In addition, ETFs use in-kind rather than cash creation/redemption of units, making them a tax-efficient investment vehicle. The ETF industry has grown so much that it now represents a significant fraction of public float as well as daily trading volume. In the U.S., ETFs accounted for 12.7% of equities and 28.2% of trading as of Q3 2023[^ishares]. The increasing weight of ETFs in the stock market motivates the study of ETF-related order flows and potential flow-based trading strategies.
    
*Leveraged* ETFs (LETFs) are a specific kind of ETF which utilize leverage to increase exposure to the underlying. The underlying is most commonly a popular stock index, but it can also be a single stock, for instance *Direxion Daily NVDA Bull 2X* (`NVDU`). For long LETFs, the leverage ratio $\beta$ is often 2 and sometimes 3. For short LETFs, also known as *inverse* ETFs, $\beta$ can be -1, -2, and more  rarely -3. Despite being relatively new, with the first fund opened in 2006[^marketwatch], the LETF industry has experienced frantic AUM growth over the past two decades, even exceeding that of the regular ETF industry. As of June 2024, LETFs accounted for more than $100B in the U.S. alone. `TQQQ` (a 3x Nasdaq 100 LETF) is the largest LETF in the world, with over $23B under management as of June 2024.

## Intuition behind LETFs

### Purpose and functioning

The announced objective of LETFs is to magnify the returns of an underlying. The underlying is often a popular stock index, but it can also be a single stock, bonds, commodities, and more recently cryptos. Throughout the rest of the paper, we will only consider LETFs linked to stock indexes.

In theory, a LETF with leverage $\beta$ ($\beta\in\lbrace -3,-2,-1,2,3 \rbrace$) will have daily close-to-close returns equal to $\beta$ times the underlying's close-to-close returns. For instance, if the S&P 500 index moves up 1% on a given day, then a corresponding 3x LETF (e.g. `SPXL`) will be up 3% on that day. Likewise, if the S&P 500 is down 2%, then that LETF will be down 6%. In practice, the fund's expense ratio and the cost of borrowing cash (resp. stocks) for long (resp. short) LETFs incur a small performance drag.

To gain leverage, the fund manager can either buy/short the index's individual components or enter into swap agreements. Regardless of the method used, the result is the same: daily LETF returns will be equal to the daily index returns multiplied by $\beta$. For the sake of simplicity, in the rest of this post we will assume that LETFs use swap agreements to gain exposure.

### Daily rebalancing

Non-leveraged ETFs tracking market capitalization-weighted indexes need to rebalance only when the underlying index itself undergoes a rebalance, which happens *quarterly* for most indexes. On the contrary, LETFs need to rebalance *daily*.

To understand why, let's consider a hypothetical 2x LETF for the S&P 500 with \\$100M under management on day 0. The index manager thus contracts $2\times100=\\$200M$ worth of swap agreement to have an exposure of $\beta=2$. Now let's assume the S&P 500 is up 1% on day 1. The fund is up $2\times 1 = 2$% and the AUM is now $1.02\times 100=\\$102M$, while our swap agreement exposure is now $1.01\times200=\\$202M$. However, to maintain the leverage $\beta=2$, we now need a swap exposure of $2\times 102=\\$204M$. Therefore we must contract another \\$2M worth of swap agreements to go from our current \\$202M exposure to the required \\$204M exposure. Likewise, if the S&P 500 goes down 1% on day 1, the new AUM is \\$98M and the new exposure is \\$198M. Since the new exposure must be $2\times 98=\\$196M$, we need to sell \\$2M worth of swap agreements. This example shows not only that a leveraged ETF must be rebalanced daily, but also that the fund manager always has to "buy high and sell low" since they increase (resp. decrease) their swap positions when the index is up (resp. down). Note that this is already a first hint of the adverse behavior of LETFs when held for more than a day.

Let's now formalize the reasoning above. Let us consider a LETF with leverage $\beta$ tracking a given index. We will denote $r_t$  the index's daily return on day $t$. In addition, $A_t$ and $S_t$ will be the AUM and swap exposure on day $t$, respectively. By definition, we have

$$
\begin{align*}
A _ {t+1}&=(1+\beta r _ t)A _ t
\\
S _ {t+1}&=(1+r _ t)S _ t=(1+r _ t)\beta A _ t
\end{align*}
$$

In order to maintain leverage $\beta$, our swap exposure at $t+1$ must be $\tilde{S} _ {t+1}=\beta A _ {t+1}=\beta(1+\beta r _ t)A _ t$, meaning that we need to update our swap positions according to
$$\Delta _ t=\tilde{S} _ {t+1}-S _ {t+1}=\beta(\beta-1)r _ t A _ t$$
We remark that the factor $\beta(\beta-1)$ is always positive for the values of $\beta$ considered (i.e. outside $[0,1]$), thus $\Delta _ t$ has the same sign as $r _ t$, which confirms that the LETF fund manager will be required to **"buy high and sell low"** every day as they rebalance their swap positions near market close.

### A common misconception

A significant fraction of retail and sometimes professional investors seem not to understand the long-term behavior of LETFs. Indeed, buy-and-hold investors should stay clear of LETFs under most circumstances[^sec]. As it turns out, the long-term performance of a LETF with leverage $\beta$ is (vastly) different from the performance of a static portfolio with leverage $\beta$[^1]. Mathematically, this is simply because we have

$$
\begin{align*}
A _ T^\text{LETF}&=A _ 0\prod _ {i=0}^{T-1}(1+\beta r _ t) \\
A _ T^\text{static}&=\beta\bigg(A _ 0\prod _ {i=0}^{T-1}(1+ r _ t)\bigg) - (\beta - 1)
\end{align*}
$$

where $A_t$ is the AUM on day $t$[^4].

To illustrate the difference in long-term performance between a LETF and a static portfolio with the same leverage, we consider the Nasdaq 100 index and its corresponding 3x LETF `TQQQ`. We compare the performance of `TQQQ` with a static portfolio holding QQQ with leverage 3. The results are shown in [Figure 1](#fig-1).

<div class="row justify-content-center" id="fig-1">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/leveraged_etf/tqqq_vs_leverage.png" title="tqqq" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/leveraged_etf/tqqq_vs_leverage_log.png" title="tqqq log" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 1. Performance of TQQQ vs. a static portfolio holding QQQ with leverage 3 (left: linear, right: log). Clearly, the long-term performance is not the same. In particular, TQQQ is extremely more volatile than the static portfolio.
</div>

To get an intuition behind the long-term divergence in performance between a LETF and the corresponding static portfolio, consider a hypothetical index whose daily returns alternate regularly between -0.9% and +1%. Over 2 days, this index will be up $\simeq0.09\%$ ($0.991\times1.01-1$) whereas a 2x LETF on that index will be up $\simeq0.16\%$ ($0.982\times1.02-1$). Compounded over a longer time period, the performance gap between the index and the corresponding 2x LETF widens: over 200 trading days, the index is up $\simeq9.5\%$ ($(0.991\times1.01)^{100}-1$) whereas the corresponding 2x LETF is up $\simeq17.8\%$ ($(0.982\times1.02)^{100}-1$). Likewise, a 3x LETF would be up roughly 24.5%.

<div style="text-align: center;" id="table-1">
    <table style="margin: 0 auto;">
        <thead>
            <tr>
                <th>Leverage $\beta$</th>
                <th>LETF</th>
                <th>Static portfolio</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>1</td>
                <td>+9.5%</td>
                <td>+9.5%</td>
            </tr>
            <tr>
                <td>2</td>
                <td>+17.8%</td>
                <td>+19.0%</td>
            </tr>
            <tr>
                <td>3</td>
                <td>+24.5%</td>
                <td>+28.5%</td>
            </tr>
            <tr>
                <td>-1</td>
                <td>-10.3%</td>
                <td>-9.5%</td>
            </tr>
            <tr>
                <td>-2</td>
                <td>-21.0%</td>
                <td>-19.0%</td>
            </tr>
            <tr>
                <td>-3</td>
                <td>-31.7%</td>
                <td>-28.5%</td>
            </tr>
        </tbody>
    </table>
</div>

**Table 1**: Performance over 200 trading days for a hypothetical underlying index alternating -0.9% and +1% daily returns. Note that for the static portfolio we have assumed no costs to borrow cash/stocks.

[Table 1](#table-1) illustrates the vast difference in performance between static portfolio and LETF for various values of $\beta$, still considering the same alternating index over 200 trading days. We remark that regardless of the leverage ratio, the LETF underperforms the corresponding static portfolio. Also, the performance drag worsens as $\| \beta \|$ increases. Let's now study a simple model to understand and quantify the observed performance discrepancy.


## Dynamics of LETFs

### Continuous-time model

In order to explain the surprising return dynamics observed previously, we can use a simple model proposed by Avellaneda[^avellaneda].
In this model, we denote by $S_t$ and $L_t$ the value of the index and the NAV[^2] of the corresponding LETF, respectively, on day $t$. In addition, let $r, f, \lambda$ be the risk-free rate, the fund's expense ratio, and the cost of borrowing the index, respectively. In particular, we assume these three variables to be constant, which is at least true on the short-term.

By construction of the LETF, we thus have[^3]
$$
\begin{equation} \label{basic-ret}
    \frac{dL_t}{L_t}=\beta \frac{dS_t}{S_t} - (\beta-1)r dt - fdt + 1_{\beta<0}\beta\lambda dt
\end{equation}
$$
where the cost of borrowing the index $\beta\lambda dt$ is incurred only for inverse LETFs i.e. when $\beta<0$.

In order to move forward we now need a model for the daily index returns $\frac{dS_t}{S_t}$. We use an Itô process
$$
\begin{equation} \label{ito}
    \frac{dS_t}{S_t}=\mu_t dt + \sigma_t dW_t
\end{equation}
$$

Plugging ($\ref{ito}$) in ($\ref{basic-ret}$) and separating the drift and noise terms, we obtain
$$
\begin{equation}
    \frac{dL_t}{L_t}=\big(\beta\mu_t-(\beta-1)r-f+1_{\beta<0}\beta\lambda\big)dt + \beta\sigma_t dW_t
\end{equation}
$$

We then use Itô's lemma to find
$$
\begin{equation}
    d[\ln{L_t}]=\big(\beta\mu_t-(\beta-1)r-f+1_{\beta<0}\beta\lambda - \frac{\beta^2 \sigma_t^2}{2}\big)dt + \beta\sigma_t dW_t
\end{equation}
$$

Integrating from 0 to $T$ yields
$$
\begin{equation} \label{eq3}
    \ln \frac{L_T}{L_0} = \beta M_T - \big((\beta-1)r -f + 1_{\beta<0}\beta\lambda\big)T - \frac{\beta^2}{2}V_T + \beta \sqrt{V_T}Z
\end{equation}
$$
where $M_T=\int_0^T \mu_t dt$, $V_T=\int_0^T \sigma_t^2 dt$ and $Z=N(0,1)$.

To simplify (\ref{eq3}), we remark that
$$
\begin{equation} \label{eq4}
    \ln \frac{S_T}{S_0} = M_T - \frac{V_T}{2} + \sqrt{V_T}Z
\end{equation}
$$

Plugging (\ref{eq4}) in (\ref{eq3}), we obtain
$$
\begin{equation}
    \ln \frac{L_T}{L_0} = \beta \ln \frac{S_T}{S_0} + \frac{\beta-\beta^2}{2}V_T - \big((\beta-1)r -f + 1_{\beta<0}\beta\lambda\big)T
\end{equation}
$$

Finally, exponentiation yields
$$
\begin{equation}
    \frac{L_T}{L_0} = \bigg(\frac{S_T}{S_0}\bigg)^\beta \exp\bigg(\frac{\beta-\beta^2}{2}V_T - \big((\beta-1)r -f + 1_{\beta<0}\beta\lambda\big)T\bigg)
\end{equation}
$$

Neglecting $r, f, \lambda$, we end up with the following neat expression
$$
\begin{equation} \label{eq:final}
\boxed{\frac{L_T}{L_0} = \bigg(\frac{S_T}{S_0}\bigg)^\beta \exp\bigg(-\frac{\beta^2-\beta}{2}V_T\bigg)}
\end{equation}
$$

### Interpretation and consequences

The first thing to note from ($\ref{eq:final}$) is that we clearly do not have the linear relationship $\frac{L_T}{L_0}-1=\beta(\frac{S_T}{S_0}-1)$ which we may naively expect. In fact, this linear relationship holds for the static portfolio only. Instead, we have a linear relationship between the *logarithms* of the LETF and the index, which is not the same thing at all! It means an *exponential* relationship between the LETF and the index, aka huge swings in the LETF's value. This is the first reason why LETFs are not suitable for long-term investments.

The second thing to note is the term $\exp\big(-\frac{\beta^2-\beta}{2}V_T\big)$ which is always strictly less than 1 since $\beta^2-\beta$ is positive for all the values of $\beta$ considered. Since $V_T=\int_0^T \sigma_t^2 dt$ is the realized volatility, we see that the LETF is adversely exposed to the index turbulence: there is a **"volatility drag"**. Thus, the higher the volatility, the larger $V_T$ and thus we need $\frac{S_T}{S_0}\gg 1$ to compensate. In particular, if we simply assume that $\forall t, \sigma_t=\sigma$, then $V_T=\sigma^2 T$ and thus $\frac{L_T}{L_0} = \bigg(\frac{S_T}{S_0}\bigg)^\beta \exp\bigg(-\frac{\beta^2-\beta}{2}\sigma^2 T\bigg)$ i.e. there is an exponential time decay in the long-term performance of the LETF.

The bottom line is that LETFs are not adequate buy-and-hold investments for the two reasons above. Holding them for too long will not yield the hoped-for linear relationship $r_\text{total}^\text{LETF}=\beta \times r_\text{total}^\text{index}$ ; worse still, it will incur an inevitable volatility drag which scales more or less as an exponential time decay.

## Illustration on TQQQ and SQQQ

Let's illustrate the volatility decay of `TQQQ` and `SQQQ`, the 3x and -3x Nasdaq 100 LETFs respectively. We will use equation ($\ref{eq:final}$) to compute $-\frac{\beta^2-\beta}{2}V_T = \ln(\frac{L_T}{L_0})-\beta \ln(\frac{S_T}{S_0})$: we expect to find a negatively sloped line.

<div class="row justify-content-center" id="fig-2">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/leveraged_etf/tqqq.png" title="tqqq" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/leveraged_etf/sqqq.png" title="sqqq" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 2. Exponential decay of TQQQ and SQQQ.
</div>

[Figure 2](#fig-2) presents the exponential decay of `TQQQ` and `SQQQ` over the past 15 years. We obtain a negatively sloped line as expected. In addition, we find that the slope is steeper for `SQQQ` than for `TQQQ`, which is consistent with the fact that the volatility drag is proportional to $\beta^2-\beta$, which is larger for `SQQQ` than for `TQQQ`. We cannot say much more about the slopes and the intersects since we've neglected $r, f, \lambda$ in our model.

## Conclusion

We've shown that LETFs are more complex than they seem. In particular, buying a $\beta$ LETF is not the same as buying a static portfolio with leverage $\beta$. Equation ($\ref{eq:final}$) nicely sums up the double problem with LETFs: 1) they vary exponentially with the underlying index 2) they incur a volatility drag. We could dig deeper by taking into account $r, f, \lambda$ in our model, but the main point is already clear: LETFs are not suitable for long-term investments.

You may wonder: who uses LETFs then? I have no definite answer, but certainly traders who wish to hedge their positions with less collateral may find LETFs useful. In addition, LETFs can be used to speculate on short-term market movements. The average detention period of `TQQQ` (~3 days) and `SQQQ` (<1 day) is a clear indication that LETFs are not meant to be held for long periods.

---

**Footnotes & References**:

[^1]: Such static portfolio is obtained by borrowing cash/stocks on day 0 to get the desired level of leverage and then holding until exit. In particular, there is no daily rebalancing and the portfolio's value can be negative.
[^2]: The Net Asset Value of a fund is the value of its assets divided by its number of shares outstanding. Technically, the NAV and the price per share are two distinct values, but in practice they remain very close for non-arbitrage reasons, and thus in this paper we will refer to LETF's NAV or price interchangeably. In addition, we will always use split-adjusted NAVs/prices.
[^3]: Technically, this formula holds true for daily returns only i.e. when $dt=\Delta t=1$ day, but we will assume it to be true on an infinitesimal time scale. Note that Avellaneda also proposed a discrete-time model equivalent to the one described here.
[^4]: Note that formally, the static portfolio can reach negative AUM.

[^avellaneda]: *Path-dependence of Leveraged ETF returns*. Avellaneda, M.
[^dynamics]: *The Dynamics of Leveraged and Inverse Exchange-Traded Funds*. Cheng, M. & Madhavan, A.
[^investopedia]: *A Brief History of Exchange-Traded Funds*. Simpson, S. [Investopedia.com](investopedia.com/articles/exchangetradedfunds/12/brief-history-exchangetraded-funds.asp)
[^ishares]: *Global ETF Market Facts: Three things to know from Q3 2023.*. Cohen, S. [iShares.com](ishares.com/us/insights/global-etf-facts)
[^marketwatch]: *ProFunds prepares first leveraged ETFs*. Spence, J. [MarketWatch.com](marketwatch.com/story/profunds-readies-first-leveraged-etfs)
[^sec]: *Updated Investor Bulletin: Leveraged and Inverse ETFs*. SEC Investor Alerts and Bulletins. [sec.gov](sec.gov/resources-for-investors/investor-alertsbulletins/updated-investor-bulletin-leveraged-inverse-etfs)


