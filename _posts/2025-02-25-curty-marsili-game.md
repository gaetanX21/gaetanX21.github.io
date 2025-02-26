---
layout: post
title: "The Curty & Marsili Forecasting Game"
date: 2025-02-25
description: "TL;DR: When faced with a forecasting task, one can either seek information or follow the crowd. The Curty & Marsili game stacks fundamentalists against herders in a binary forecasting task, revealing phase coexistence and ergodicity breaking under certain conditions. We propose a theoretical study of the game's behavior and validate it through ABM simulations."
tags: agent-based-model, game-theory
thumbnail: assets/img/posts/curty_marsili_game/runs.png
---

$$
\newcommand{\E}{\mathbb{E}}
\newcommand{\N}{\mathcal{N}}
$$

In this post, we present Curty & Marsili’s forecasting game, a simple model that captures how herding behavior can lead to non-trivial opinion outcomes, in particular **phase coexistence** and **ergodicity breaking** under certain conditions. After motivating the study of herding, we formally introduce the Curty & Marsili game and propose a mathematical analysis of its key features. We then perform **Agent-Based Model** (ABM) simulations of the game to validate our theoretical predictions. Finally, we discuss how the game can converge to a Nash equilibrium where fundamentalists and herders coexist and the system is efficient.

---

## I. Motivation

Herding is a widespread phenomenon in society: people often imitate or follow the actions of others, whether it’s in fashion trends, product adoption, or even protests. In finance, herding can lead to anomalous fluctuations in asset prices. The alternative to herding is to gather (private) information and make decisions based on one’s own analysis. Curty & Marsili’s forecasting game[^curtymarsili] is a simple ABM which precisely focuses on this tension between **individual forecasting** and **collective herding**.

In a nutshell, the question Curty & Marsili tried to answer is: **what's more efficient, following the crowd or relying on your own judgment?** We'll see that the answer is not straightforward and can depend on the parameters of the game. In essence, we'll find that herding can be a sound strategy, but if the proportion of followers in the market becomes too large, herding becomes a dangerous strategy.

*Let's now present in details of the Curty & Marsili forecasting game.*


## II. The Game

The Curty & Marsili requires the following ingredients:
- $N\gg 1$ agents who must make a binary forecast (e.g., election outcome, buy or sell, protest or not, etc.)
- A fraction $z\in[0,1]$ of agents are **fundamentalists** $F_i$ who rely solely on their private information. They are correct with probability $p>\frac{1}{2}$ (i.e., they have an edge). Their forecast is fixed once and for all and crucially, fundamentalists' forecasts are mutually independent.
- The remaining fraction $1-z$ are **herders** $H_i$ who each have a fixed group $G_i$ of $M$ agents which they follow. Their action is determined by majority voting within their group (note that group size $M$ is odd to avoid draws). Importantly, note that groups may include both fundamentalists and herders.

The game then dynamically evolve according to the following rules:
- At each time step $t$, all herders are chosen one by one (in a random order) and update their forecast based on the majority opinion of their group $G_i$. (note that herders' initial forecast are i.i.d. random i.e. correct with probability $\frac{1}{2}$)
- The fundamentalists $F_i$ do not change their forecast over time. (reflecting their reliance on private information)
- The process is repeated until convergence to a fixed point (i.e. herders are all following the majority opinion of their group).

The question then is to study the final probability $q$ that a herder makes the correct forecast, computed as the fraction of herders who forecast the correct outcome after the game has converged. More precisely, we want to study $q(z)$, the final probability of a herder making the correct forecast as a function of the fraction of fundamentalists $z$ in the market. 

*We now delve into the mathematical analysis of the game.*

## III. Mathematical Analysis

Let's introduce two important notations:
- $q_t$ is the probability that a herder makes the correct forecast at time $t$.
- $\pi_t$ is the probability that a randomly chosen agent makes the correct forecast at time $t$.

Since agents are either fundamentalists or herders, we have the following static equation:
$$
\begin{equation}
\label{eq:static}
\pi_t = zp + (1-z)q_t.
\end{equation}
$$

In addition, a herder will make the correct forecast at time $t+1$ if and only if the majority of his group $G_i$ makes the correct forecast at time $t$, i.e. at least $\frac{M+1}{2}$ agents in the group make the correct forecast. This leads to the following dynamic equation:
$$
\begin{equation}
\label{eq:dynamic}
q_{t+1} = \sum_{k=\frac{M+1}{2}}^M \binom{M}{k} \pi_t^k (1-\pi_t)^{M-k}.
\end{equation}
$$

Combining \eqref{eq:static} and \eqref{eq:dynamic}, we can write the evolution of $q_t$ as:
$$
\begin{equation}
\label{eq:evolution}
q_{t+1} = F_z(q_t).
\end{equation}
$$

where $F_z(q) = \sum_{k=\frac{M+1}{2}}^M \binom{M}{k} (zp + (1-z)q)^k (1-(zp+(1-z)q))^{M-k}$.[^condorcet]

We can then compute fixed points $q^*(z)$ for the evolution equation \eqref{eq:evolution} for various values of $z\in[0,1]$. The results are illustrated in [Figure 1](#fig-1).

<div class="row justify-content-center" id="fig-1">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/curty_marsili_game/fixed_points.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 1. Fixed points of the evolution equation $q_{t+1} = F_z(q_t)$ for various values of $z$. Note the critical value $z_c\simeq \frac{1}{2}$ separating the two regimes. Each curve corresponds to a different fixed point: green for $q_+^*(z)$, blue for $q_-^*(z)$, and red for $q_u^*(z)$ the unstable fixed point.
</div>

We distinguish two regimes, separated by a critical value $z _ c\simeq \frac{1}{2}$:
- for $z>z _ c$ i.e. when there are mostly fundamentalists, there is a single (stable) fixed point $q _ +^ * (z)$. Interestingly, $q^ * (z)>p$ in this regime, meaning that herders are on average more accurate than fundamentalists. As $z$ decreases (while staying above $z_c$), the performance of herders gets even better! This can seem somewhat surprising, but in fact results from the fact that more herders means herders will have more herders in their group $G_i$, which in turn increases their forecast accuracy since in this regime herders are more accurate than fundamentalists.
- for $z<z _ c$ i.e. when there are mostly herders, two new fixed points appear, both under the line $q=\frac{1}{2}$ which means that these fixed points are bad for herders. Note that $q _ +^* (z)$ keeps increasing as $z$ decreases, while $q _ -^* (z)$ decreases. The unstable fixed point $q _ u^* (z)$ is also shown in red. Numerical simulations show that the system will converge to either $q _ +^* (z)$ or $q _ -^* (z)$ depending on the initial conditions. In fact, the unstable fixed point $q _ u^* (z)$ acts as a *separatrix* between the two regimes. Thus the initial condition $q_0$ will determine the final state of the system: if $q _ 0>q _ u^* (z)$, the system will converge to $q _ +^* (z)$, otherwise it will converge to $q _ -^* (z)$. This will be useful in the last section where we compare $\langle q \rangle$ to $p$ to find the Nash equilibrium.

What's interesting is the **phase coexistence** in herding regime $z<z_ c$: if the system converges towards $q_ -^*(z)$, then the majority of herders will make the wrong forecast; likewise, if the system converges towards $q_ +^*(z)$, the majority of herders will make the correct forecast. This is a clear example of **ergodicity breaking** where the system is stuck in one of the two phases, depending on the initial conditions $q_0$. In the last section we take into account the distribution $q_0\sim N(\frac{N}{2},\frac{N}{4})$ to compute the probability $p_ -$ of the system converging to $q_-$ (and similarly $p_+$ for $q_+$) so we can finally compute the average probability $\langle q \rangle$ of a herder making the correct forecast and compare it to fundamentalists' accuracy $p$.

*Now that we've analyzed the theoretical aspects of the game, let's move on to simulations to check if its consistent.*

## IV. ABM Simulation

The ABM is pretty straightforward here, with two agents classes (fundamentalists and herders) and at each step of the system, herders are picked one by one in a random order and update their forecast based on the majority opinion of their group. We iterate until convergence of the system i.e. herders' opinions are stable. We use the `mesa` python library[^mesa] which helps build ABMs very easily.

Throughout the simulations, we use the following parameters:
- $N=1000$ agents in total, with $z\in[0,1]$ the fraction of fundamentalists.
- $p=0.52$ the probability that a fundamentalist makes the correct forecast.
- $M=7$ the size of the groups of herders.
Note that modifying these parameters will result in quantitative but not qualitative changes in the outcomes.

We run simulations for various values of $z\in[0,1]$ and for each simulation we record $q_t$ at each time step $t$ of the system. We especially care about the final probability $q_\text{final}$ that a herder makes the correct forecast, which is simply the fraction of herders who forecast the correct outcome after the game has converged. We observe the following:
- in the low-herding regime $z>z_c$, $q_\text{final}$ is always above $p$ and very close to $q_+$.
- in the high-herding regime $z<z_c$, $q_\text{final}$ is either close to $q_-$ or $q_+$ depending on the initial conditions. This is consistent with the phase coexistence and ergodicity breaking observed in the theoretical analysis.

In [Figure 2](#fig-2), we plot $q_t$ over time for $n=100$ simulations for various values of $z$. We observe that the system converges to a fixed point after a few time steps, and the final probability $q_\text{final}$ is consistent with the theoretical predictions. As expected, we find that:
- in the high-herding regime $z<z_c$, the system can converge to either $q_-$ or $q_+$ depending on the initial conditions, and we have $q_- \simeq 0$ and $q_+ \simeq 1$ as illustrated in [Figure 1](#fig-1).
- in the low-herding regime $z>z_c$, the system converges but toward a wider spectrum of values, which are above $p$ on average i.e. $\langle q_\text{final} \rangle > p$.

We see in particular that the richness of the phase transition towards $z\simeq z_c$ is well captured by the ABM simulations.

<div class="row justify-content-center" id="fig-2">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/curty_marsili_game/runs.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 2. Evolution of $q_t$ over time for $n=100$ simulations for various values of $z$. The system converges to a fixed point after a few time steps. The ensemble final probability $q_\text{final}$ is indicated by the y-tick on the right.
</div>

We notice that $q_\text{final}(z)$ seems to increase with $z<z_c$ then decrease with $z>z_c$. To investigate this behavior, we plot the average final probability $\langle q_\text{final} \rangle$ over $n=100$ simulations for various values of $z$. The results are given in In [Figure 3](#fig-3). Interestingly, we see that $\langle q_\text{final} \rangle > p$ for all values of $z$, except near the $z=0$. This implies that herding is always a better strategy than being a fundamentalist... **except when there are too many herders in the system!**

<div class="row justify-content-center" id="fig-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/curty_marsili_game/q_z.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 3. Average final probability $\langle q_\text{final} \rangle$ over $n=1000$ simulations for various values of $z$. Note that $\langle q_\text{final} \rangle > p$ for all values of $z$ except near $z=0$, meaning that herding is a better strategy than being a fundamentalist except when virtually everyone adopts the herding strategy!
</div>

*Let's finish by showing how letting $z$ fluctuate naturally leads to a Nash equilibrium.*

## V. Nash Equilibrium

We have seen that if there aren't too many herders (i.e. if $z$ isn't too low), then $\langle q_\text{final}(z) \rangle > p$, i.e. herders are more accurate than fundamentalists on average. In this case, it is rational for fundamentalist agents to become herders, which means that $z$ will decrease. However there cannot be too many herders, since in the limit $z\to 0$ we have $q_\text{final}=\frac{1}{2}$ as all agents are herders and thus there is not information ("edge") in the system. We thus expect the system to self-organize until the proportion $z$ of fundamentalists fluctuates around a critical value $z^\dagger$ such that $\langle q_\text{final}(z^\dagger) \rangle = p$. This is the Nash equilibrium of the system, where fundamentalists and herders coexist and the system is efficient. (or arbitrage-free in the context of financial markets)

We can show [^curtymarsili] that the Nash equilibrium $z^\dagger$ is given by $z^\dagger \sim N^{1/2}$ where $N$ is the total number of agents. This means that most agents are followers and there is a little minority of $\sqrt{N}$ fundamentalists feeding information ("edge") to the system.

Additionally, note that since $z^\dagger$ is very small, we have $q_-\simeq 0$ and $q_+\simeq 1$ as illustrated in [Figure 1](#fig-1). Then, if we denote $p_-$ (resp $p_+$) the probability that the system converges to $q_-$ (resp $q_+$) given the initial conditions, we have $\langle q_\text{final} \rangle = p_- q_- + p_+ q_+$, which at the Nash equilibrium rewrites $p=p_+$, meaning that the probability of the herder mass to converge to the truth ($q_+$) is $p$, as if they represented a single fundamentalist agent!


## Conclusion

Despite its simplicity, the Curty & Marsili game suffices to display non-trivial behavior such as phase coexistence and ergodicity breaking. The game is a good illustration of how herding can be a good strategy... until too many agents adopt it and the whole herding population starts behaving like a single agent which is correct with probability $\langle q_\text{final} \rangle$. Finally, if we let agents switch strategy, $z$ will naturally converge to the efficient state $z^\dagger$ where $\langle q_\text{final} \rangle = p$ such that no strategy has an edge over the other. We find that $z^\dagger \sim N^{1/2}$, meaning that **it is optimal (in a game-theoretic sense) that most agents are followers and a little minority of fundamentalists is feeding information to the system**.

---

**References**:

[^curtymarsili]: *Phase coexistence in a forecasting game.* Curty, P. & Marsili, M. (2008) [PDF](https://wrap.warwick.ac.uk/id/eprint/1769/1/WRAP_Curty_fwp05-15.pdf)
[^condorcet]: Note the similarity with the **Condorcet Jury Theorem**, where the probability of a correct decision by a majority vote increases with the number of jurors and their individual accuracy. [wikipedia](https://en.wikipedia.org/wiki/Condorcet%27s_jury_theorem)
[^mesa]: *Mesa: An Agent-Based Modeling Framework in Python.* [mesa.readthedocs.io](https://mesa.readthedocs.io/)