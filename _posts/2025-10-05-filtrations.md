---
layout: post
title: "Filtrations demystified"
date: 2025-10-05
description: "TL;DR: Filtrations are a key ingredient in defining stochastic processes and modeling the accumulation of available information over time. Filtrations are also often poorly understood; this post aims to demystify them."
tags: probability-theory, stochastic-processes
thumbnail: assets/img/posts/filtrations/filtrations.png
---

$$
\newcommand{\R}{\mathbb{R}}
\newcommand{\Q}{\mathbb{Q}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\A}{\mathcal{A}}
\newcommand{\P}{\mathbb{P}}
\newcommand{\T}{\mathbb{T}}
\newcommand{\F}{\mathcal{F}}
\newcommand{\FF}{\mathbb{F}}
$$

If you've ever studied stochastic processes[^random-process] in a formal setting, you must have come across the concept of *filtration*. I remember my stochastic calculus course where the professor introduced filtrations in a very abstract way, which left me quite confused. He then added that "filtrations encode the information available at each time step", which was a bit more intuitive but still quite vague. Anyway, this was how I understood filtrations for quite some time, until one day I decided to dig deeper into the topic and really understand what filtrations are and why they are useful. This post aims to share this understanding!

**NB:** I will assume that you are familiar with *σ-algebras*. If not, I recommend reading my previous post on [measurability](/blog/2025/measurability/) before continuing!

---


## I. Motivation

When dealing with basic random variables, we simply need:
1. a probability space $(\Omega, \A, \P)$,
2. a random variable $X: \Omega \to E$, where $(E, \mathcal{E})$ is another measurable space.

This setup is sufficient for many applications in probability theory.

However, defining stochastic processes requires a subtler formalism. The core reason is that stochastic processes are collections of random variables indexed by time[^indexing], and as such the information contained in the process accumulates over time, which must be accounted for in the formalism. Indeed, just like we need σ-algebras to define which events are measurable for a random variable $X$, we need a mathematical object to define which events are measurable *at a given time $t$* for a stochastic process $(X_t) _ {t\in\T}$. This mathematical object is called a *filtration*.


## II. Defining filtrations

### A. Intuitive definition

Before delving into the formal definition of filtrations, I think the [Wikipedia page](https://en.wikipedia.org/wiki/Filtration_(probability_theory)) sums it up quite nicely:
> "In the theory of stochastic processes, a subdiscipline of probability theory, **filtrations are totally ordered collections of subsets that are used to model the information that is available at a given point** and therefore play an important role in the formalization of random (stochastic) processes."

### B. Formal definition

Formally, let $(\Omega, \A, \P)$ be a probability space and $\T$ be a totally ordered set (typically $\N$ or $\R_+$).

A *filtration* is a family of sub-σ-algebras $(\F_t)_{t\in\T}$ of $\A$ such that for all $s, t \in \T$ with $s \leq t$, we have $\F_s \subseteq \F_t (\subseteq \A)$. In other words, the σ-algebras are nested and non-decreasing over time. We often denote a filtration with the symbol $\FF$.

If $(\Omega, \A, \P)$ is a probability space and $\FF=(\F_t)_{t\in\T}$ is a filtration on this space, then the quadruplet $(\Omega, \A, \FF, \P)$ is called a *filtered probability space*.

Importantly, note that we do not need stochastic processes to define filtrations. A filtration $\FF$ can be defined on any probability space, regardless of whether it is associated with a stochastic process or not. However, filtrations are almost always constructed in relation to a stochastic process, using what we call the *natural filtration* of the process.

### C. Natural filtration

Given a stochastic process $(X_t) _ {t\in\T}$, its natural filtration is defined as $\F_t = \sigma(X_s \| s \leq t)$ for each $t \in \T$. In other words, $\F_t$ is the smallest σ-algebra that makes all random variables $X_s$ for $s \leq t$ measurable. Intuitively, $\F_t$ declares all the information about the process **available** at time $t$.

### D. Available information ≠ information

I really want to emphasize the difference between *available information* and (actual) *information*, which is often the source of confusion when dealing with filtrations.

Let's consider a stochastic process $(X_t) _ {t\in\T}$ with its natural filtration $\FF=(\F_t)_{t\in\T}$.

Interpreting $\F_t$ as containing the information $(X_s)_{s\leq t}$ is **incorrect**. That is, $\F_t$ *does not* contain the realized values of $X_s$ for $s \leq t$. If this isn't clear, remark that ---by construction--- the σ-algebra $\F_t$ will be identical regardless of the specific values taken by the random variables $X_s$ for $s \leq t$; thus, it cannot possibly encode this information.[^filtration-independent] Instead, $\F_t$ declares explicitly which information about the process is **available** at time $t$, without holding the information in itself!

The key insight ---which is not so straightforward in my opinion--- is that being *measurable* with respect to $\F_t$ means being *determinable* using the information available at time $t$. In other words, if a random variable $Z$ is $\F_t$-measurable, then at time $t$ we will be able to determine its value. In particular, for any statement $S$ about the process, if $Y=\mathbf{1}_S$ is $\F_t$-measurable, then at time $t$ we will know with certainty whether $S$ is true or false. Once again, the information about the veracity of $S$ *is not* contained in $\F_t$; $\F_t$ simply declares that we will know if $S$ is true or false at time $t$.

The above intuition can be formalized as follows:
>Consider a random process $(X_t) _ {t\in\T}$ and its natural filtration $\FF=(\F_t) _ {t\in\T}$. Let $t\in\T$ be a specific point in time, and $A\in\F_t$ a $\F_t$-measurable event. Consider $\omega\in\Omega$ an arbitrary realization. Then, at time $t$, we will know deterministically whether $\omega\in A$ or not.

This is what it means when we say that $\F_t$ encodes the *available* information at time $t$.

## III. Examples

At this point, you may still be a bit confused about what filtrations really are. A few examples should help clarify things!

### A. Coin tosses

Consider a sequence of two consecutive coin tosses, modeled by the stochastic process $(X_1, X_2)$ (i.e. $\T=\lbrace 1, 2\rbrace$) where $X_i$ is the outcome of the $i$-th toss ($H$ or $T$).
1. The sample space is $\Omega = \lbrace HH, HT, TH, TT\rbrace$.
2. The σ-algebra is $\A = \mathcal{P}(\Omega)$ (the power set of $\Omega$).
3. The probability measure $\P$ assigns a probability of $1/4$ to each outcome.

Now let's think about the natural filtration $\FF$ of this process.
0. At time $t=0$ (before any toss), we have $\F_0 = \big\lbrace\emptyset, \Omega\big\rbrace$, i.e. there is no information available whatsoever about the outcomes of the tosses.
1. At time $t=1$ (after the first toss), we have $\F_1 = \sigma(X_1) = \big\lbrace\emptyset, \Omega, \lbrace HH, HT\rbrace, \lbrace TH, TT\rbrace\big\rbrace$, i.e. we now know the outcome of the first toss, but not the second. Indeed, after observing the first toss, we can distinguish between the events *"first toss is H"* ($\lbrace HH, HT\rbrace$) and *"first toss is T"* ($\lbrace TH, TT\rbrace$). In other words, we can tell whether our realization $\omega$ lies in $\lbrace HH, HT\rbrace$ or in $\lbrace TH, TT\rbrace$.
2. At time $t=2$ (after the second toss), we have
$$
\begin{aligned}
\F_2
&=\sigma(X_1, X_2) \\
&=\big\lbrace \emptyset, \Omega, \lbrace HH \rbrace, \lbrace HT \rbrace, \lbrace TH \rbrace, \lbrace TT \rbrace, \lbrace HH, HT \rbrace, \lbrace HH, TH \rbrace, \lbrace HH, TT \rbrace, \lbrace HT, TH \rbrace, \lbrace HT, TT \rbrace, \lbrace TH, TT \rbrace, \lbrace HH \rbrace^c, \lbrace HT \rbrace^c, \lbrace TH \rbrace^c, \lbrace TT \rbrace^c \big\rbrace \\
&=\mathcal{P}(\Omega)
\end{aligned}
$$

i.e. we know the outcomes of both tosses, so we can distinguish between all possible outcomes.


### B. Academic outcomes
 
We can make a more concrete version of the previous example. Let's consider a world in which students can go to two universities: Stanford ($S$) or Carnegie Mellon ($C$). After graduating, they can either get a job ($J$) or pursue a PhD ($P$). We can model this situation with a stochastic process $(X_1, X_2)$ where $X_1$ is the university attended and $X_2$ is the post-graduation outcome. 

The sample space is $\Omega = \lbrace SJ, SP, CJ, CP\rbrace$, the σ-algebra is $\A = \mathcal{P}(\Omega)$, and we can define a uniform probability measure $\P$ for simplicity.

The filtration $\FF$ of this process is as follows:
0. At time $t=0$ (before university), we have $\F_0 = \big\lbrace\emptyset, \Omega\big\rbrace$, i.e. there is no information available about the student's academic outcome.
1. At time $t=1$ (after university), we have $\F_1 = \sigma(X_1) = \big\lbrace\emptyset, \Omega, \lbrace SJ, SP\rbrace, \lbrace CJ, CP\rbrace\big\rbrace$, i.e. we now know which university the student attended, but not their post-graduation outcome.
2. At time $t=2$ (after graduation), we have $\F_2 = \sigma(X_1, X_2) = \mathcal{P}(\Omega)$, i.e. we know the student's entire academic outcome.


### C. Brownian motion

I'll assume that you are familiar with Brownian motion. If not, check out the Wikipedia page on the [Wiener process](https://en.wikipedia.org/wiki/Wiener_process).

For the one-dimensional Brownian motion $(B_t)_{t\geq 0}$:
1. The sample space is $\Omega = \R^\T$[^continuous-paths] with $\T = [0, +\infty)$.
2. The σ-algebra is $\A = \sigma(B_t : t \geq 0)$ (the σ-algebra generated by the Brownian motion).
3. The probability measure $\P$ is (by definition) the Wiener measure.

Unlike the previous examples, $(B_t) _ {t\geq 0}$ is a *continuous-time* stochastic process, so we cannot explicitly write down the filtration $\FF=(\F_t)_{t\geq 0}$, except for $\F_0 = \lbrace\emptyset, \Omega\rbrace$. However, $\F_t$ still represents the information available about the Brownian motion up to time $t$. For instance, the event $A = (B_1 > 0)$ is such that:
1. $A \notin \F_t$ for all $t < 1$ i.e. before time $t=1$ we cannot know whether $B_1$ is positive or not
2. $A \in \F_t$ for all $t \geq 1$ i.e. at time $t=1$ and afterwards we can know for sure whether $B_1$ is positive or not.

Although it's somewhat less straightforward to describe what filtrations look like for continuous-time processes, the key idea remains the same: $\F_t$ encodes all the information that *can be known for sure* about the process at time $t$.

## Conclusion

I've tried to make this post as intuitive as possible, and hopefully the examples helped in that regard. Filtrations do not need to be complicated, they are simply a way to formalize the idea of accumulating **available** information over time in the context of stochastic processes. Yes, you can live without knowing the full truth behind filtrations, but beyond a certain point, it's worth deeply understanding the mathematical objects you manipulate.

In machine learning, many people use quite complicated models without really grasping the underlying mathematics[^diffusion]. Of course, this is fine to a certain extent, but I firmly believe that having an intuitive understanding of the theory is key whenever you want to venture *just a tiny bit* off the beaten path.

Finally, I must mention that filtrations exist outside the realm of probability theory. In essence, given a set $X$, any family of nested subsets of $X$ is a filtration; in particular, these subsets *do not* need to be σ-algebras. One typical use case in linear algebra is to simplify the study an infinite-dimensional vector space $V$ by instead considering an increasing sequence $(V_t) _ {t\in\N}$ of finite-dimensional vector spaces such that $\bigcup_{t\in\N} V_t = V$. This is called a *filtration of $V$*.


---

**References**:

[^random-process]: Stochastic processes are equivalently called random processes in some literature.
[^indexing]: More precisely, the index $\T$ can be any totally ordered set, but in most cases we have $\T=\N$ or $\T=\R_+$ which are interpreted as discrete and continuous time respectively.
[^filtration-independent]: Another way to put it is that the filtration $\FF$ is defined purely in terms of the σ-algebras generated by the random variables $(X_t) _ {t\in\T}$, and not their realizations. Thus, $\FF$ is independent of the specific outcome $\omega\in\Omega$. This clearly shows that $\F_t$ cannot contain the actual information about the values taken by $(X_s) _ {s\leq t}$.
[^continuous-paths]: In practice, one can prove that the sample paths $B(\omega,\cdot): t \mapsto B(\omega,t)$ are almost surely continuous, so we could restrict $\Omega$ to $\mathcal{C}(\T,\R)$ without loss of generality.
[^diffusion]: For instance, it's easy to overlook that diffusion modeling amounts to solving a time-reversed stochastic differential equation, yet seeing diffusion under the light of SDEs yields [fruitful insights](https://yang-song.net/blog/2021/score/).