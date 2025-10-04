---
layout: post
title: "Measurability and σ-algebras"
date: 2025-10-04
description: "TL;DR: σ-algebras are omnipresent when doing probability, yet they are somewhat arcane. Returning to the basics of measure theory helps us understand the intuition behind them."
tags: probability-theory, measure-theory
thumbnail: assets/img/posts/measurability/sigma-algebra.png
---

$$
\newcommand{\R}{\mathbb{R}}
\newcommand{\Q}{\mathbb{Q}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\A}{\mathcal{A}}
\newcommand{\P}{\mathbb{P}}
$$

Whenever we want to do probabilities ---which is basically all the time in the context of probabilistic machine learning--- we need to define a probability space.

A probability space is a mathematical object that consists of three components: a sample space $\Omega$, a σ-algebra $\A$ over $\Omega$, and a probability measure $\P$ defined on $\A$. It is commonly denoted as the triplet $(\Omega, \A, \P)$.

While the concepts of sample space and probability measure are relatively intuitive, σ-algebras are often more elusive and harder to grasp. The goal of this post is to demystify σ-algebras by revisiting the foundational concept of *measurability*[^measure-wiki].

---


## I. Measurability

### A. Defining measure

Before we can understand σ-algebras, we need to understand what a *measure* is.

> Simply put, a measure is a function that assigns a number to a set in a way that generalizes the concepts of *length, area, and volume* to arbitrary sets and dimensions.

Formally, if $X$ is a set and $\Sigma$ a σ-algebra over $X$[^sigma-algebra], a measure is a function $\mu: \Sigma \to [0, +\infty]$ such that:
1. $\mu(\emptyset) = 0$ (the measure of the empty set is zero),
2. **non-negativity**: for all $A \in \Sigma$, $\mu(A) \geq 0$,
3. **σ-additivity**: for any countable collection of disjoint sets $\{A_k\} _ {k\in\N} \subseteq \Sigma$, we have $\mu\left(\bigcup_{k\in\N} A_k\right) = \sum_{k\in\N} \mu(A_k)$.

In this context, the couple $(X, \Sigma)$ is called a *measurable space*, while the triplet $(X, \Sigma, \mu)$ is called a *measure space*.

Importantly, if $\mu$ is a probability measure (i.e. $\mu(X) = 1$), then $(X, \Sigma, \mu)$ is called a *probability space*. This is a key remark because, as my probability theory professor put it once:

> "Probability theory is just a fancy name for measure theory."

### B. The Lebesgue measure

Now that we have defined what a measure is, let's look at an important example: the Lebesgue measure on the real line $\R$.[^single-dimension] Note that we will happily skip its theoretical mathematical construction as it would only obscure the main ideas.

> Simply put, the Lebesgue measure $\lambda$ on $\R$ is a function that assigns to each interval its length, i.e. $\lambda([a, b]) = b - a$ for any $a < b$.

This is quite natural and intuitive. However, we would like to be able to measure much more complicated subsets of $\R$, such as unions of intervals, fractals, or even more exotic sets. As it turns out, the Lebesgue measure is designed in a way that is extendable to a wide class of sets, though not all of them (we will come back to this later).

Perhaps the easiest way to grasp the Lebesgue measure is through its characterization as *the only non-trivial measure that is translation-invariant and agrees with our intuition of length on intervals*.

As a side note, one may wonder whether other measures exist on $\R$. The answer is yes, for instance the *counting measure* which simply counts the number of elements in a set (it assigns $\infty$ to infinite sets). However, the Lebesgue measure is by far the most important one in analysis and probability theory. In particular, most useful measures on $\R$ (e.g. Gaussian measure, exponential measure, etc.) are absolutely continuous with respect to the Lebesgue measure, which means that they can be expressed as integrals against the Lebesgue measure, i.e.

$$
\mu(A) = \int_A f(x) d\lambda(x) 
$$ 

where $f$ is a non-negative measurable function called the *density* of $\mu$ with respect to $\lambda$.[^radon-nikodym]

### C. Not all sets are measurable

So far we have defined the Lebesgue measure on basic intervals $I=[a,b]$, and we have further claimed that it can be extended to a wide class of sets. However, it turns out that not all subsets of $\R$ are measurable with respect to the Lebesgue measure. That is, there exists some subsets $A \subseteq \R$ such that $\lambda(A)$ is not defined. In other words, the Lebesgue measure can only be defined on strict subsets $\Sigma$ of the power set of $\R$, i.e. $\Sigma \subsetneq \mathcal{P}(\R)$.

**This is a deep and somewhat counter-intuitive result in measure theory.**

Indeed, one may wonder why we cannot simply define the Lebesgue measure on all subsets of $\R$. The reason is that doing so would lead to contradictions with the properties of a measure, in particular σ-additivity. If that can reassure you, non-measurable sets are quite pathological in the sense that they are entangled with their complement in a way that defies our usual intuition about sets. Intuitively, you can think of non-measurable sets as scattered dust instead of nice continuous chunks of space. In fact, non-measurable sets are so counter-intuitive that they cannot be constructed without invoking the Axiom of Choice.[^zf]

The Vitali sets are a classic example of non-measurable subsets of $[0,1]$.[^vitali-construction] Other famous results include the Haussdorff paradox, which demonstrates the existence of non-measurable subsets of the sphere $S^2$, and the Banach-Tarski paradox, which shows that it is possible to decompose a solid ball in $\R^3$ into a finite number of non-measurable pieces and then reassemble them into two solid balls identical to the original!

## II. σ-algebras

### A. Motivation

We have seen that not all subsets of $\R$ are measurable with respect to the Lebesgue measure. Thus, we need to specify which subsets of $\R$ we want to consider when defining a measure. *This is exactly what a σ-algebra does.*

Intuitively, a σ-algebra is a collection of subsets of a set $X$ (i.e. $\Sigma \subseteq \mathcal{P}(X)$) whose purpose is to declare explicitly which subsets of $X$ we want to measure and which we do not. In other words, a σ-algebra is a way to formalize the notion of "measurable sets". In addition, a σ-algebra has to satisfy the following intuitive properties[^boolean] so that it works well with the measure:
1. $X \in \Sigma$: we want to be able to measure the whole space,
2. If $A \in \Sigma$, then $A^c \in \Sigma$: we want to be able to measure the complement of a measurable set,
3. If $A_1, A_2, \ldots \in \Sigma$, then $\bigcup_{n=1}^\infty A_n \in \Sigma$: we want to be able to measure countable unions of measurable sets.

Thus, when we define a measurable space $(X, \Sigma)$, we are essentially saying that we want to be able to measure the sets in $\Sigma$ and not the others. This is crucial because it allows us to avoid the paradoxes and contradictions that arise when trying to measure all subsets of $X$, which is impossible in general, as we have seen above with the Lebesgue measure on $\R$.[^ulam]

The natural question that arises is: **how do we choose a σ-algebra?** The answer depends on the context and the specific application. Logically, we want to have a σ-algebra which is as large as possible (i.e. not the trivial σ-algebra $\lbrace \emptyset, X\rbrace$) so that we can measure as many sets as possible, but not too large so that we avoid non-measurable sets.

In the next section, we will look at a specific example of a σ-algebra on $\R$ that is widely used in analysis and probability theory: the Borel σ-algebra.

### B. The Borel σ-algebra

First of all, note that if $X$ is finite (e.g. $X = \lbrace1, 2, 3\rbrace$) or countable (e.g. $X = \N$), then the power set $\mathcal{P}(X)$ is a σ-algebra and we can define a measure on it without any issues. Non-measurable sets only arise when $X$ is **uncountable**, which is the case for $\R$. In this case, the power set $\mathcal{P}(\R)$ is too large to be a σ-algebra for the Lebesgue measure, so we need to find a suitable σ-algebra $\Sigma \subsetneq \mathcal{P}(\R)$. This is where the *Borel σ-algebra* comes into play. Without going into too much detail, it is defined as follows:

> The Borel σ-algebra $\mathcal{B}(\R)$ is the smallest σ-algebra containing all open intervals in $\R$.

In other words, it is generated by the collection of all open sets in $\R$. The Borel σ-algebra is important because it provides a rich structure of measurable sets that can be used in analysis and probability theory. Thus, in nearly all practical applications, we will consider the measurable space $(\R, \mathcal{B}(\R))$ when working with the Lebesgue measure.[^lebesgue-algebra]

### C. Link with probability theory

Hopefully, by now the concepts of measurability and σ-algebras are clearer. Now we need to connect them to probability theory.

In probability theory, we are often interested in assigning probabilities to events, which can be thought of as subsets of a sample space. To do this in a rigorous way, we need to work within a measurable space. Specifically, we need a σ-algebra that contains all the events we want to assign probabilities to.

Let's consider a probability space $(\Omega, \A, \P)$ and a random variable $X: \Omega \to E$, where $(E, \mathcal{E})$ is another measurable space. For $X$ to be a valid random variable, it must be *measurable*, which means that for every set $B \in \mathcal{E}$, the preimage $X^{-1}(B) = \{\omega \in \Omega : X(\omega) \in B\}$ must belong to $\A$. This **ensures** that we can assign a probability to the event $X \in B$ using the probability measure $\P$.

One way to think about it is that we know how to measure things in $(\Omega, \A, \P)$, and we want to transfer this knowledge to $(E, \mathcal{E})$ through the random variable $X$. The measurability condition on $X$ ensures that the events in $E$ that we care about can be "pulled back" to events in $\Omega$ that we know how to measure. In the language of probability theory, this means that for any event $B$ in the target space $E$, we can compute its probability by looking at the corresponding event in the original space $\Omega$.

## Conclusion

Measure theory can easily be overlooked when first learning about probability theory. Yet, probability theory is ---arguably--- just a corollary of measure theory. That's why I think that thoroughly understanding the concepts of measurability and σ-algebras is crucial to grasp the axiomatic foundations of probability theory. Hopefully this post was clarifying in this regard. Note that it will be followed by another related post that deals with an even more obscure concept in probability theory: filtrations.

---

**References**:

[^measure-wiki]: If you're new to measure theory, the wikipedia page is a good starting point. [Wikipedia](https://en.wikipedia.org/wiki/Measure_(mathematics))
[^sigma-algebra]: The definition of a σ-algebra will come soon, for now assume that $\Sigma=\mathcal{P}(X)$ for simplicity.
[^single-dimension]: For the sake of simplicity, we will only consider the Lebesgue measure on $\R$. The concepts extend to $\R^n$ in a straightforward manner.
[^radon-nikodym]: More formally, we write $f=\frac{d\mu}{d\lambda}$ and call it the *Radon-Nikodym derivative* of $\mu$ with respect to $\lambda$. [Wikipedia](https://en.wikipedia.org/wiki/Radon%E2%80%93Nikodym_derivative)
[^zf]: And indeed, one can show that removing the Axiom of Choice from Zermelo-Fraenkel set theory (i.e. using ZF instead of ZFC) makes all subsets of $\R$ Lebesgue measurable. [Wikipedia](https://en.wikipedia.org/wiki/Zermelo%E2%80%93Fraenkel_set_theory)
[^vitali-construction]: Vitali sets are constructed by first quotienting $\R$ by $\Q$ (the set of rational numbers), and then using the Axiom of Choice to select one representative $\tilde{r}\in\R$ for each equivalence class in $\R \backslash \Q$, with the condition that this representative lies in the interval $[0, 1]$. The resulting set $V$ is a Vitali set, and it can be shown that $V$ is non-measurable with respect to the Lebesgue measure.
[^boolean]: These properties seem very intuitive when you replace "measurable set" with "event" in the context of probability theory. If we want assign a probability to an event, we also want to be able to assign a probability to its complement and to countable unions of events.
[^ulam]: Note that the existence of non-measurable sets is not due to some pathological property of the Lebesgue measure. In fact, a result known as Ulam's theorem states that there exist no atomless probability measure on the probability space $(\R, \mathcal{P}(\R))$. In other words, whatever atomless probability measure we try to define on $\mathcal{P}(\R)$ is doomed to fail. [Wikipedia](https://fr.wikipedia.org/wiki/Th%C3%A9or%C3%A8me_d%27Ulam)
[^lebesgue-algebra]: Technically, the Lebesgue σ-algebra $\mathcal{L}(\R)$ is the *completion* of the Borel σ-algebra $\mathcal{B}(\R)$ with respect to the Lebesgue measure. This means that it contains all Borel sets as well as all subsets of Borel sets that have Lebesgue measure zero (and these new sets will be given measure zero). Thus, the Lebesgue σ-algebra is strictly larger than the Borel σ-algebra, and it is the natural domain for the Lebesgue measure. In practice we tend to abuse notation and refer to the Borel σ-algebra when we actually mean the Lebesgue σ-algebra.