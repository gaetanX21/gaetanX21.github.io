---
layout: post
title: "Solving the assignement problem using Optimal Transport"
date: 2024-11-15
description: "TL;DR: The discrete Kantorovich problem amounts to a LP problem. In the uniform case, the solution is a permutation matrix which in fact solves the assignement problem."
tags: optimal-transport
thumbnail: assets/img/posts/ot_permutation_problem/ranks.png
---

$$
\newcommand{\R}{\mathbb{R}}
\newcommand{\tn}[1]{\textnormal{#1}}
$$

We first introduce the discrete Kantorovich problem and show that in the uniform case it amounts to solving the permutation problem. We then illustrate this with a student internship assignement problem. We run Monte Carlo simulations for different cost functions and show that the choice of cost function crucially impacts the optimal assignement.

## The discrete Kantorovich Problem
Let $X, Y$ be two measurable spaces (for simplicity, $X=Y=\R^d$). Consider two discrete distributions (i.e. weighted point clouds) $\alpha\in \tn{P}(X), \ \beta\in\tn{P}(Y)$ given by
$$
\begin{equation}
    \label{eq:def}
    \alpha = \sum_{i=1}^n a_i \delta_{x_i}, \quad \beta = \sum_{j=1}^m b_j \delta_{y_j},
\end{equation}
$$
and a cost function $c:X\times Y \rightarrow \R^+$.

The discrete Kantorovich problem then formulates as:
$$
\begin{equation}
    \label{eq:K}
    \tag{K}
    P^\star = \arg  \min_{P\in U(\alpha,\beta)} \langle C,P\rangle
\end{equation}
$$
where $C=\big(c(x_i,y_j)\big)_{i,j} \in \R^{n\times m}$ and $U(\alpha,\beta)=\lbrace P\in \R^{n\times m} | P\geq 0, P\mathbb{1}_m=a, P^T \mathbb{1}_n=b \rbrace$.

Notice that $P\mapsto \langle C,P \rangle$ is a convex functional and $U(\alpha,\beta)$ is a convex subset of $\R^{n\times m}$, such that (\ref{eq:K}) is a convex problem.

Even better, it is a linear programming (LP) problem since $P\mapsto \langle C,P \rangle$ is linear and $U(\alpha,\beta)$ encodes linear constraints.

Thus, in the discrete case, Optimal Transport (OT) can be seen as an LP problem, and thus solved with off-the-shelf LP solvers such as the `cvxpy` Python library.

## The Uniform Case
Let's consider the uniform case i.e. $n=m$ and $a_i=b_j=\frac{1}{n} \ \forall i,j$.

In that scenario, one can show that there exists at least one OT coupling $P^\star$ which is a permutation matrix. This comes from the fact that the extremal points of the polytope $U(1,1)$ are permutation matrices.

Thus, in the uniform case there exists a permutation $\sigma^\star \in S_n$ such that $P^\star=P _ {\sigma^\star}=\big( \mathbb{1} _ {\sigma^\star(i)=j} \big) _ {i,j}$. In particular, $\sigma^\star$ solves the permutation problem
$$
\begin{equation}
    \label{eq:permutation-problem}
    \tag{PP}
    \sigma^\star = \arg \min_{\sigma\in S_n} \sum_{i=1}^n C_{i,\sigma(j)}
\end{equation}
$$

## Student Internship Assignment
To illustrate the method described, let's apply the uniform case, which solves the permutation problem, to assign $n$ students $x_i$ to $n$ internships $y_j$ in a *optimal* manner.

Let's consider that each student $x_i$ expresses their preference through a ranking $\sigma_i$ of the internships where $\sigma_i(j)$ is the ranking of internship $y_j$ according to student $x_i$ (i.e. $\sigma_i(j)=1$ for $x_i$'s dream internship and $\sigma_i(j)=n$ for $x_i$'s least desired internship).

There are many possible choices for the cost function $c$, but it must clearly be an increasing function of $\sigma_i(j)$. The most natural is probably $c(x_i,y_j)=\sigma_i(j)$ i.e. a linear penalization of the integer distance between the student's favorite ($c=1$) and least wanted internship ($c=n$). However, the optimal assignment $P^\star=P_{\sigma^\star}$ depends crucially on the choice of $c$! Intuitively, rapidly increasing function e.g. quadratic cost $c(x_i,y_j)=\sigma_i(j)^2$ will prevent any student from being attributed an internship deemed too undesirable. This means no student will get an awful internship, the hidden cost being that presumably fewer student will get their first wish. On the contrary, a slowly increasing function e.g. log cost $c(x_i,y_j)=\log\sigma_i(j)$ will only slightly penalize poor internship attributions, and thus we except to see lots of students get their first wish alongside a handful of students getting very low-ranked internships.

We test those intuitions by running Monte Carlo simulations for each of the aforementioned cost functions (linear, quadratic, log). More precisely, for a given cost function $c$, we run $M$ simulations, each with $n$ students. Each simulation returns a integer array `ranks` of length $n$ where `ranks[i]` is student i's ranking of the internship they were attributed. For each cost function $c$, We concatenate the $M$ `ranks` arrays and then plot a histogram of their distribution.

The results are presented in [Figure 1](#fig-assignment) and confirm our intuition, although there is no difference between linear and quadratic cost. We used $n=20$ students and ran $M=100$ iterations for each cost function $c$.

<div class="row justify-content-center" id="fig-assignment">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/ot_permutation_problem/ranks.png" title="ranks" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 1. Empirical distribution of students’ ranking of their obtained internship for different cost functions $c$. The log penalty increases slowly such that it’s tolerable to highly disappoint a handful of students if that can help the majority obtain their first wish. This is not the case for the linear and quadratic penalties, which penalize highly the worst attributions.
</div>

## Conclusion

We have shown that discrete OT amounts to a LP problem. However, LP problems do not scale well. This motivates the introduction of entropic regularization, which makes (\ref{eq:K}) much easier and faster to solve when $n$ becomes too large for a LP approach. We will discuss this in a future post.
