---
layout: post
title: "The magic of Embeddings"
date: 2025-05-17
description: "TL;DR: Embeddings are so powerful that they can seem almost magical. We go back to the basics (linear algebra) with the Johnson-Lindenstrauss lemma, which illustrates the blessing of dimensionality."
tags: linear-algebra
thumbnail: assets/img/posts/embeddings/mnist_tsne.jpg
---

$$
\newcommand{\R}{\mathbb{R}}
\newcommand{\tn}[1]{\textnormal{#1}}
$$

In this post, I will discuss the magic of embeddings and then move onto the Johnson-Lindenstrauss lemma, which is a fundamental result in linear algebra that illustrates the blessing of dimensionality. I will also give a sketch of the proof of the lemma, which is based on the idea of random projections. Finally, I will briefly mention the LinFormer paper, which proposes a linear time and space complexity self-attention mechanism for transformers based on the JL lemma.

---

<div class="row justify-content-center" id="fig-1">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/embeddings/mnist_tsne.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 1. 3D t-SNE embeddings of MNIST data. <a href="https://towardsdatascience.com/visualizing-bias-in-data-using-embedding-projector-649bc65e7487/">Source</a>.
</div>

## I. Motivation

Embeddings have always fascinated me for (at least) three reasons:

1) they compactly store large amounts of information (e.g. LLM token embeddings which essentially encapsulate all the subtleties of human language)

2) they have meaningful geometric properties (e.g. the dot product encodes similarity, `queen`-`king`+`man`=`woman`, etc.)

3) they can accommodate different modalities (e.g. CLIP embeddings which can encode both text and image data)

This begs the question:
> How come dense vector representations work so well?

**As often in machine learning, behind the magic lurks good old linear algebra**. In the case of embeddings, the *blessing of dimensionality* is at play. To put it simply, the sheer size of the embedding space allows for a lot of flexibility and expressiveness.

In a famous paper[^superposition], researchers at Anthropic study the phenomenon of *superposition* in large language models. They show that the model can learn to represent multiple concepts in a single embedding, which is a direct consequence of the high dimensionality of the space. In particular, they highlight the fact that although a $d$-dimensional space can only hold $d$ orthogonal vectors, if we allow for quasi-orthogonal vectors, we can fit a much larger number of them, which is in fact exponential in $d$! This is a consequence of the *Johnson-Lindenstrauss lemma*, which we introduce and prove in the next section.

## II. The Johnson-Lindenstrauss lemma

> The Johnson-Lindenstrauss lemma[^jl] (1984) or "JL lemma" states that a set of points in a high-dimensional space can be embedded into a lower-dimensional space while preserving pairwise distances approximately.

In other words, the JL lemma guarantees the existence of low-distortion embeddings for any finite set of points in a high-dimensional space. This is particularly useful in machine learning, where we often deal with high-dimensional data and need to reduce its dimensionality for various tasks such as visualization, clustering, or classification.

The JL lemma can be formally stated as follows:

**Lemma (Johnson–Lindenstrauss).**  
Let $\epsilon \in (0, 1)$, $X$ a set of $N$ points in $\mathbb{R}^n$, and consider an integer $k>\frac{8 \ln(N)}{\epsilon^2}$. Then, there exists a linear map $f:\mathbb{R}^n \to \mathbb{R}^k$ such that for all $u,v \in X$:

$$
(1-\epsilon) \|u-v\|^2_2 \leq \|f(u)-f(v)\|^2_2 \leq (1+\epsilon) \|u-v\|^2_2
$$

NB: The bound on $k$ is tight i.e. there exists a set $X$ that needs dimension $\Omega(\frac{\ln(N)}{\epsilon^2})$ to be embedded with distortion $\epsilon$.

NB: Interestingly enough, the bound on $k$ is independent of the original dimension $n$! This means that in theory, if we have say $N=10^6$ points living in dimension $n=10^{83}$, we can project them down to $k=\frac{8 \ln(10^6)}{0.1^2} \approx 10^4$ dimensions while preserving pairwise distances with a distortion of $10\%$! [^catch]


## III. Proof of the lemma

I find the *proof* of the JL lemma interesting in its own right. It is based on the idea of random projections, which are linear maps that project high-dimensional data onto a lower-dimensional subspace. The key idea is the following: if we randomly choose a projection from the $\R^n$ to $\R^k$, there is a non-zero probability that the projection will preserve the pairwise distances of all the points in $X$ up to a factor of $(1+\epsilon)$. And because this probability is non-zero, it means that such projections must exist!

NB: This proof technique is called the "probabilistic method": we use a probabilistic argument to state a deterministic result. In particular, the lemma does not give us a constructive way to find a working $\R^n \to \R^d$ projection, but rather guarantees that at least one exists.

We first present a high-level sketch of the proof, followed by a more rigorous step-by-step derivation.

#### A. Sketch of the proof
Here I will give a sketch of the proof in two parts:

1) I'll show that if we randomly project a vector $u \in \R^n$ onto a $k$-dimensional subspace $v\in \R^k$ with $k>\frac{8 \ln(N)}{\epsilon^2}$, then we have 

$$
\mathbb{P}(\|v\|^2_2 \in \big[(1-\epsilon) \|u\|^2_2, (1+\epsilon) \|u\|^2_2\big]) \geq \frac{2}{N^2}
$$

2) From this result I will show that if we have $N$ points in $\R^n$, then the probability that all of them are projected into a $k$-dimensional subspace with distortion $\epsilon$ is non-zero, effectively proving the lemma.

#### B. Actual derivation

1) Let $u \in \R^n$ and let $k$ be some integer which we'll fix later. Consider $P \sim \mathcal{N}(0,1)^{\otimes (k,n)}$ a random projection matrix of from $\R^n$ to $\R^k$ and define $v=\frac{1}{\sqrt{k}}Pu$. For ease of notation, we write $P$ as:

$$
P = 
\begin{bmatrix}
P_1^T \\
\vdots \\
P_k^T 
\end{bmatrix}
$$

It is then clear that $v_i = \frac{1}{\sqrt{k}}P_i^Tu \sim N(0,\frac{\|u\|^2_2}{k}) \ i.i.d.$ for $i=1,\ldots,k$. As such, we can define $x=\frac{\sqrt{k}}{\|u\|_2}v$ and we have $x \sim N(0,I_k)$. Consequently, we have:

$$
\|x\|^2_2 = \frac{1}{k}\frac{\|v\|^2_2}{\|u\|^2_2} \sim \chi^2_k
$$

From there, we can easily use concentration inequalities on the $\chi^2$ distribution to show that:

$$
\mathbb{P}(\|v\|^2_2 \in [(1-\epsilon) \|u\|^2_2, (1+\epsilon) \|u\|^2_2]) \geq 2e^{-\frac{k}{4}(\epsilon^2-\epsilon^3)}	
$$

We then fix $k>\frac{8 \ln(N)}{\epsilon^2}$ which gives us the desired $\frac{2}{N^2}$ bound.

2) Now, let $X=\lbrace x_1,\ldots, x_N \rbrace$ be a set of $N$ points in $\R^n$. The above result applies to all the vectors $u = x_i - x_j$ for all pairs $1\leq i,j \leq N$. Let $E_{\lbrace i,j\rbrace}$ be the event that the projection of the pair $\lbrace x_i,x_j\rbrace$ violates the distortion bound. There are $N(N-1)/2$ $\lbrace i, j \rbrace$ pairs, such that the probability of having at least one of them violate the distortion bound $\epsilon$ is given by:

$$
p_\text{invalid projection} = \mathbb{P}(\bigcup_{\{i,j\} \in pairs} E_{\{i,j\}}) \leq \sum_{\{i,j\} \in pairs} \mathbb{P}(E_{\{i,j\}}) \leq \frac{N(N-1)}{2}\frac{2}{N^2} = 1-\frac{1}{N}
$$

Consequently,

$$
p_\text{valid projection}=1-p_\text{invalid projection} \geq \frac{1}{N} > 0
$$

Thus, when sampling a random projection $P$ from $\mathcal{N}(0,1)^{\otimes (k,n)}$, we have a non-zero probability that all the points in $X$ are projected into a $k$-dimensional subspace with distortion $\epsilon$.

This proves the lemma.

## IV. LinFormer

I won't delve into the many real-life corollaries of the JL lemma, since essentially all linear dimensionality reduction techniques implicitly rely on it.

However, I will mention the LinFormer paper[^linformer] which proposes a **linear time and space complexity self-attention mechanism for transformers**. The key idea is to use a low-rank approximation of the attention matrix[^spectrum], which can be achieved using random projections. The JL lemma is used by the authors to provide theoretical guarantees for this approximation: they demonstrate that for a given distortion $\epsilon$, there is a corresponding dimension $k<n$[^n] which ensures that the rank-$k$ approximation induces an $\epsilon$-bounded distortion!

## Conclusion

While the Johnson-Lindenstrauss lemma is not directly "practical" in itself, I believe it is the kind of linear algebra result that is good to keep in mind. In particular, I think it helps build a better intuition of what happens in high-dimensional spaces, where both the curse *and* the blessing of dimensionality are at play.

---

**References**:

[^jl]: Wikipedia. *Johnson-Lindenstrauss lemma.* [Link](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma) 
[^superposition]: Anthropic (2022). *Toy Models of Superposition.* [Link](https://www.anthropic.com/news/toy-models-of-superposition)
[^catch]: The catch, however, is that finding a projection that works would take a lot of time in practice, since this time would scale with  the initial dimension $n$.
[^linformer]: Wang, S., et al. (2020). *Linformer: Self-Attention with Linear Complexity.* [Link](https://arxiv.org/abs/2006.04768)
[^spectrum]: The paper also studies the spectrum of attention matrices and shows that they are low-rank, which is a key insight for the LinFormer approach. Even more interestingly, they show that as we go deeper in the transformer, the attention matrices become more and more low-rank, which is to say the information becomes more and more compressible!
[^n]: Here, $n$ is the sequence length.