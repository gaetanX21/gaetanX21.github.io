---
layout: post
title: "Residual Matrix Transformers"
date: 2025-11-11
description: "TL;DR: As we increase the size of (standard) transformers, parameters and FLOPs scale quadratically, but the residual stream width scales linearly. Thus, the residual stream can become a bottleneck as we scale up. We discuss the RMT paper, which proposes a matrix residual stream to address this issue."
tags: llm, pretraining, research
thumbnail: assets/img/posts/residual_matrix_transformer/rmt.png
---

$$
\newcommand{\R}{\mathbb{R}}
\newcommand{\X}{\mathbb{X}}
$$

A colleague at work recently shared a paper[^paper] from ICML 2025 entitled *Residual Matrix Transformers: Scaling the Size of the Residual Stream*. I found it so interesting that I decided to write a short summary of the main ideas.

The starting point of the paper is the observation that, as we scale up transformers, model parameters and FLOPs scale *quadratically* while the residual stream width scales *linearly*. Hence, as we keep increasing model size, the residual stream may become a bottleneck. The authors' solution is to replace the vector token representation with a matrix representation inspired by outer-product memories. The resulting **Residual Matrix Transformer** (RMT) architecture is more efficient in terms of parameters and FLOPs while outperforming the standard transformer on downstream evaluations after pretraining. Additionally, the size of the residual stream can be increased without (significantly) increasing the number of parameters or FLOPs, effectively creating a new scaling dimension with its own scaling law. Finally, the RMT exhibits improved variance propagation properties.

---

<div class="row justify-content-center" id="fig-1">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/residual_matrix_transformer/rmt-wide.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 1. To read features from the residual stream, the standard transformer computes (learned) linear transformations of the residual stream vector, whereas the Residual Matrix Transformer uses (learned) key vectors applied to the residual stream matrix.
</div>

## I. Outer-product memory

It may at first seem arbitrary to replace the vector token representation with a matrix representation. However, this design choice makes sense if we interpret the residual stream as an **outer-product memory store**.

An outer-product memory is a type of *content-addressable memory* (or *associative memory*), i.e. a memory model where data is accessed not with a specific address (e.g. give me the item at position 4) but with a content-based query (e.g. give me the word closest to "appl"). This is loosely inspired by how the human brain works, where we associate certain cues with specific memories (e.g. if you think of potatoes, you won't be accessing a specific "potato" address in your brain, but rather you'll recall a flurry of memories associated with potatoes).

In the case of outer-product memories, we want to store a set of $n$ key-value pairs $\lbrace k_i, v_i \rbrace_{i=1}^n$ where $k_i\in\R^{D_k}, v_i\in\R^{D_v}$.
We store these pairs in a matrix $M \in \R^{D_k \times D_v}$ defined as

$$M = \sum_{i=1}^n k_i \otimes v_i$$

where $\otimes$ denotes the outer product i.e. $k_i \otimes v_i = k_i v_i^\top$.

Then, to retrieve the value $v_p$ associated with the key $k_p$, we compute the dot product $k_p \cdot_1 M$ where $\cdot_1$ denotes the dot product (or "tensor contraction") along the first dimension. We thus have

$$\hat{v}_p = k_p \cdot_1 M =\sum_{i=1}^n (k_p \cdot k_i) v_i = v_p + \text{cross-talk noise}$$

where we hope that the keys $k_i$ are sufficiently orthogonal to each other so that cross-talk noise is negligible.

Thus, the outer-product memory $M$ has two operations:
- `READ`: $v \leftarrow k \cdot_1 M$
- `WRITE`: $M \leftarrow M + k \otimes v$

The RMT paper uses this outer-product memory as a building block to construct a matrix residual stream.

## II. Model Architecture

Let's now see in practice how the RMT implements the two main transformer operations: the attention block and the feedforward block.

We will consider a single sequence of $N$ tokens, thus the residual stream is:
- a matrix $X \in \R^{D_{model} \times N}$ for the standard transformer
- a tensor $\X \in \R^{D_k \times D_v \times N}$ for the RMT

### A. Attention Block

For the attention block, there are two differences between the standard transformer and the RMT:
1. How information is **read** from the residual stream to compute the query, key and value matrices.
2. How the output of the attention block is **written** to the residual stream.

#### Reading information from the residual stream

In the standard transformer, the query, key and value matrices for each attention head are computed as follows:

$$Q^h = W_Q^h X, \quad K^h = W_K^h X, \quad V^h = W_V^h X$$

where $W_Q^h, W_K^h, W_V^h \in \R^{D_h \times D_{model}}$ are learnable matrices.

In the RMT, the query, key and value matrices are computed as follows:

$$Q^h = r_Q^h \cdot_1 \X, \quad K^h = r_K^h \cdot_1 \X, \quad V^h = r_V^h \cdot_1 \X$$

where $r_Q^h, r_K^h \in \R^{D_k}$ and $r_V^h \in \R^{D_v}$ are learnable vectors.

Thus, the key, query, and value matrices are obtained
- with a *linear transformation* of the residual stream matrix $X$ in the standard transformer
- with a `READ` operation on the residual stream tensor $\X$ in the RMT

*Note that once the key, query, and value matrices are computed, single-head attention is computed in the same way in both models. In particular, this means that the KV-cache is the same in both models (so, no explosion of memory usage because of the matrix residual stream).*

#### Writing information to the residual stream

In the standard transformer, the output of the multihead attention block is computed as follows:

$$\text{MHA}(X) = \sum_{h=1}^H W_{out}^h \text{SHA}(Q^h, K^h, V^h) \in \R^{D_{model} \times N}$$

where $W_{out}^h \in \R^{D_{model} \times D_h}$ is a learnable matrix.

In the RMT, the output of the attention block is computed as follows:

$$\text{MHA}(\X) = \sum_{h=1}^R w_{out}^h \otimes \text{SHA}(Q^h, K^h, V^h) \in \R^{D_k \times D_v \times N}$$

where $w_{out}^h \in \R^{D_k}$ is a learnable vector and $R$ is a hyperparameter[^R].

Thus, once the output of each attention head $\text{SHA}(Q^h, K^h, V^h)$ is computed, it is written back to the residual stream
- with a *concatenation then linear transformation* of the $H$ heads in the standard transformer
- with a `WRITE` operation for each individual head in the RMT

*In particular, notice how scaling $D_k$ and $D_v$ in the RMT results in a linear increase in the number of parameters and FLOPs, while scaling $D_{model}$ in the standard transformer results in a quadratic increase in the number of parameters and FLOPs. This is because we've replaced the matrices for reading from ($W_Q, W_K, W_V$) & writing to ($W_{out}$) the residual stream with vectors for `READ` ($r_Q, r_K, r_V$) and `WRITE` ($w_{out}$) operations.*

### B. Feedforward Block
We could be tempted to apply the same treatment to the feedforward block. That is, use vectors instead of matrices to read from & write to the residual stream. However --- as the authors point out --- there is strong evidence that the feedforward weights actually store factual information in the transformer[^interpretation], so we don't want to replace them with vectors!

The authors' solution is to transform $\X$ a bit to recover the standard feedforward block. Specifically, we retrieve $R$ data vectors from $\X$ using `READ` operations and concatenate them to form a matrix

$$X_{FF} = \text{concat}_{1\leq h\leq R}(r_{FF}^h \cdot_1 \X) \in \R^{RD_v \times N}$$

which is then fed to the feedforward block. The output $\text{FF} _ {standard}(X_{FF})\in\R^{RD_v \times N}$ of the standard feedforward block is then reshaped into $R$ matrices with the $\text{unvec}_1$ operation along the first dimension, and each of these matrices is written in the residual stream using `WRITE` operations with the key vectors $w _ {FF}^h \in \R^{D_k}$, such that the final output of the feedforward block is

$$\text{FF}(\X) = \sum_{h=1}^R w_{FF}^h \otimes \text{unvec} _ 1(\text{FF}_{standard}(X_{FF}))_h \in \R^{D_k \times D_v \times N}$$


## III. Theoretical properties

I'll go quickly here, more details can be found in the paper. There are two key theoretical properties of the RMT:
1. **Resource scaling**: we can scale the size of the residual stream without significantly increasing the number of parameters or FLOPs, thus creating a new scaling dimension with its own scaling law.
2. **Variance propagation**: the RMT has better variance propagation properties than the standard transformer, i.e. the analysis of retrieval and storage operations shows the variance of activations & gradients is better preserved in the RMT than in the standard transformer. (i.e. both $\frac{\sigma^2 _ {x_{out}}}{\sigma^2 _ {x_{in}}}$ and $\frac{\sigma^2 _ {g_{out}}}{\sigma^2 _ {g_{in}}}$ are closer to $1$ in the RMT than in the standard transformer)


## IV. Pretraining performance

The authors conducted several pretraining experiments (with models ranging from 46M to 405M parameters, arguably small by today's standards) to compare the RMT to the standard transformer. They found that the RMT is:
1. more parameter efficient (because the read/write weight matrices are replaced with key vectors)
2. more data efficient (i.e. faster convergence on a train-token basis)
3. more compute efficient (essentially a corollary of the above two points)
4. more memory efficient (same footprint during training, lower footprint at inference[^memory])
5. slightly worse at runtime (however, this is largely due to the current suboptimal implementation of the RMT)


## V. Conclusion

Overall, the Residual Matrix Transformer feels as significant as Mixture of Experts models (MoEs) in the sense that it offers a new dimension along which to scale transformers. On a more personal level, I also find the interpretation of the residual stream matrix as an outer-product memory very elegant and satisfying!

Also, unlike (somewhat) radical transformer alternatives like State-Space Models or Recurrent Neural Networks, the RMT architecture is close enough to the standard transformer that we can actually hope to see it implemented in the near future! It would be interesting to see whether this residual matrix approach holds up at the billion (and trillion) parameter scale, enabling us to further scale transformers.

---

**References**:

[^paper]: Mak, B., & Flanigan, J. (2025). *Residual Matrix Transformers: Scaling the Size of the Residual Stream.* [[arXiv](https://arxiv.org/abs/2506.22696)]
[^interpretation]: The common interpretation is that attention layers allow tokens to talk to each other (communication step) while feedforward layers let tokens think on their own after having communicated (processing step).
[^R]: $R$ is not simply the number of heads, but more generally the number of `WRITE` operations in the attention layers and `READ` operations in the feedforward layers.
[^memory]: During training, the increased size of the residual stream directly translates into more memory usage because of larger gradient checkpoints, but this memory cost is offset by the reduced model size. At inference time, we don't checkpoint gradients anymore and thus RMT actually becomes more memory efficient because of its smaller size.