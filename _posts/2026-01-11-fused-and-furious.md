---
layout: post
title: "Fused & Furious: Sinkhorn Triton Kernels"
date: 2026-01-11
description: "TL;DR: DeepSeek's recent mHC paper relies on Sinkhorn's algorithm to project matrices onto Birkhoff's polytope. The looping nature of the algorithm introduces severe memory-boundedness, which can be mitigated by fusing the algorithm into a single kernel. We implement increasingly fast versions of the algorithm in Triton."
tags: llm, pretraining, research
thumbnail: assets/img/posts/fused_and_furious/fused_and_furious.png
---

$$
\newcommand{\R}{\mathbb{R}}
\newcommand{\Rnn}{\mathbb{R}^{n\times n}}
\newcommand{\D}{D_{model}}
\newcommand{\x}{\mathbf{x}}
\newcommand{\F}{\mathcal{F}}
\newcommand{\Hpre}{\mathcal{H}^{pre}}
\newcommand{\Hpost}{\mathcal{H}^{post}}
\newcommand{\Hres}{\mathcal{H}^{res}}
\newcommand{\B}{\mathcal{B}_n}
\newcommand{\one}{\mathbb{1}_n}
\newcommand{\T}{\text{Tr}}
\newcommand{\KL}{\text{KL}}
$$

In their last paper, charmingly titled *Manifold-Constrained Hyper-Connections*[^mhc], DeepSeek improved upon preexisting research on **residual stream scaling** with two major contributions:
1. an approach that's much stabler than previous papers, with demonstrated performance at scale
2. an efficient infrastructure design that minimizes the overhead of hyper-connections compared to vanilla transformers

While I encourage you to read the full paper, my goal in this post is to delve deeper into the second part. More precisely, I want to focus on the Sinhkorn projection step of their architecture: why we need custom fused GPU kernels to implement it, and how to do it. I'll first discuss the theory around *mHC* and Sinkhorn, then I'll showcase increasingly fast (and tricky) Triton kernels for the Sinkhorn algorithm.


---

<div class="row justify-content-center" id="fig-1">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/residual_matrix_transformer/rmt-wide.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 1. To read features from the residual stream, the standard transformer computes (learned) linear transformations of the residual stream vector, whereas the Residual Matrix Transformer uses (learned) key vectors applied to the residual stream matrix.
</div>

## I. Scaling the residual stream

A current trend in macro design of transformers is to scale the **size** of the residual stream. That is, instead of being a vector $\x\in\R^\D$, the per-token residual stream becomes a matrix $X\in\R^{n\times\D}$ where $n$ is a new scaling dimension (often a small value e.g. $n=4$). The motivation behind this architectural shift is that in vanilla transformers, model parameters and FLOPs scale *quadratically* with $\D$ whereas the residual stream scales *linearly*. Hence, as we keep increasing model size, the residual stream may become a **bottleneck**.

### A. Residual Matrix Transformers

My [previous post](/blog/2025/residual-matrix-transformer/) discussed the **Residual Matrix Transformer** (RMT) architecture, which introduces a matrix representation of the per-token residual stream inspired by outer-product memories. The key idea is that using a matrix instead of a vector increases the effective storage capacity of the residual stream. The implementation of RMT is fairly simple, as the only difference compared to the classic transformer architecture is how we *read from* and *write to* the residual stream. Crucially, the micro design remains untouched (i.e. Attention and Feed-Forward blocks do not change), which implies that scaling the residual stream incurs negligible computational overhead, though it does cause memory access overhead if implemented naively (i.e. without fused kernels) as we will see.

In a nutshell, RMT replaces the classic residual connection

$$\x_{l+1} = \x_l + \F_l(\x_l)$$

with

$$X_{l+1} = X_l + {\Hpost}^T \F_l(\Hpre X_l)$$

where $\x_l\in\R^{\D}$ (resp. $X_l\in\R^{n\times\D}$) is the $l$-th layer vector (resp. matrix) residual stream, and $\F_l$ is the $l$-th transformer block.

$\Hpre_l, \Hpost_l \in \R^{1\times n}$ are *learned* vectors which correspond respectively to `READ` and `WRITE` operations on the residual stream memory store. 


### B. Hyper-Connections

Hyper-Connections[^hc] (HC) extend the idea of RMT by enabling *communication* between the $n$ channels of the residual stream.

The motivation is to increase the topological complexity of the residual mapping without increasing the computational complexity. The main improvement over RMT is the introduction of $\Hres_l\in\Rnn$ which acts as a "mixing matrix" allowing the $n$ channels to exchange information instead of simply evolving independently. Also, $\Hpre_l, \Hpost_l, \Hres_l$ are now both *static* (i.e. learned parameters) and *dynamic* (i.e. dependent on input $X_l$) to further increase the topological complexity of the connection network (see original paper for more details). Note that we will omit the dependence on $X_l$ to keep notations lightweight (e.g. we'll write $\Hres_l$ instead of $\Hres_l(X_l)$)

Thus, the residual connection becomes even more rich than RMT:

$$X_{l+1} = \Hres_l X_l + {\Hpost}^T \F_l(\Hpre X_l)$$


### C. Manifold-Constrained Hyper-Connections

DeepSeek's Manifold-Constrained Hyper-Connections (mHC) address one critical flaw in HC: instability as depth $L$ scales. Indeed, HC abandons the identity mapping of residual connections, which is crucial to the training stability of deep neural network architectures[^resnet]. It's easy to see why that is problematic. If we recursively extend the residual connection across multiple layers, we have:

1. $$\x_{l+m} = \x_l + \sum_{k=0}^{m-1}\F_k(\x_{l+k})$$ for the standard transformer
2. $$X_{l+m} = X_l + \sum_{k=0}^{m-1}{\Hpost_{l+k}}^T \F_k(\Hpre_{l+k} X_{l+k})$$ for RMT
3. $$X_{l+m} = \big(\prod_{k=0}^{m-1}\Hres_{l+k}\big) X_l + \sum_{k=0}^{m-1}{\Hpost_{l+k}}^T \F_k(\Hpre_{l+k} X_{l+k})$$ for HC

The matrix product $\Pi=\prod_{k=0}^{m-1}\Hres_{l+k}$ has no reason to behave nicely and keep a spectral norm close to 1. In fact, DeepSeek shows that instead of preserving the signal strength, $\Pi$ tends to amplify or attenuate it as depth increases, resulting in signal rescaling across **several orders of magnitudes**, which is problematic for both the forward pass (exploding/vanishing activations) and the backward pass (exploding/vanishing) gradients. The propagation of instability as depth increases is illustrated in [Figure 3](#fig-3). 

<div class="row justify-content-center" id="fig-3">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/fused_and_furious/27b_forward_backward_gain.pdf" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 3. This figure illustrates the propagation dynamics of (a) the single-layer mapping $\Hres_l$ and (b) the composite mapping $\prod_{i=1}^{L-1}\Hres_{L-i}$ within a 27B model. The layer index $l$ (x-axis) unrolls each standard Transformer block into two independent layers (Attention and FFN). The <i>Amax Gain Magnitude</i> (y-axis) is calculated as the maximum absolute row sum (for the forward signal) and column sum (for the backward gradient), averaged over all tokens in a selected sequence. Figure taken from DeepSeek <i>m</i>HC paper.
</div>

DeepSeek's solution to this instability problem is to constrain $\Hres_l$ to be a **bistochastic matrix** (i.e. a square matrix with non-negative entries where each row and column sums to 1). Bistochastic matrices have three nice properties that make them ideal for stabilizing the residual stream:
1. they have a spectral norm of 1, meaning that they preserve signal strength across layers
2. they are closed under matrix multiplication, meaning that the product of multiple bistochastic matrices is also bistochastic
3. they can be given a probabilistic / optimal transport interpretation as *soft permutations*, meaning that they allow each channel to attend to and exchange with every other channel, without collapsing to a single channel

I'm a bit surprised that the authors of HC did not not think of the bistochastic constraint in the first place, as it seems like an intuitive solution to the instability problem. Perhaps they lacked an efficient implementation of the projection step, which is precisely the focus of this post!

Indeed, to enforce the bistochastic constraint, we must first compute $\Hres_l$ as in HC, and then project it onto the manifold of bistochastic matrices $\B$, also known as Birkhoff's polytope. The next section goes into the details of this projection, which is known as Sinkhorn's algorithm.


## II. Sinkhorn's Algorithm

The goal of Sinkhorn's algorithm is to project a given square matrix $M\in\Rnn$ onto the Birkhoff polytope

$$\B=\lbrace P\in\Rnn |P\one=\one, \one^P=\one, P\geq 0\rbrace$$.

Note that $\B$ is convex as we will use this later, in fact it is the convex hull generated by the $n!$ permutation matrices in $\Rnn$.

We will first present the Sinkhorn algorithm, then derive it intuitively.

### A. Projecting under the generalized KL divergence

We've been talking about projections, but to define a projection, one needs a metric. In other words, we need a sensible metric $d: \Rnn \times \Rnn \to \Rnn$ and then we can try to find
$$\text{Proj}^{\B}(M) = \arg\min_{P \in \B} d(P,M)$$
for a given matrix $M \in \Rnn$.

One classic metric on $\Rnn$ is the one induced by the Frobenius norm ${\Vert M \Vert} _ F = \sqrt{\T (MM^T)} = \sqrt{ \sum_{1 \leq i,j \leq n} M_{ij}^2}$. However, using this metric would mean we're doing a Euclidean ($L^2$) projection, which intuitively doesn't feel right here given the probabilistic / optimal transport interpretation of bistochastic matrices. It would also require solving a linear programming problem in $O(n^3)$, which isn't ideal.

Instead, one much more sensible metric[^divergence] is the KL divergence. Since we're dealing with bistochastic matrices, we use the generalized KL divergence:
$$\KL(P\Vert M) = \sum_{1 \leq i,j \leq n} \left( P_{ij} \log\left(\frac{P_{ij}}{M_{ij}}\right) - P_{ij} + M_{ij} \right)$$

The extra terms serve a purpose: $P_{ij}$ ensures $\KL(P\Vert M)$ is minimized when $P=M$ and $M_{ij}$ guarantees non-negativity.

Thus, the problem we want to solve is:

$$\min_{P \in \B} \KL(P\Vert M) \quad (S)$$

Crucially, $P \mapsto \KL(P\Vert M)$ is strictly convex and $\B$ is convex, meaning that $(S)$ has a unique solution!


### B. Deriving the algorithm

Now that we've defined the projection as solving $(S)$, let's derive Sinkhorn's algorithm. We'll tackle this problem just like any optimization problem, using Lagrange's multipliers. We introduce the Lagrangian:

$$\mathcal{L}(P,\mathbf{f},\mathbf{g})=\KL(P\Vert M) + \sum_i f_i\big(\sum_j P_{ij}-1\big) + \sum_j g_j\big(\sum_i P_{ij}-1\big)$$

where $\mathbf{f}, \mathbf{g} \in \R^n$ are Lagrange multipliers.

Solving for $\frac{\partial \mathcal{L}}{\partial P_{ij}}=0$ yields $P_{ij}=e^{f_i} M_{ij} e^{g_j}=u_i M_{ij} v_j$ where we introduced $\mathbf{u}=\exp{\mathbf{f}}$ and $\mathbf{v}=\exp{\mathbf{g}}$.

Plugging this into $\frac{\partial \mathcal{L}}{\partial f_i}=0$ yields $u_i = 1 / (\sum_j P_{ij}u_j)$, i.e. $\mathbf{u}=\frac{1}{A\mathbf{v}}$.Likewise, we get $\mathbf{v}=\frac{1}{A^T\mathbf{u}}$.

In conclusion, we know that the (unique!) solution of $(S)$ is of the form

$$P= \text{diag}(\mathbf{u}) M \text{diag}(\mathbf{v})$$

where $\mathbf{u}, \mathbf{v}$ satisfy the above equations.

This leads to the following iterative algorithm, known as Sinkhorn's algorithm[^stability]:

$$
\begin{array}{l}
\hline
\textbf{Algorithm: } \text{Sinkhorn} \\
\hline
\text{1. Initialize } \mathbf{u} \leftarrow \mathbf{1}_n, \mathbf{v} \leftarrow \mathbf{1}_n \\
\text{2. }\textbf{for } k = 1 \text{ to } n_{iter} \textbf{ do}: \\
\quad \quad \mathbf{u} \leftarrow 1 \oslash (M\mathbf{v}) \\
\quad \quad \mathbf{v} \leftarrow 1 \oslash (M^T\mathbf{u}) \\
\text{3. }\textbf{return } P = \text{diag}(\mathbf{u}) M \text{diag}(\mathbf{v}) \\
\hline
\end{array}
$$

Note that $n_{iter}$ is a hyperparameter that controls the accuracy of the projection: the larger it is, the closer $P$ is to the true projection. In practice, $n_{iter}$ is often set between 10 and 20 ($n_{iter}=20$ in mHC).


## III. Triton kernels for Sinkhorn

What's remarkable (and indeed remarked in the age of GPU scientific computing[^cuturi]) about Sinkhorn's algorithm is its simplicity: it only requires matrix-vector multiplications and element-wise operations! This makes it extremely well-suited for GPU implementations, as matrix-vector multiplications can be efficiently parallelized.

However, the looping nature of the algorithm introduces severe memory-boundedness if implemented naively, as each iteration requires reading and writing the entire matrix $M$ as well as the vectors $\mathbf{u}, \mathbf{v}$. This can be mitigated by fusing the algorithm into a single kernel, which reduces memory access overhead and dramatically improves performance.

Indeed, if we look at the memory access pattern of the Sinkhorn algorithm, we have:

| Phase | Read Access (words) | Write Access (words) | Total Access (words) |
| :--- | :--- | :--- | :--- |
| **Per Iteration** | $2n^2 + 4n$ | $2n$ | $2n^2 + 6n$ |
| **Final Write** | $n^2 + 2n$ | $n^2$ | $2n^2 + 2n$ |
| **Total** | $n_{iter}(2n^2 + 4n) + n^2 + 2n$ | $2nn_{iter} + n^2$ | $n_{iter}(2n^2 + 6n) + 2n^2 + 2n$ |

Thus, excluding the final write, memory access simply scales with $n_{iter}$, which is very inefficient.

| Phase | FLOPS |
| :--- | :--- |
| **Per Iteration** | $4n^2 + 2n$ |
| **Final Write** | $2n^2$ |
| **Total** | $n_{iter}(4n^2 + 2n) + 2n^2$ |

Also, if we look at FLOPS, we see that each iteration requires only $4n^2+2n$ FLOPS for $2n^2+6n$ memory accesses. With 4 bytes per word since we're using FP32 precision, this yields an arithmetic intensity of roughly 0.5 FLOPS/byte, which is terrible. Hence, Sinkhorn's algorithm is completely memory-bound.

Sinkhorn's algorithm memory-boundedness can be addressed with kernel fusion. We will now explore increasingly complex Triton kernels implementing Sinkhorn's algorithm[^backward].


### A. Naive PyTorch implementation

Note: we're benchmarking on a NVIDIA RTX 4000 Ada Generation (released Aug 9th, 2023, quite old) [with](https://www.content.shi.com/cms-content/accelerator/media/pdfs/pny/pny-052124-nvidia-rtx-4000-ada.pdf)
- 20GB of VRAM (GDDR6, not HBM :()
- 160-bit memory bus
- 360 GB/s memory bandwidth
- 48 SMs
- 192 Tensor Cores (4th gen, 4 per SM)
- 6144 CUDA cores (Ada architecture, 128 per SM)

Delivering a peak performance of
- 327 TFLOPS for Tensor Cores (FP8, using sparsity)
- 26 TFLOPS for CUDA Cores (FP32)

Let's begin with a simple PyTorch implementation for reference.

```python
def sinkhorn_pytorch(
    log_M: torch.Tensor,  # logits
    n_iter: int,  # increase for better convergence
    epsilon: float  # numerical stability
) -> torch.Tensor:
    """
    PyTorch baseline for comparison with Triton kernels.
    """
    M = torch.exp(log_M)
    B, N, _ = M.shape

    # Initialize scalars
    u = torch.ones(B, N, device=M.device)
    v = torch.ones(B, N, device=M.device)

    # Loop
    for _ in range(n_iter):
        # Row normalization
        # Note: v[:, None, :] broadcasts v to shape (B, 1, N)
        u = 1.0 / ((M * v[:, None, :]).sum(dim=-1) + epsilon)
        
        # Column normalization
        # Note: u[:, :, None] broadcasts u to shape (B, N, 1)
        v = 1.0 / ((M * u[:, :, None]).sum(dim=-2) + epsilon)

    # Final scaled matrix
    return M * u[:, :, None] * v[:, None, :]
```

This version is highly inefficient because $M, \mathbf{u}, \mathbf{v}$ are each read from and written to global memory at each iteration, effectively scaling memory access with $n_{iter}$. We can be smarter!

### B. Basic fused kernel

Currently we have two types of objects doing back and forths between global memory and registers:
1. vector scalers $\mathbf{u}$, $\mathbf{v}$ of size $n$ each
2. matrix $M$ of size $(n,n)$

In my first kernel attempt, I decided to have $\mathbf{u}$ and $\mathbf{v}$ live in registers but keep $M$ in global memory for now. This gave a kernel looking something like the pseudo-code below:

$$
\begin{array}{l}
\hline
\textbf{Algorithm: } \text{Fused Sinkhorn (Global Memory Access)} \\
\hline
\textbf{Input: } M \in \mathbb{R}^{n \times n} \text{ (Global Memory)} \\
\textbf{Output: } P \in \mathbb{R}^{n \times n} \text{ (Global Memory)} \\
\textbf{Scalers: } \mathbf{u}, \mathbf{v}, \mathbf{t} \in \mathbb{R}^n \\
\hline
\text{1. Initialize scalers in registers: } \mathbf{u} \leftarrow \mathbf{1}_n, \mathbf{v} \leftarrow \mathbf{1}_n \\
\text{2. }\textbf{for } k = 1 \text{ to } n_{iter} \textbf{ do}: \\
\quad \quad \text{// Update } \mathbf{u} \text{ (Row reduction)} \\
\quad \quad \textbf{for } i = 1 \text{ to } n \textbf{ do}: \\
\quad \quad \quad \text{// Read row } M_{i,:} \text{ from global memory} \\
\quad \quad \quad u_i \leftarrow 1 / \sum_j(M_{i,j}v_j) \\
\\
\quad \quad \text{// Update } \mathbf{v} \text{ (Column reduction via row streaming)} \\
\quad \quad \mathbf{t} \leftarrow \mathbf{0}_n \\
\quad \quad \textbf{for } i = 1 \text{ to } n \textbf{ do}: \\
\quad \quad \quad \text{// Read row } M_{i,:} \text{ from global memory} \\
\quad \quad \quad \mathbf{t} \leftarrow \mathbf{t} + u_i M_{i,:} \quad \text{// Accumulate scaled rows} \\
\quad \quad \mathbf{v} \leftarrow 1 \oslash (\mathbf{t} + \epsilon) \\
\\
\text{3. }\textbf{Write to Global Memory}: \\
\quad \textbf{for } i = 1 \text{ to } n \textbf{ do}: \\
\quad \quad \text{// Read row } M_{i,:} \text{ from global memory} \\
\quad \quad P_{i,:} \leftarrow u_i M_{i,:} \odot \mathbf{v} \\
\hline
\end{array}
$$


### C. Loading $M$ in registers

We've started reducing I/O bandwidth by keeping $\mathbf{u}$ and $\mathbf{v}$ in registers, but we can still do much better!
First of all, we need a bit of context: we're using Sinkhorn's algorithm to project $\Hres_l$ on Birkhoff's polytope. But $\Hres$ is a small matrix since the scaling $n$ of the residual stream is small. We'll use $n=4$ like in mHC. This means that $M$ is effectively a $4\times 4$ matrix, which can easily fit in registers! Thus, we can update the kernel to have $M$ live in the registers, which saves us some more I/O bandwidth!

This gives us exactly the same algorithm as above, except that $M$ is loaded in the registers at the beginning of the kernel, so we don't need to read it from memory at every row / column normalization step!

### D. Block Tiling

The previous solution may seem optimal, but we can in fact do much better.

I realized this by experimenting on my own and seeing that the unoptimized PyTorch solution was beating my kernel for very large batch sizes (2048 and above). The reason is simple: PyTorch is smart enough to pack small matrices together whenever it can, helping the GPU better saturate its cores. Indeed, given a batch of $B$ matrices of size $(n,n)$ to process, it will try to concatenate them into an array of

$$N_{block} = \lceil B / \text{BLOCK\_SIZE} \rceil$$

tensors of shape $(\text{BLOCK\_SIZE},n,n)$ and feed each block to a different GPU core. This is much more efficient than using one block per matrix, which is what we're currently doing.

Let's now set `TARGET_BLOCK_SIZE=256` (~ sweep spot for block size) and concatenate our $B$ matrices into $N_{block}$ blocks of size $(B_{block}, n, n)$ where $B_{block} = \lceil B / N_{block} \rceil$. We can then use one block per block of matrices, which will be much more efficient for large $B$! This technique is known as block tiling and it allows us to better saturate the GPU's cores.

### E. Coalesced memory access

We're now getting to a really good kernel. This last one combines some last tricks to make it truly SOTA:
- uses coalesced memory access to read rows of $M$ in a single instruction
- log on the fly to avoid the initial exp
- initializes scalers without the +1.0 but directly as ones

---

**References**:

[^rmt]: Mak, B., & Flanigan, J. (2025). *Residual Matrix Transformers: Scaling the Size of the Residual Stream.* [[arXiv](https://arxiv.org/abs/2506.22696)]
[^hc]: D. Zhu, H. Huang, Z. Huang, Y. Zeng, Y. Mao, B. Wu, Q. Min, and X. Zhou. (2024). *Hyper-connections.* [[arXiv](https://arxiv.org/abs/2409.19606)]
[^mhc]: Xie, Z., Wei, Y., Cao, H., Zhao, C., Deng, C., Li, J., Dai, D., Gao, H., Chang, J., Yu, K., Zhao, L., Zhou, S., Xu, Z., Zhang, Z., Zeng, W., Hu, S., Wang, Y., Yuan, J., Wang, L., & Liang, W. (2025). *mHC: Manifold-Constrained Hyper-Connections.* [[arXiv](https://arxiv.org/abs/2512.24880)]
[^resnet]: He et al. (2015). *Deep Residual Learning for Image Recognition* [[arXiv](https://arxiv.org/abs/1512.03385)]
[^divergence]: The (generalized) KL divergence isn't actually a metric but a (Bregman) divergence. We omit this detail for the sake of clarity.
[^stability]: Sinkhorn's algorithm is numerically unstable for very small entries of $M$. In practice, one usually adds a small constant $\epsilon$ to $M$ to avoid division by zero. One can also work in log-space to improve numerical stability. Finally, since we want $P$ to have strictly positive entries, we must exponentiate $M$ if it has negative entries; some implementations also scale $M$ by a temperature parameter before exponentiating to control the sharpness of $P$.
[^cuturi]: Cuturi, M. (2013). *Sinkhorn Distances: Lightspeed Computation of Optimal Transport.* [[arXiv](https://arxiv.org/abs/1306.0895)][^backward]: Although I didn't cover it in this post, efficiently implementing the backward pass of Sinkhorn's algorithm is non-trivial as it not only requires fused kernels, but also activation recomputation to avoid storing all intermediate $\mathbf{u}, \mathbf{v}$ vectors. I may cover this in a future post though!
[^backward]: Although I didn't cover it in this post, efficiently implementing the backward pass of Sinkhorn's algorithm is non-trivial as it not only requires fused kernels, but also activation recomputation to avoid storing all intermediate $\mathbf{u}, \mathbf{v}$ vectors. I may cover this in a future post though!