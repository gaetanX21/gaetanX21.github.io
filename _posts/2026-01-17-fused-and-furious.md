---
layout: post
title: "Fused & Furious: Sinkhorn Triton Kernels"
date: 2026-01-17
description: "TL;DR: DeepSeek's recent mHC paper relies on Sinkhorn's algorithm to project matrices onto Birkhoff's polytope. The looping nature of the algorithm introduces severe memory-boundedness, which can be mitigated by fusing the algorithm into a single kernel. We implement increasingly fast versions of the algorithm in Triton."
tags: llm, pretraining, research
thumbnail: assets/img/posts/fused_and_furious/cover.jpg
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
\newcommand{\one}{\mathbf{1}_n}
\newcommand{\T}{\text{Tr}}
\newcommand{\KL}{\text{KL}}
$$


* Table of Contents
{:toc}

In their last paper, charmingly titled *Manifold-Constrained Hyper-Connections*[^mhc], DeepSeek improved upon preexisting research on **residual stream scaling** with two major contributions:
1. an approach that's more **stable** than previous papers, with demonstrated performance at scale
2. an **efficient** infrastructure design that minimizes the overhead of hyper-connections compared to vanilla transformers

While I encourage you to read the full paper, my goal in this post is to delve deeper into the second part. More precisely, I want to focus on the Sinkhorn projection step of their architecture: why we need custom fused GPU kernels to implement it, and how to do it.

I'll first discuss the theory around mHC and Sinkhorn, then I'll showcase increasingly fast (and tricky) Triton kernels for the Sinkhorn algorithm.

*The code for this project can be found [here](https://github.com/gaetanX21/sinkhorn-triton).*


---

<div class="row justify-content-center" id="fig-1">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/fused_and_furious/speedup.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <b>Figure 1.</b> For batch sizes below 1k, both the naive PyTorch function and the optimized Triton kernel operate in the latency-bound regime, hence the speedup is roughly constant (and scales with <code>n_iter</code>). Past this threshold, we enter the memory-bound regime where the Triton kernel's I/O awareness shines.
</div>

## I. Scaling the residual stream

A current trend in macro design of transformers is to scale the **size** of the residual stream. That is, instead of being a vector $\x\in\R^{\D}$, the per-token residual stream becomes a matrix $X\in\R^{n\times\D}$ where $n$ is a new scaling dimension (often a small value e.g. $n=4$). The motivation behind this architectural shift is that in vanilla transformers, model parameters and FLOPs scale *quadratically* with $\D$ whereas the residual stream scales *linearly*. Hence, as we keep increasing model size, the residual stream may become a **bottleneck**.

### A. Residual Matrix Transformers

My [previous post](/blog/residual-matrix-transformer/) discussed the **Residual Matrix Transformer** (RMT) architecture, which introduces a matrix representation of the per-token residual stream inspired by outer-product memories. The key idea is that using a matrix instead of a vector increases the effective storage capacity of the residual stream. The implementation of RMT is fairly simple, as the only difference compared to the classic transformer architecture is how we *read from* and *write to* the residual stream. Crucially, the micro design remains untouched (i.e. Attention and Feed-Forward blocks do not change), which implies that scaling the residual stream incurs negligible computational overhead, though it *does* cause memory access overhead if implemented naively (i.e. without fused kernels) as we will see.

In a nutshell, RMT replaces the classic residual connection

$$\x_{l+1} = \x_l + \F_l(\x_l)$$

with

$$X_{l+1} = X_l + {\Hpost}^T \F_l(\Hpre X_l)$$

where $\x_l\in\R^{\D}$ (resp. $X_l\in\R^{n\times\D}$) is the $l$-th layer vector (resp. matrix) residual stream, and $\F_l$ is the $l$-th transformer block.

$\Hpre_l, \Hpost_l \in \R^{1\times n}$ are *learned* vectors which correspond respectively to `READ` and `WRITE` operations on the residual stream memory store. 


### B. Hyper-Connections

Hyper-Connections[^hc] (HC) extend the idea of RMT by enabling *communication* between the $n$ channels of the residual stream.

The motivation is to **increase the topological complexity of the residual mapping without increasing the computational complexity**. The main improvement over RMT is the introduction of $\Hres_l\in\Rnn$ which acts as a "stream-mixing matrix" allowing the $n$ channels to **exchange information** instead of simply evolving independently. Also, $\Hpre_l, \Hpost_l, \Hres_l$ are now both *static* (i.e. learned parameters) and *dynamic* (i.e. dependent on input $X_l$) to further increase the topological complexity of the connection network (see original paper for more details). Note that we will omit the dependence on $X_l$ to keep notations lightweight (e.g. we'll write $\Hres_l$ instead of $\Hres_l(X_l)$)

Thus, the residual connection in HC is even richer than in RMT thanks to $\Hres_l$:

$$X_{l+1} = \Hres_l X_l + {\Hpost}^T \F_l(\Hpre X_l)$$


### C. Manifold-Constrained Hyper-Connections

DeepSeek's Manifold-Constrained Hyper-Connections (mHC) address one critical flaw in HC: instability as the network depth $L$ scales.

Indeed, HC abandons the identity mapping of residual connections and replaces it with $\Hres_l$. Yet, the identity mapping is crucial to the training stability of deep neural network architectures[^resnet].

It's easy to see why replacing $\mathbb{I}_n$ with $\Hres_l$ is problematic: if we recursively extend the residual connection across multiple layers, we have:

1. $$\x_{l+m} = \x_l + \sum_{k=0}^{m-1}\F_{k+l}(\x_{l+k})$$ for the standard residual connection
2. $$X_{l+m} = X_l + \sum_{k=0}^{m-1}{\Hpost_{l+k}}^T \F_{l+k}(\Hpre_{l+k} X_{l+k})$$ for RMT
3. $$X_{l+m} = \big(\prod_{k=0}^{m-1}\Hres_{l+k}\big) X_l + \sum_{k=0}^{m-1}{\Hpost_{l+k}}^T \F_{l+k}(\Hpre_{l+k} X_{l+k})$$ for HC

The matrix product $\Pi=\prod_{k=0}^{m-1}\Hres_{l+k}$ has no reason to behave nicely i.e. keep a spectral norm close to 1. In fact, mHC shows that instead of preserving the signal strength, $\Pi$ tends to amplify or attenuate it as depth increases, resulting in signal rescaling across **several orders of magnitudes**, which is problematic for both the forward pass (exploding/vanishing activations) and the backward pass (exploding/vanishing gradients). The propagation of instability as depth increases is illustrated in [Figure 2](#fig-2). 

<div class="row justify-content-center" id="fig-2">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/fused_and_furious/mhc_forward_backward_gain.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <b>Figure 2.</b> This figure illustrates the propagation dynamics of (a) the single-layer mapping $\Hres_l$ and (b) the composite mapping $\prod_{i=1}^{L-1}\Hres_{L-i}$ within a 27B model. The layer index $l$ (x-axis) unrolls each standard Transformer block into two independent layers (Attention and FFN). The <i>Amax Gain Magnitude</i> (y-axis) is calculated as the maximum absolute row sum (for the forward signal) and column sum (for the backward gradient), averaged over all tokens in a selected sequence. Figure taken from DeepSeek mHC paper.
</div>

<div class="row justify-content-center" id="fig-3">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/fused_and_furious/mhc_teaser.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <b>Figure 3.</b> Illustrations of residual connection paradigms. This figure compares the structural design of (a) standard Residual Connection, (b) Hyper-Connections (HC), and (c) mHC. Unlike the unconstrained HC, mHC focuses on optimizing the residual connection space by projecting the matrices onto a constrained manifold to ensure stability. Figure taken from DeepSeek mHC paper.
</div>

DeepSeek's solution to this instability problem is to constrain $\Hres_l$ to be a **bistochastic matrix** i.e. a square matrix with non-negative entries where each row and column sums to 1.

Bistochastic matrices have three nice properties that make them ideal for stabilizing the residual stream:
1. they have a spectral norm of 1, meaning that they preserve signal strength across layers
2. they are closed under matrix multiplication, meaning that the product of multiple bistochastic matrices is also bistochastic
3. they can be given a probabilistic / optimal transport interpretation as *soft permutations*, meaning that they allow each channel to attend to and exchange with every other channel, without collapsing to a single channel

I'm a bit surprised that the authors of HC did not think of the bistochastic constraint in the first place, as it seems like an intuitive solution to the instability problem. Perhaps they lacked an efficient implementation of the projection step, which is precisely the focus of this post!

Indeed, to enforce the bistochastic constraint, we must first compute $\Hres_l$ as in HC, and then project it onto the manifold of bistochastic matrices $\B$, also known as Birkhoff's polytope. The next section goes into the details of this projection, which uses Sinkhorn's algorithm.


## II. Sinkhorn's Algorithm

Quite plainly, the goal of Sinkhorn's algorithm is to project a given square matrix $M\in\Rnn$ onto the Birkhoff polytope

$$\B=\lbrace P\in\Rnn \mid P\one=\one, \one P=\one, P\geq 0\rbrace$$.

Note that $\B$ is convex as we will use this property later. In fact it is the convex hull generated by the $n!$ permutation matrices in $\Rnn$, which are the extreme points of $\B$.[^mhc-lite]

Let's now present Sinkhorn's algorithm and derive its formulation intuitively.

### A. Projecting under the generalized KL divergence

We've been talking about projections, but to define a projection, one needs a metric. In other words, we need a sensible metric $d: \Rnn \times \Rnn \to \Rnn$ and then we can try to find
$$\text{Proj}_d^{\B}(M) = \arg\min_{P \in \B} d(P,M)$$
for a given matrix $M \in \Rnn$.

One classic metric on $\Rnn$ is the one *induced* by the Frobenius norm ${\Vert M \Vert} _ F = \sqrt{\T (MM^T)} = \sqrt{ \sum_{1 \leq i,j \leq n} M_{ij}^2}$.

However, using this metric would mean we're doing a Euclidean ($L^2$) projection, which intuitively doesn't feel right here given the probabilistic / optimal transport interpretation of bistochastic matrices. It would also require solving a linear programming problem in $O(n^3)$, which isn't ideal.

Instead, one much more sensible metric[^divergence] is the KL divergence. Since we're dealing with bistochastic matrices, we use the *generalized* KL divergence:

$$\KL(P\Vert M) = \sum_{1 \leq i,j \leq n} \left( P_{ij} \log\left(\frac{P_{ij}}{M_{ij}}\right) - P_{ij} + M_{ij} \right)$$

The extra terms serve a purpose: $P_{ij}$ ensures $\KL(P\Vert M)$ is minimized iff $P=M$, and $M_{ij}$ guarantees non-negativity.

Thus, the problem we want to solve is:

$$\min_{P \in \B} \KL(P\Vert M) \quad (S)$$

**Crucially, $P \mapsto \KL(P\Vert M)$ is strictly convex and $\B$ is convex, meaning that $(S)$ has a unique solution!**

<div class="row justify-content-center" id="fig-4">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/fused_and_furious/proj_kl.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <b>Figure 4.</b> Two ways to project a matrix $M$ onto Birkhoff's polytope: using the L2 norm, which amounts to finding a Euclidian geodesic and yields linear programming problem, or using the generalized KL divergence, which means finding a geodesic for the generalized KL divergence and is solved with Sinkhorn's algorithm.
</div>


### B. Deriving the algorithm

Now that we've defined the projection as solving $(S)$, let's derive Sinkhorn's algorithm. We'll tackle this problem just like any optimization problem, using Lagrange's multipliers.

First, we introduce the Lagrangian:

$$\mathcal{L}(P,\mathbf{f},\mathbf{g})=\KL(P\Vert M) + \sum_i f_i\big(\sum_j P_{ij}-1\big) + \sum_j g_j\big(\sum_i P_{ij}-1\big)$$

where $\mathbf{f}, \mathbf{g} \in \R^n$ are Lagrange multipliers.


Then, solving for $\frac{\partial \mathcal{L}}{\partial P_{ij}}=0$ yields $P_{ij}=e^{-f_i} M_{ij} e^{-g_j}=u_i M_{ij} v_j$ where we introduced $\mathbf{u}=\exp{(-\mathbf{f})}$ and $\mathbf{v}=\exp{(-\mathbf{g})}$.

Next, plugging this into $\frac{\partial \mathcal{L}}{\partial f_i}=0$ yields $u_i = 1 / (\sum_j P_{ij}v_j)$, i.e. $\mathbf{u}=1 \oslash (M\mathbf{v})$.

Likewise, we get $\mathbf{v}=1 \oslash (M^T\mathbf{u})$.

Finally, we know that the (unique!) solution of $(S)$ is of the form

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

What's remarkable (and indeed remarked in the age of GPU scientific computing[^cuturi]) about Sinkhorn's algorithm is its simplicity: it only requires matrix-vector multiplications and element-wise operations! This makes it extremely well-suited for GPU implementations, as these operations can be efficiently parallelized between the (tens of) thousands of cores of modern GPUs.

However, the looping nature of the algorithm introduces **severe memory-boundedness** if implemented naively, as each iteration requires reading and writing the entire matrix $M$ as well as the vectors $\mathbf{u}, \mathbf{v}$. This can be mitigated by **fusing** the algorithm into a single kernel, which helps dramatically reduce memory access overhead and improve performance.

Indeed, if we look at the memory access pattern of the Sinkhorn algorithm, we have:

| **Phase** | **Read Access (words)** | **Write Access (words)** | **Total Access (words)** |
| :--- | :--- | :--- | :--- |
| **Per Iteration** | $2n^2 + 4n$ | $2n$ | $2n^2 + 6n$ |
| **Final Write** | $n^2 + 2n$ | $n^2$ | $2n^2 + 2n$ |
| **Total** | $n_{iter}(2n^2 + 4n) + n^2 + 2n$ | $2n_{iter}n + n^2$ | $n_{iter}(2n^2 + 6n) + 2n^2 + 2n$ |

Thus, excluding the final write, memory access **scales with $n_{iter}$**, which is very inefficient.

| **Phase** | **FLOPS** |
| :--- | :--- |
| **Per Iteration** | $4n^2 + 2n$ |
| **Final Write** | $2n^2$ |
| **Total** | $n_{iter}(4n^2 + 2n) + 2n^2$ |

Also, if we look at FLOPS, we see that each iteration requires only $4n^2+2n$ FLOPS for $2n^2+6n$ memory accesses. With 4 bytes per word since we're using FP32 precision, this yields an arithmetic intensity of roughly 0.5 FLOPS/byte, which is terrible. Hence, **Sinkhorn's algorithm is completely memory-bound**.

Sinkhorn's algorithm memory-boundedness can be addressed with kernel fusion. We will now explore increasingly complex implementations based on Triton kernels[^backward].

### A. Hardware details

We're benchmarking on a **NVIDIA RTX 4000 Ada Generation** (released Aug 9th, 2023) [with](https://www.content.shi.com/cms-content/accelerator/media/pdfs/pny/pny-052124-nvidia-rtx-4000-ada.pdf):
- 20GB of VRAM (GDDR6, not HBM)
- 360 GB/s memory bandwidth and 160-bit memory bus
- 48 Streaming Multiprocessors (SMs)
- 192 Tensor Cores (4th gen, 4 per SM)
- 6144 CUDA cores (Ada architecture, 128 per SM)

Delivering a peak performance of:
- 327 TFLOPS for Tensor Cores (Achtung: that's in FP8 and "with sparsity"[^sparsity])
- 26 TFLOPS for CUDA Cores (FP32)

This isn't some fancy dual-die GB300 or whatever, but it's still enough to do some serious benchmarking. Let's get to it!


### B. Mini-refresher on memory hierarchy

I've said that Sinkhorn's algorithm is **memory-bound**. This may seem unclear if you're not familiar with "memory hierarchy". I'll recap it *very* briefly here, focusing on GPUs.

Basically, GPUs have a **hierarchical memory system** with different types of memory at different levels of the hierarchy. While the full theory is quite complicated, a simple binary model of GPU memory is sufficient to grasp the main challenges. Thus, you can think of GPU memory as being split in two components:
1. **SRAM** where "S" stands for "Static", in reference to the *static* nature of the stored information (uses a 6-transistor cell).
2. **VRAM** where "V" stands for "Video" since GPUs were mostly for video games & animation historically. It's also called DRAM because it *is* DRAM i.e. it uses *dynamic* memory cells (based on capacitors which must be refreshed frequently). Whenever we talk about HBM, we're talking about this memory[^hbm].

The key is that SRAM is fast but small, whereas VRAM is slow but large. *One* goal of GPU kernels is to **maximize the amount of computation done in SRAM** (i.e. inside the registers) and **minimize the amount of data exchanged with VRAM**. That's because moving data between VRAM and SRAM is extremely slow (orders of magnitude) compared to computation inside the registers.

Below is a table summarizing the key differences between SRAM and VRAM. [Figure 5](#fig-5) gives a more detailed view of the memory hierarchy of the NVIDIA H100 GPU.

| Feature | SRAM | VRAM ("HBM") |
| :--- | :--- | :--- |
| **Location** | On-chip | Off-chip |
| **Capacity** | Small capacity (10s-100s MBs) | Large capacity (10s-100s GBs) |
| **Latency** | Fast as hell | Not so fast |
| **Bandwidth** | Massive | Large |
| **Components** | Registers + Shared memory + L1/L2 cache | Global memory + "local memory" |

<br>

<div class="row justify-content-center" id="fig-5">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/fused_and_furious/mem_hierarchy.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <b>Figure 5.</b> Memory hierarchy of the H100 (SXM5) GPU. Taken from <a href="https://www.aleksagordic.com/blog/matmul">this</a> outstanding blog post.
</div>

*Note: despite its name, "local memory" actually lives in the GPU's VRAM. This memory is accessed when registers are full but we need more storage capacity. Register spill over into local memory is something we really want to avoid as it completely kills performance due to the slow access time of VRAM.*

**The key thing to remember is that by default, data lives in VRAM until you want to do stuff with it and then it gets loaded into SRAM for computation, then the result is written back to VRAM.**

This means that if you implement things naively, each operation costs you a back and forth between VRAM and SRAM. For example, imagine you have 2 matrices $A$ and $B$ and want to compute their product $C$. Three things happen sequentially:
1. $A$ and $B$ are loaded into SRAM. (VRAM → SRAM)
2. The computation is done in SRAM.
3. The result $C$ is written back to VRAM. (SRAM → VRAM)

This seems fair. But now imagine you have a matrix $A$ and want to do 2 things on it: first square it, then exponentiate it. If you do it naively, you'll have to do 4 back and forths between VRAM and SRAM:
1. $A$ is loaded into SRAM. (VRAM → SRAM)
2. $A$ is squared in SRAM.
3. The result $A^2$ is written back to VRAM. (SRAM → VRAM)
4. $A^2$ is loaded into SRAM. (VRAM → SRAM)
5. $A^2$ is exponentiated in SRAM.
6. The result $\exp(A^2)$ is written back to VRAM. (SRAM → VRAM)

This is clearly suboptimal. What if we could do both operations in one go? That's where **fused kernels** come in. A fused kernel is a kernel that does multiple sequential operations in the registers, hence minimizing the number of back and forths between VRAM and SRAM. This is exactly what we're going to do in the next section.

Also, whenever we refer to *registers*, it's just a more granular way to refer to SRAM. The SRAM is more than registers (it also includes shared memory & L1/L2 cache) but for the purpose of this blog post, you can equate these two concepts.

There's a lot more to be said about memory hierarchy but we'll stop here for now. If you want to learn more, I recommend [this](https://www.aleksagordic.com/blog/matmul) deep dive.

---

### C. Implementations

Now that the hardware is out of the way, let's dive into the implementations. We'll start with a naive PyTorch implementation, then move on to the Triton kernels!

#### 1. Naive PyTorch implementation

We begin with a simple PyTorch implementation for reference.

```python
import torch

torch.set_float32_matmul_precision("high")  # we want to leverage Tensor Cores!

def sinkhorn_pytorch(
    log_M: torch.Tensor,  # logits
    n_iter: int,  # increase for better convergence
    epsilon: float,  # numerical stability
) -> torch.Tensor:
    """PyTorch baseline for comparison with Triton kernels."""
    M = torch.exp(log_M)
    M_T = M.transpose(-1, -2)  # free transpose (view trick)
    B, N, _ = M.shape

    # initialize scalers
    u = torch.ones(B, N, 1, device=M.device)
    v = torch.ones(B, N, 1, device=M.device)

    # loop
    for _ in range(n_iter):
        u = 1.0 / (M @ v + epsilon)  # row normalization
        v = 1.0 / (M_T @ u + epsilon)  # column normalization

    # final scaled matrix
    return u * M * v.transpose(-1, -2)
```

This version is **highly inefficient** because $M, \mathbf{u}, \mathbf{v}$ are each read from and written to global memory at each iteration, effectively scaling memory access with $n_{iter}$. We can be smarter!


In addition, we define a slightly more optimized baseline, which technically *is* a fused kernel but uses poor auto-fusion heuristics provided by PyTorch's `torch.compile`. We also use `torch.inference_mode` to drop the computational graph and tensor version tracking since we're only interested in the forward pass. Thus, this second baseline has no unfair disadvantage compared to the Triton kernels coming next.

Formally, we construct this second baseline as a wrapper of the first one:
```python
sinkhorn_pytorch_compiled = torch.inference_mode()(torch.compile(sinkhorn_pytorch))
```

This second version is still inefficient, but it's a good sanity check to ensure that our Triton kernels are actually faster.


#### 2. Basic fused kernel

Currently we have two types of objects doing back and forths between global memory and registers:
1. vector scalers $\mathbf{u}$, $\mathbf{v}$ of size $n$
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
\quad \quad \text{// Update } \mathbf{u} \text{ (Row normalization)} \\
\quad \quad \textbf{for } i = 1 \text{ to } n \textbf{ do}: \\
\quad \quad \quad \text{// Read row } M_{i,:} \text{ from global memory} \\
\quad \quad \quad u_i \leftarrow 1 / \sum_j(M_{i,j}v_j) \\
\\
\quad \quad \text{// Update } \mathbf{v} \text{ (Column normalization)} \\
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

Note that we use an accumulator $\mathbf{t}$ for the column normalization step. This is to avoid reading $M$ column-by-column, which is inefficient due to strided memory access[^strided] (whereas reading $M$ row-by-row offers efficient coalesced memory access).

This first kernel is a good start as we've reduced the I/O bandwidth caused by $\mathbf{u}$ and $\mathbf{v}$'s back-and-forths in global memory, but $M$ is still read twice per iteration, meaning that memory access *still* scales with $n_{iter}$. We can do much better!


#### 3. Loading $M$ in registers

First of all, we need a bit of context: remember that we're using Sinkhorn's algorithm to project $\Hres_l$ onto Birkhoff's polytope. But **$\Hres_l$ is a tiny matrix** since the scaling factor $n$ of the residual stream is a small integer value. For the sake of simplicity, we'll stick with $n=4$ like in mHC. This means that $M$ is effectively a $4\times 4$ matrix, which can easily fit in registers! Thus, we can update the kernel to have $M$ live in the registers, which will save us a lot more I/O bandwidth!

This gives us exactly the same algorithm as above, except that $M$ is loaded in the registers at the beginning of the kernel, so we don't need to read it from memory at every row / column normalization step! One other difference: the `exp` operation on $M$ is now done on-the-fly inside the registers, which saves one back-and-forth between global memory and registers.

#### 4. Block Packing

The previous solution may seem optimal, but we can in fact still do *much* better!

I realized this by experimenting on my own and observing the unoptimized PyTorch solution beating my kernel for batch sizes of 2048 and above. The reason is simple: PyTorch is smart enough to pack small matrices together whenever it can, helping the GPU better saturate its cores.

Indeed, given a batch of $B$ matrices of size $(n,n)$ to process, instead of using one block per matrix, PyTorch will try to *concatenate* them into an array of

$$N_{block} = \bigg\lceil \frac{B}{\text{BLOCK\_SIZE}} \bigg\rceil$$

tensors of shape $(\text{BLOCK\_SIZE},n,n)$, and then process a *full tensor* per block. This technique is known as block tiling, it's useful when the data objects you're processing are too small to saturate the GPU's cores. See [Figure 5](#fig-5) for an illustration.

<div class="row justify-content-center" id="fig-6">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/fused_and_furious/block_packing.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <b>Figure 6.</b> Block packing: instead of using one block per matrix, we pack $B$ matrices into $N_{block}$ tensors of shape $(\text{BLOCK_SIZE}, n, n)$ and process a full tensor per block. Consequently, the threads dedicated to each block are better utilized (higher occupancy). In this example, $B=8, N_{block}=2, \text{BLOCK_SIZE}=4$.
</div>

This is much more efficient than using one block per matrix, which is what the two previous kernels are doing. The reason for that is that **a block has significant resources**: 4 warps by default, each with 32 threads, meaning a total of 128 threads. Thus, assigning a single $4\times 4$ matrix per block means we're effectively dedicating 8 threads per matrix element, which is *way* too much and leads to underutilization i.e. idle threads. This in turn greatly slows down the kernel as it hinders parallelism (since the idle threads cannot be used to process other matrices).

Choosing the right $\text{BLOCK\_SIZE}$ i.e. how many matrices to pack in a single block is a bit of an art. One easy method is to use on `triton.autotune` to benchmark different values. I did so for $N=4$ and found that $\text{BLOCK_SIZE}=64$ yielded the best results.[^num-warps]

Our last kernel thus reuses the same layout as the previous one, except that we now process a full tensor of shape $(\text{BLOCK_SIZE},N,N)$ per block. For $N=4$, we thus end up using tensors of dimension $(64, 4, 4)$.


### D. Benchmark

We have presented in total **5 implementations** of Sinkhorn's algorithm:
1. naive PyTorch (`sinkhorn_pytorch`)
2. naive Pytorch but auto-fused and with inference mode (`sinkhorn_pytorch_compiled`)
3. Triton kernel with scalers $\mathbf{u}, \mathbf{v}$ in registers but $M$ in global memory (`sinkhorn_A_in_global_memory`)
4. Triton kernel with $\mathbf{u}, \mathbf{v}, M$ in registers (`sinkhorn_A_in_registers`)
5. Triton kernel with $\mathbf{u}, \mathbf{v}, M$ in registers and block packing (`sinkhorn_A_in_registers_block_packing`)

*Note: `A` refers to the input matrix $M$, as I happened to have used different naming conventions in the code.*

We will now benchmark these implementations on our NVIDIA RTX 4000 Ada Generation.

We'll stick to $N=4$ to have simple 2D plots instead of heatmaps / 3D plots.

As for the batch size $B$, we will sweep from $1$ to $2^{24}\sim 16M$. The reason for sweeping to such large batch sizes is twofold:
1. because $\Hres_l$ is defined at the **token-level**, it means that for a given network layer $l$, we need as many Sinkhorn projections as we have tokens in our batch. Thus, the relevant batch size $B$ to benchmark on is the microbatch size $mbs$. Today's pretraining pipelines commonly use microbatch sizes in the range $mbs\in[4096, 16384]$[^mbs], which justifies benchmarking at least up to $B=16k$.
2. it allows us to escape the uninteresting *latency-bound* regime and get to the much cooler *memory-bound* regime (that's why we push $B$ much further than the $16k$ required for pretraining)

### E. Figures


[Figure 7](#fig-7) is the vanilla metric.


#### 1. Vanilla metric

It shows the speedup of the most optimized Triton kernel (`sinkhorn_A_in_registers_block_packing`) over the compiled PyTorch implementation (`sinkhorn_pytorch_compiled`). We can see that for batch sizes below 1k, both implementations operate in the latency-bound regime, hence the speedup is roughly constant (and scales with $n_{iter}$). Past this threshold, we enter the memory-bound regime where the Triton kernel's I/O awareness shines!

In any case, the speedup ranges from 20x to **139x**, which is pretty cool!

<div class="row justify-content-center" id="fig-7">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/fused_and_furious/speedup.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <b>Figure 7.</b> For batch sizes below 1k, both the naive PyTorch function and the optimized Triton kernel operate in the latency-bound regime, hence the speedup is roughly constant (and scales with <code>n_iter</code>). Past this threshold, we enter the memory-bound regime where the Triton kernel's I/O awareness shines.
</div>


#### 2. Comparing implementations

The next three figures compare the implementations across three performance metrics:
1. **Execution time** (lower is better) on [Figure 8](#fig-8)
2. **Memory bandwidth** (higher is better) on [Figure 9](#fig-9)
3. **Compute throughput** (higher is better) on [Figure 10](#fig-10)

Note that the execution time uses the **median** (instead of the mean) to avoid being skewed by outliers, since kernel execution times tend to exhibit positive skew.

We also compute the 99% confidence interval (`q01` to `q99`) to ensure the **statistical significance** of the results.

Also, the memory bandwidth and compute throughput are computed as follows:
- **Memory bandwidth** = $\frac{\text{total bytes read from/written to global memory}}{\text{median execution time}}$
- **Compute throughput** = $\frac{\text{total FLOPS}}{\text{median execution time}}$


Across all three metrics, we observe an order relation:
`sinkhorn_pytorch` < `sinkhorn_pytorch_compiled` < `sinkhorn_A_in_global_memory` < `sinkhorn_A_in_registers` < `sinkhorn_A_in_registers_block_packing`, as expected!

Also, we consistently witness two regimes:
1. $B\ll 1k$ is the **latency-bound regime**: the amount of data is not sufficient to keep all the GPU's SMs busy, and as such execution time is roughly constant. In this regime, the GPU's throughput scales linearly with batch size as more SMs are utilized, meaning we can effectively scale memory bandwidth and compute throughput "for free" thanks to increased parallelism.
2. $B\gg 1k$ is the **memory-bound regime**: there's now enough data to use all the GPU's SMs, and as such execution time now scales linearly with batch size i.e. no more "free" performance from increased parallelism. That's why global throughput mostly plateaus beyond $B=16k$.

<div class="row justify-content-center" id="fig-8">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/fused_and_furious/timing.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <b>Figure 8.</b> Comparison of median execution time for various implementations of Sinkhorn's algorithm. The shading represents the 99% confidence interval (<code>q01</code> to <code>q99</code>). For batch sizes below 1k, we operate in the latency-bound regime (i.e. some of the GPUs SMs are idle) and as such execution time is roughly constant. Past this threshold, we enter the memory-bound regime where execution time scales linearly with batch size.
</div>

<div class="row justify-content-center" id="fig-9">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/fused_and_furious/memory_bandwidth.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <b>Figure 9.</b> Memory bandwidth increases linearly with batch size in the latency-bound regime as we distribute the work to more SMs, then saturates in the memory-bound regime. Note the peak bandwidth of 238 GB/s, satisfyingly close to the hardware limit of 360 GB/s for this GPU. Note that the memory bandwidth decreases as <code>n_iter</code> increases, which is expected as we spend more time computing inside the registers and less time exchanging data between global memory and registers. For <code>n_iter=1</code> we get 312 GB/s peak I/O bandidth i.e. 87% of the hardware limit.
</div>

<div class="row justify-content-center" id="fig-10">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/fused_and_furious/compute_throughput.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <b>Figure 10.</b> Like memory bandwidth, compute throughput increases linearly with batch size in the latency-bound regime, then saturates in the memory-bound regime. Note the peak throughput of 2.7 TFLOPS, which is very far from the hardware limit of 26 TFLOPS for this GPU. This isn't surprising as Sinkhorn's algorithm is memory-bound.
</div>

---

**References**:

[^rmt]: Mak, B., & Flanigan, J. (2025). *Residual Matrix Transformers: Scaling the Size of the Residual Stream.* [[arXiv](https://arxiv.org/abs/2506.22696)]
[^hc]: D. Zhu, H. Huang, Z. Huang, Y. Zeng, Y. Mao, B. Wu, Q. Min, and X. Zhou. (2024). *Hyper-connections.* [[arXiv](https://arxiv.org/abs/2409.19606)]
[^mhc]: Xie, Z., Wei, Y., Cao, H., Zhao, C., Deng, C., Li, J., Dai, D., Gao, H., Chang, J., Yu, K., Zhao, L., Zhou, S., Xu, Z., Zhang, Z., Zeng, W., Hu, S., Wang, Y., Yuan, J., Wang, L., & Liang, W. (2025). *mHC: Manifold-Constrained Hyper-Connections.* [[arXiv](https://arxiv.org/abs/2512.24880)]
[^resnet]: He et al. (2015). *Deep Residual Learning for Image Recognition* [[arXiv](https://arxiv.org/abs/1512.03385)]
[^mhc-lite]: This result is known as the Birkhoff-von Neumann theorem. We can use it to parametrize $M\in\B$ as a convex combination of $n!$ permutation matrices. Note that such a parametrization allows to completely bypass Sinkhorn's algorithm while also obtaining a perfect projection. This is the approach taken by [mHC-lite](https://arxiv.org/abs/2601.05732), a paper written a few days after mHC.
[^strided]: In practice, this doesn't make a huge difference for small matrices (e.g. $N=4$) as the stride is tiny, but it can be significant for larger ones (e.g. $N=256$).
[^cuturi]: Cuturi, M. (2013). *Sinkhorn Distances: Lightspeed Computation of Optimal Transport.* [[arXiv](https://arxiv.org/abs/1306.0895)]
[^divergence]: The (generalized) KL divergence isn't actually a metric but a (Bregman) divergence. We omit this detail for the sake of clarity.
[^stability]: Sinkhorn's algorithm is numerically unstable for very small entries of $M$. In practice, one usually adds a small constant $\epsilon$ to $M$ to avoid division by zero. One can also work in log-space to improve numerical stability. Finally, since we want $P$ to have strictly positive entries, we must exponentiate $M$ if it has negative entries; some implementations also scale $M$ by a temperature parameter before exponentiating to control the sharpness of $P$.
[^backward]: Although I didn't cover it in this post, efficiently implementing the backward pass of Sinkhorn's algorithm is non-trivial as it not only requires fused kernels, but also activation recomputation to avoid storing all intermediate $\mathbf{u}, \mathbf{v}$ vectors. I may cover this in a future post though!
[^sparsity]: Here, "with sparsity" refers to NVIDIA's [structured sparsity](https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/), a Tensor Core feature which essentially doubles your compute throughput if you have a 2:4 sparsity pattern, i.e. among each group of four contiguous values, at least two are zero. Since this feature effectively requires a 50% sparsity rate whereas neural network matrices are dense (unless you prune them for inference), I'm a bit skeptical regarding the relevance of this "with sparsity" metric. I guess it's yet another trick from NVIDIA's marketing guys to boost GPU stats.
[^num-warps]: Out of curiosity, I also tuned $\text{num_warps}$ and ran a grid search on $(\text{BLOCK\_SIZE},\text{num\_warps})\in[64,128,256,512,1024] \times [4,8,16\]$. I found that depending on the batch size $B$, different configurations yielded the best results. I chose to omit this ablation here for the sake of simplicity.
[^mbs]: Here we define the microbatch size as $mbs={seq\\_per\\_batch}\times{seq\\_len}$, where we commonly have $seq\\_per\\_batch \in [1,2,4]$ and $seq\\_len \in [4096, 8192, 16,384]$ in pretraining pipelines.
[^hbm]: Technically, HBM is one *type* of VRAM technology, which is used in high-end AI chips e.g. NVIDIA's Hopper architecture. But not all GPUs have their HBM VRAM. In fact, gaming GPUs using GDDR, which is much less expensive than HBM. (it's the reason why your gaming graphics card doesn't cost $30k)