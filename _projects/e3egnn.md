---
layout: page
title: Equivariant Diffusion for Molecule Generation in 3D
description: Demonstration of the benefits of incorporating E(3)-equivariance in Graph Neural Networks through toy model experiments on the QM9 drugs dataset.
img: assets/video/e3egnn/molecule-diffusion.gif
importance: 1
category: work
related_publications: false
---
<div class="row justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/e3egnn/overview_diffusion_equivariance.png" title="Equivariance" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Overview of the EDM. To generate a molecule, a normal distributed set of points is denoised into a molecule consisting of atom coordinates $x$ in 3D and atom types $h$. As the model is rotation equivariant, the likelihood is preserved when a molecule is rotated by $R$.
</div>

NB: This post is just a recap of my work, but you can get my full report <a href="https://github.com/gaetanX21/e3-egnn/blob/main/report/report.pdf">here</a>.

## Project Overview

The E(3) Equivariant Diffusion Model (EDM) presented by Hoogeboom et al. [^e3egnn] introduces a novel paradigm for 3D molecular generation. It is a diffusion model which goes from an initial noise $x_T\sim\mathcal{N}(0,I)$ in $(\mathbb{R}^{\text{nf}+3})^M$ (where $M$ is the number of atoms and $\text{nf}$ the atom features) to a final denoised object $x_0$. A final reconstruction step then transforms $x_0$ to a 3D molecule with $M$ atoms. The novelty in this approach comes from the utilization of an E(3) invariant Graph Neural Network as the denoising network, which effectively creates an *inductive bias* that helps the EDM generalize better.

This project discusses the geometric underpinnings of the E(3) EDM. We demonstrate the benefits of incorporating E(3)-equivariance in Graph Neural Networks (GNNs) through a toy model experiment.


## E(3) Equivariant Diffusion Model

The E(3) EDM is a complicated model so we break it up into chunks to understand it better.

### Diffusion

Diffusion models are a class of generative models that operate by iteratively denoising a noised signal.
In essence, diffusion models define a (forward) noising process which diffuses the complex target distribution $p_\text{data}(x)$ into a simple distribution we know how to sample, usually a Gaussian distribution.
The training consists in learning the reverse process, usually by leveraging neural networks which given the noisy signal at step $t$ predict the noise increment so that we can recover the original signal at step $t-1$.
Once the reverse process is learned, one can sample from the target distribution by starting from the simple distribution $x_T \sim\mathcal{N}(0,I)$ and iteratively denoising it $T$ times to get a sample from the target distribution $x_0 \sim p_\text{data}(x)$.

Diffusion models were first introduced in the context of image generation [^ddpm] but they can also be applied to generate other modalities such as molecules.
One simple approach to do so is to represent the molecule as a vector $\left[x, h \right]$ where $x \in (\mathbb{R}^3)^M$ is the position of the atoms in 3D space and $h \in (\mathbb{R}^\text{nf})^M$ is the atom features.
Here $M$ is the number of atoms in the molecule and $\text{nf}$ is the number of features used to embed the atoms.
In the paper, the atom features are the atom type (H, C, O, N, F one-hot encoded in a 5-dimensional vector) and charge (integer) such that $\text{nf} = 6$.

It makes sense to treat differently the atom positions $x$ and the atom features $h$ since the former live in $\mathbb{R}^3$ and are subject to the symmetries of the Euclidean group $E(3)$ while the latter live in $\mathbb{R}^\text{nf}$ and are not subject to these symmetries.
For this reason, the latent space distingues between the atom positions and the atom features by representing the molecule as a vector concatenation $\left[z^{(x)}_t, z^{(h)}_t \right]$.
However, the two vectors do interact in the reverse diffusion process i.e. there is only one diffusion process for the whole molecule.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/e3egnn/diffusion.png" title="Diffusion" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Diffusion process for molecule generation.
</div>

### Graph Neural Networks

Graph Neural Networks (GNNs) are a class of neural networks that operate on graph-structured data.
They are designed to capture the structure of the graph and the interactions between its nodes.
More precisely, *a GNN is an optimizable transformation on all attributes of the graph (nodes, edges, and global attributes) 
that preserves graph symmetries*. (permutation invariance) [^gentle-intro-gnn]

<div class="row justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/e3egnn/message_passing.png" title="Message Passing" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Message-passing scheme of GNNs.
</div>

### Equivariance, Invariance and the E(3) Group

Equivariance is a form of symmetry for functions. Roughly speaking, a function $f$ is equivariant to a group of transformations $G$ 
if and only if $\forall T\in G, f(T(x)) = T(f(x))$.
Likewise, invariance is when the function is constant under the action of the group i.e. $\forall T\in G, f(T(x)) = f(x)$.
In the context of molecule generation, we are interested in the spatial symmetries of the molecules. Indeed, only the relative positions of the atoms matter, not their absolute positions.
Therefore, we would want our generative model's final output to be invariant to the Euclidean group $E(3)$, which is the group of translations, rotations and reflections in 3D space.
Likewise, since we use diffusion to gradually denoise the molecule, we would want the denoising process to be equivariant to $E(3)$.
This **inductive bias** is expected to help the model learn the spatial structure of the molecules more efficiently i.e. the model will be able to generalize better and with less training data.

<div class="row justify-content-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/e3egnn/equivariance-invariance.png" title="Equivariance Invariance" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Equivariance & Invariance illustrated.
</div>

### E(3) Equivariant GNN

In their paper[^e3egnn], Hoogeboom et. al propose a GNN architecture that is equivariant to the Euclidean group $E(3)$. To do so, they first define an Equivariant Graph Convolutional Layer (EGCL) that is equivariant to the Euclidean group $E(3)$. This layer is then stacked to form the Equivariant Graph Neural Network (EGNN).

More precisely, the EGCL computes $x^{t+1}, h^{t+1} = \text{EGCL}(x^t, h^t)$ as follows:
$$
\begin{gather}
    m_{ij} = \phi_e(h_i^t, h_j^t, d_{ij}^2), \quad \tilde{e}_{ij} = \phi_{inf}(m_{ij}) \\
    h_i^{t+1} = \phi_h(h_i^t, \sum_{j\neq i} \tilde{e}_{ij} m_{ij}) \\
    x_i^{t+1} = x_i^t + \sum_{j\neq i} \frac{x_i^t - x_j^t}{d_{ij}+1} \phi_x(h_i^t, h_j^t, d_{ij}^2, a_{ij})
\end{gather}
$$
where $\phi_e, \phi_{inf}, \phi_h, \phi_x$ are neural networks and $d_{ij}$ is the Euclidean distance between atoms $i$ and $j$.

Importantly, the EGCL is equivariant to actions the Euclidean group $E(3)$ on the atom positions $x$ since replacing $x_i^t$ by $R x_i^t + t$ for any $R\in SO(3)$ and $t\in \mathbb{R}^3$ results in the same transformation on $x_i^{t+1}$.
Besides, one can easily show by induction that the EGNN is also equivariant to $E(3)$ since it is a stack of EGCLs.

### Wrapping it together: EDM

Finally, the Equivariant Diffusion Model (EDM) is obtained by combining the diffusion model and the EGNN.
The training objective is $\mathbb{E}_{t, \epsilon_t}[\|\| \epsilon_t - \hat{\epsilon}_t \|\|^2]$ i.e. we want to predict the noise.
We use the EGNN to predict the noise increment $\hat{\epsilon}_t$ given the noisy molecule $\left[x_t, h_t \right]$ nad the time step $t$.
To be precise, we set $\hat{\epsilon}_t = \left[ \hat{\epsilon}_t^{(x)}, \hat{\epsilon}_t^{(h)} \right] = \text{EGNN}(z_t^{(x)}, z_t^{(h)}) - \left[ z_t^{(x)}, 0 \right]$. This recentering trick is necessary for translational equivariance of the EGNN. More details can be found in the original paper[^e3egnn].

Once the EGNN is trained, we can sample from the target distribution by starting from the simple distribution $z_T \sim \mathcal{N}(0,I)$ and iteratively denoising it $T$ times to get $z_0$.
We must then decode $z_0 = \left[z_0^{(x)}, z_0^{(h)} \right]$ to get the molecule $\left[x_0, h_0 \right]$. Doing so is non-trivial since we have continuous (positions), categorical (atom type) and ordinal (charge) variables.
In the paper, the authors use Bayes rule and some approximations to decode $z_0$.
- The atom positions $x_0$ are obtained by sampling from a gaussian distribution centered around $z_0^{(x)}$ (plus a correction term).
- The atom type is obtained by sampling from a categorical distribution which essentially amounts to taking the nearest one-hot encoded vector to $(z_0^{(h)})_{1:5}$.
- The charge is obtained by sampling from a gaussian distribution centered around $(z_0^{(h)})$ which essentially amounts to taking the nearest integer to $(z_0^{(h)})$.
In addition, the authors use a heuristic distance-based method to create the edges of the molecule. That is, for each pair of atoms $i$ and $j$, they compute the distance $d_{ij}$ and based on that distance and a table of bond lengths for these atoms they decide whether there is a bond between $i$ and $j$ and if so what type of bond it is.

Finally, note that since the initial distribution $z_T \sim \mathcal{N}(0,I)$ is E(3)-invariant and the EGNN is E(3)-equivariant, the final distribution of the model is also E(3)-invariant. (this can be shown easily by induction)


## Experiments on toy model

The key contribution of Hoogeboom's paper is the introduction of E(3)-equivariance. Their choice is motivated by the spatial symmetries of molecules. **We aim to demonstrate the benefits of incorporating E(3)-equivariance in Graph Neural Networks (GNN) through a toy model experiment**. We will focus on a regression task, which is less computationally intensive than generation and yet sufficient to highlight the advantages of equivariant models.

### Dataset
Our dataset is **QM9**: it only contains small molecules with atoms in H, C, O, N, F. This allows us to focus on the model's ability to capture the spatial structure of the molecules without being overwhelmed by the complexity of larger molecules. Additionnally, since QM9 has over 100,000 molecules, we filter to keep only molecules with 12 atoms or less. This brings us down to 4005 molecules.

Each sample in the dataset is a graph with several properties:
- Number of atoms $M$
- Atom positions $x \in (\mathbb{R}^3)^M$
- Atom features $h \in (\mathbb{R}^\text{nf})^M$ where $\text{nf} = 11$
    - 1-5: atom type (one-hot: H, C, O, N, F)
    - 6: atomic number (number of protons)
    - 7: aromaticity (binary)
    - 8-10: electron orbital hybridization (one-hot: sp, sp2, sp3)
    - 11: number of hydrogens
- Graph edges (bonds) $\mathcal{E} \in \mathbb{R}^{2\times\text{\#edges}}$
- Edge features $\mathcal{E}_\text{type} \in \mathbb{R}^{\text{\#edges}\times 4}$ (one-hot: single, double, triple, aromatic)

The QM9 dataset has 19 regression targets which are various physical properties of the molecules. We will focus on the first target which is the **dipole moment** of the molecule. We chose this target because it is typically geometric so we expect to clearly see the benefits of 
incorporating E(3)-equivariance in the model.

### Models
Taking inspiration from[^colab-gnn], we consider the following models:
- MPNN: a Message Passing Neural Network (MPNN) built with stacked Graph Convolutional Layers (GCL)
- EGNN: an E(3)-equivariant GNN built with stacked E(3)-Equivariant Graph Convolutional Layers (EGCL)
- EGNN_edge: same architecture as EGNN but using the edge features $\mathcal{E}_\textnormal{type}$ as well when computing the messages

All models have $L=4$ layers and $d_\text{embed}=11$ (i.e. the same as the input dimension).

We also consider **LinReg**, a linear regression model (projects each atom to a scalar and sums them up to get the final prediction), for baseline comparison with the geometric models. We expect poor performance as it doesn't leverage the graph structure of the molecules.

<div style="display: flex; justify-content: center;">
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Perm-Eq Layers</th>
                <th>Perm-Inv</th>
                <th>E(3)-Eq Layer</th>
                <th>E(3)-Inv</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>MPNN</td>
                <td>Yes</td>
                <td>No</td>
                <td>No</td>
                <td>Yes</td>
            </tr>
            <tr>
                <td>EGNN</td>
                <td>Yes</td>
                <td>Yes</td>
                <td>Yes</td>
                <td>Yes</td>
            </tr>
            <tr>
                <td>EGNN_edge</td>
                <td>Yes</td>
                <td>Yes</td>
                <td>Yes</td>
                <td>Yes</td>
            </tr>
            <tr>
                <td>LinReg</td>
                <td>No</td>
                <td>Yes</td>
                <td>No</td>
                <td>Yes</td>
            </tr>
        </tbody>
    </table>
</div>
<p style="text-align: center;"><em>Table: Symmetry Properties of the Different Models</em></p>

### Results

The results are reported on two figures: the training loss and the validation loss.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/e3egnn/train_loss.png" title="train loss" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/e3egnn/validation_loss.png" title="validation loss" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Training and validation losses for the different models.
</div>

To begin with, the **LinReg** model performs very poorly as expected. Unlike its geometric counterparts, it cannot leverage the graph structure of the molecules and thus fails to learn anything meaningful.

Then, **MPNN** and **EGNN_edge** have similar training losses but **EGNN_edge** drastically outperforms **MPNN** on the validation set. This gap shows that **EGNN_edge**'s *inductive bias* helps it *generalize* better than **MPNN**.

Finally, **EGNN** is also significantly better than **MPNN** but learns much slower than **EGNN_edge**. This suggests that incorporating edge features is beneficial to the model's performance.
Nonetheless, edge features can intuitively be inferred from the atom positions and types (using a distance-based heuristic like in the original paper), so the improvement is not as significant as the one brought by E(3)-equivariance.
That would explain why the **EGNN** is still learning and could potentially be as good as **EGNN_edge** with more training. To verify this hypothesis, we trained the model for 1000 epochs, but the validation loss and training loss actually plateau around 500 epochs. There could be two reasons for that: either **EGNN** doesn't have enough capacity to infer the edge types itself (need for more layers or higher embedding dimension) or **EGNN** got stuck in a local minimum during training.

Finally, given the (intuitively) low complexity of our regression task, we expect that scaling up and fine-tuning the **EGNN** would lead to much better performance i.e. a much lower validation loss. However, the goal of our toy experiment was solely to demonstrate the superiority of E(3) equivariant models over architectures which do not leverage the geometry of the problem. We leave the task of maximizing performance to researchers with professional computational resources.

**References**:

[^e3egnn]: *Equivariant Diffusion for Molecule Generation in 3D*. Hoogeboom et al. [arXiv](https://arxiv.org/abs/2203.17003)
[^ddpm]: *Denoising Diffusion Probabilistic Models.*. Ho et al. [arXiv](https://arxiv.org/abs/2006.11239)
[^gentle-intro-gnn]: *A Gentle Introduction to Graph Neural Networks*. Benjamin Sanchez-Lengeling. [distill.pub](https://distill.pub/2021/gnn-intro/)
[^colab-gnn]: *A Gentle Introduction to Geometric Graph Neural Networks*. Chaitanya Joshi. [colab.google](https://colab.research.google.com/github/chaitjo/geometric-gnndojo/blob/main/geometric_gnn_101.ipynb)