// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-projects",
          title: "projects",
          description: "A growing collection of cool projects. (older projects can be found on my GitHub)",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "post-fused-amp-furious-sinkhorn-triton-kernels",
      
        title: "Fused &amp; Furious: Sinkhorn Triton Kernels",
      
      description: "TL;DR: DeepSeek&#39;s recent mHC paper relies on Sinkhorn&#39;s algorithm to project matrices onto Birkhoff&#39;s polytope. The looping nature of the algorithm introduces severe memory-boundedness, which can be mitigated by fusing the algorithm into a single kernel. We implement increasingly fast versions of the algorithm in Triton.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/fused-and-furious/";
        
      },
    },{id: "post-residual-matrix-transformers",
      
        title: "Residual Matrix Transformers",
      
      description: "TL;DR: As we increase the size of (standard) transformers, parameters and FLOPs scale quadratically, but the residual stream width scales linearly. Thus, the residual stream can become a bottleneck as we scale up. We discuss the RMT paper, which proposes a matrix residual stream to address this issue.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/residual-matrix-transformer/";
        
      },
    },{id: "post-filtrations-demystified",
      
        title: "Filtrations demystified",
      
      description: "TL;DR: Filtrations are a key ingredient in defining stochastic processes and modeling the accumulation of available information over time. Filtrations are also often poorly understood; this post aims to demystify them.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/filtrations/";
        
      },
    },{id: "post-measurability-and-σ-algebras",
      
        title: "Measurability and σ-algebras",
      
      description: "TL;DR: σ-algebras are omnipresent when doing probability, yet they are somewhat arcane. Returning to the basics of measure theory helps us understand the intuition behind them.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/measurability/";
        
      },
    },{id: "post-subliminal-learning-amp-information-bandwidth",
      
        title: "Subliminal Learning &amp; Information Bandwidth",
      
      description: "TL;DR: Take a LLM and finetune it to love owls. Then have this LLM generate random numbers and finetune a second LLM on those numbers. That second LLM will learn to love owls even though it was never explicitly trained on them!",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/subliminal-learning/";
        
      },
    },{id: "post-copula-theory-and-the-subprime-mortgage-crisis",
      
        title: "Copula Theory and the Subprime Mortgage Crisis",
      
      description: "TL;DR: Copulas are a powerful tool for modeling the correlation structure between random variables. We propose an intuition-first introduction to copula theory, culminating in a discussion of the role of copulas in the 2008 subprime mortgage crisis.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/copulas/";
        
      },
    },{id: "post-the-magic-of-embeddings",
      
        title: "The Magic of Embeddings",
      
      description: "TL;DR: Embeddings are so powerful that they can seem almost magical. We go back to the basics (linear algebra) with the Johnson-Lindenstrauss lemma, which illustrates the blessing of dimensionality.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/embeddings/";
        
      },
    },{id: "post-adding-salt-to-the-bitter-lesson",
      
        title: "Adding salt to the Bitter Lesson",
      
      description: "TL;DR: The &quot;Bitter Lesson&quot; of AI states that general methods that leverage computation are ultimately the most effective to build powerful AI systems. We propose to qualify this lesson by introducing the notion of signal-to-noise ratio (SNR) of the problem at hand. In domains such as quantitative finance and computational biology, I believe that the SNR is so low that Sutton&#39;s lesson may not directly apply.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/bitter-lesson/";
        
      },
    },{id: "post-the-curty-amp-marsili-forecasting-game",
      
        title: "The Curty &amp; Marsili Forecasting Game",
      
      description: "TL;DR: When faced with a forecasting task, one can either seek information or follow the crowd. The Curty &amp; Marsili game stacks fundamentalists against herders in a binary forecasting task, revealing phase coexistence and ergodicity breaking under certain conditions. We propose a theoretical study of the game&#39;s behavior and validate it through ABM simulations.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/curty-marsili-game/";
        
      },
    },{id: "post-listening-to-the-market-mode",
      
        title: "Listening to the Market Mode",
      
      description: "TL;DR: Performing PCA on returns amounts to constructing a statistical factor model. The largest eigenvalue corresponds to the market mode and far outweighs the other factors. Thus, one can perform rolling PCA on equities&#39; returns to monitor the market risk over time.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/market-mode/";
        
      },
    },{id: "post-jeffreys-39-prior-in-bayesian-inference",
      
        title: "Jeffreys&#39; Prior in Bayesian Inference",
      
      description: "TL;DR: Bayesian inference requires us to specify a prior distribution. When we&#39;re unsure what prior to pick and want to stay as objective as possible, one option is to use Jeffreys&#39; prior, which leverages the Fisher information to provide a reparametrization-invariant prior.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/jeffreys-prior/";
        
      },
    },{id: "post-regression-dilution",
      
        title: "Regression Dilution",
      
      description: "TL;DR: When covariates in linear regression are subject to noise, the estimated regression coefficients shrink towards zero. We derive this effect mathematically and illustrate it with simulations.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/regression-dilution/";
        
      },
    },{id: "post-a-geodesic-from-cat-to-dog",
      
        title: "A geodesic from cat to dog",
      
      description: "TL;DR: Entropic regularization relaxes the Kantorovitch problem into a strictly convex problem which can be solved efficiently with the Sinkhorn algorithm. We can use this to efficiently compute Wasserstein distances, barycenters, and finally geodesics between distributions.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/ot-geodesic/";
        
      },
    },{id: "post-solving-the-assignement-problem-using-optimal-transport",
      
        title: "Solving the assignement problem using Optimal Transport",
      
      description: "TL;DR: The discrete Kantorovich problem amounts to a LP problem. In the uniform case, the solution is a permutation matrix which in fact solves the assignement problem.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/ot-assignement-problem/";
        
      },
    },{id: "post-intuitions-behind-benford-39-s-law",
      
        title: "Intuitions behind Benford&#39;s Law",
      
      description: "TL;DR: Many real-world datasets follow Benford&#39;s Law, which states that distribution of the first digit is not uniform. We provide three different intuitions behind this phenomenon.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/benford-law/";
        
      },
    },{id: "post-the-case-against-leveraged-etfs",
      
        title: "The case against leveraged ETFs",
      
      description: "TL;DR: Leveraged ETFs amplify daily returns, which is not the same as basic leverage, especially in the long term. Digging into the math reveals that leveraged ETFs are not suitable buy-and-hold investments as they 1) exhibit huge price swings 2) incur a volatility drag.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/leveraged-etf/";
        
      },
    },{id: "projects-diffusion-schrödinger-bridge",
          title: 'Diffusion Schrödinger Bridge',
          description: "Theoretical study of the Schrödinger Bridge problem &amp; PyTorch implementation of the Diffusion Schrödinger Bridge algorithm to study convergence properties in the Gaussian case.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/dsb/";
            },},{id: "projects-equivariant-diffusion-for-molecule-generation-in-3d",
          title: 'Equivariant Diffusion for Molecule Generation in 3D',
          description: "Demonstration of the benefits of incorporating E(3)-equivariance in Graph Neural Networks through toy model experiments on the QM9 drugs dataset.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/e3egnn/";
            },},{id: "projects-knowledge-graph-integration-for-biological-foundation-models",
          title: 'Knowledge Graph Integration for Biological Foundation Models',
          description: "Presentation of four distinct approaches to integrate knowledge graphs for biological foundation models.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/kg-for-bfm/";
            },},{id: "projects-score-based-generative-modeling",
          title: 'Score-Based Generative Modeling',
          description: "Theoretical study of Score-Based Generative Modeling &amp; PyTorch implementation to compare Langevin, SDE and ODE sampling methods. Also explored controlled generation techniques, including conditional generation and inpainting.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/sde/";
            },},{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
