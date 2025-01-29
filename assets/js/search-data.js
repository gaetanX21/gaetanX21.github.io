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
          description: "A growing collection of cool projects. (older projects are not listed here but can be found on my GitHub/CV)",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "nav-repositories",
          title: "repositories",
          description: "GitHub repos for my main projects.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/repositories/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "post-a-geodesic-from-cat-to-dog",
      
        title: "A geodesic from cat to dog",
      
      description: "TL;DR: Entropic regularization relaxes the Kantorovitch problem into a strictly convex problem which can be solved efficiently with the Sinkhorn algorithm. We can use this to efficiently compute Wasserstein distances, barycenters, and finally geodesics between distributions.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2024/ot-geodesic/";
        
      },
    },{id: "post-solving-the-assignement-problem-using-optimal-transport",
      
        title: "Solving the assignement problem using Optimal Transport",
      
      description: "TL;DR: The discrete Kantorovich problem amounts to a LP problem. In the uniform case, the solution is a permutation matrix which in fact solves the assignement problem.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2024/ot-assignement-problem/";
        
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
