# The Geometry of Noise

A long-form interactive essay tracing one mathematical idea — randomness becoming geometry — from random walks through diffusion, quantum mechanics, optimization, and generative AI. Eleven chapters, ten live sandboxes, one real trained neural network.

## Reading the article

Open `index.html` in any modern browser. No build step, no server (for most things), no dependencies beyond what's bundled in `vendor/`.

If you want to develop locally with the score-field sandbox (Chapter 9), you need a static server because ES modules and `fetch()` to JSON don't work over `file://`:

```bash
python3 -m http.server 8000
# then open http://127.0.0.1:8000
```

## What's in here

```
.
├── index.html                       Article + all chapter prose
├── styles/
│   └── main.css                     Typography, layout, sandbox styling
├── scripts/
│   ├── core/
│   │   ├── math.js                  RNG, Gaussian, heat kernel, vectors
│   │   ├── canvas.js                DPI handling, coordinate transforms
│   │   ├── ui.js                    Sliders, buttons, presets, readouts
│   │   ├── story.js                 Story-so-far widget + sticky TOC
│   │   └── score_net.js             Pure-JS forward pass of trained model
│   └── chapters/
│       ├── ch00_01.js               Hook + random walk sandboxes
│       ├── ch02_03.js               Scaling limit + Brownian roughness
│       ├── ch04_05.js               Laplacian + heat equation
│       ├── ch06_07.js               Quantum + Langevin
│       └── ch08_09.js               Optimization + score field
├── assets/
│   ├── score_model.json             1.5 MB: weights of a trained denoising score net
│   └── two_moons.json               600 sample points from the training distribution
└── vendor/
    └── katex/                       KaTeX 0.16.9 bundled locally for equation rendering
```

Total weight: roughly **2 MB**, dominated by the trained score model. The actual article + sandboxes + math rendering is under 400 KB.

## The score model (Chapter 9)

The neural network in Chapter 9 was trained in PyTorch on a 2D two-moons distribution using variance-preserving denoising score matching (Song et al. 2021). Architecture:

- Time embedding: sinusoidal features at 16 fixed frequencies (32 dims) → `Linear(32→128) → SiLU → Linear(128→128)`
- Main net: `concat(x, t_emb)` (130 dims) → `Linear(130→128) → SiLU → Linear(128→128) → SiLU → Linear(128→128) → SiLU → Linear(128→2)`
- Schedule: VP-SDE, β_min = 0.1, β_max = 20.0
- Training: 6000 steps of Adam, batch 512, lr 2e-3

The weights are serialized to `assets/score_model.json` and loaded at runtime. The forward pass is reimplemented in vanilla JS (`scripts/core/score_net.js`) — verified to match the PyTorch output to 1e-7 floating-point precision. No deep-learning library is needed in the browser.

Reverse-SDE sampling uses Euler-Maruyama on Anderson's reverse-time SDE (Anderson 1982):
```
dx = [-0.5 β(t) x - β(t) s_θ(x, t)] dt + sqrt(β(t)) dW̄
```
integrated from t=1 down to t=0. About 100 steps per sample, all run client-side.

## Deploying

Drop the whole folder onto any static host:

- **GitHub Pages**: push to a repo, enable Pages, point at the folder root.
- **Cloudflare Pages / Vercel / Netlify**: drag and drop, or `git push`.
- **Any web server**: copy to docroot.

No build step. No environment variables. No backend.

## Color semantics

Strict — enforced by CSS custom properties in `:root`. The same colors mean the same things across all eleven chapters:

| token       | meaning                                      | hex         |
|-------------|----------------------------------------------|-------------|
| `--bg`        | background                                   | `#0a0e1a`   |
| `--particle`  | walkers, particles, hikers                   | `#f4f6fb`   |
| `--density`   | probability density, diffusion, heat        | `#5bc5f2`   |
| `--drift`     | drift, force, gradient                       | `#f5a623`   |
| `--quantum`   | Schrödinger amplitude, phase                | `#b084eb`   |
| `--ai`        | loss landscape, score field, learned things | `#3ddc97`   |
| `--neutral`   | axes, body text                              | `#c8cdd7`   |
| `--accent`    | highlighted symbols, takeaways              | `#f4d35e`   |
| `--caution`   | caveats, "analogy not identity"             | `#e07856`   |

## Article structure

Eleven chapters, in two halves:

**Half 1 (chapters 1–5) — build the theory.** Discrete random walks → continuous limit → Brownian motion → infinitesimal generator → heat equation. Each chapter has its own five-beat rhythm: (1) hook with everyday analogy, (2) first sandbox, (3) promote to math, (4) second sandbox, (5) bridge to next chapter. The hiker is the protagonist.

**Half 2 (chapters 6–9) — deploy the theory.** The hiker becomes, in turn, a wave (quantum), a particle in a landscape (Langevin), a parameter vector (optimization), and a candidate image (generative AI). The mathematical skeleton from Half 1 carries through unchanged. Chapter 10 (synthesis) ties it together.

Chapter 6 (quantum) is the only one *not* an application of the diffusion theory — it's framed honestly as a structural cousin connected via Wick rotation. The "Analogy, not identity" caveat is rendered prominently.

## Mathematical rigor

Every chapter has the main intuition flow plus a `<details>`-collapsed "going deeper" section with proper derivations. Examples of what's in the expandable sections:

- Chapter 1: why variances of independent random variables add (via covariance).
- Chapter 2: Donsker's invariance principle.
- Chapter 3: heuristic argument for nowhere-differentiability of Brownian paths.
- Chapter 4: Kolmogorov backward equation; the generator and its action.
- Chapter 5: heat kernel as Green's function; convolution structure of solutions.
- Chapter 6: path-integral connection between QM and statistical mechanics.
- Chapter 7: Fokker-Planck derivation; Kramers escape rate.
- Chapter 8: SGD as discretized Langevin, with state-dependent noise.
- Chapter 9: why denoising score matching has the marginal score as its optimum; Anderson's reverse-SDE result.

These sections are collapsed by default so the main flow remains intuitive. Click any of them to read the full derivation.

## Browser support

Tested in Chromium 131. Should work in any browser with ES module support and Canvas 2D — Chrome 80+, Firefox 75+, Safari 13+, Edge 80+. Mobile Safari and Chrome Mobile work but the wide sandboxes will be cramped on small screens.

## Performance

Most sandboxes run at 60 fps with a few thousand particles. The Langevin sandbox is the heaviest (particles + contour rendering); the score-field sandbox does a forward pass through the network on every grid point each time you drag the noise slider (about 324 forward passes per slider event), which is fine for the small network we shipped but would be slow for production-scale models.

Sandboxes are paused when scrolled out of view, so the article doesn't cost CPU when you're reading prose far from the active sandbox.

## License

Source code: see project repo. The trained score model is small enough (1.5 MB) and trained on synthetic data (two moons), so it's freely usable. Adapt for your own essays as you like.
