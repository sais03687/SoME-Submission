// ============================================================================
// Pure-JS implementation of the score network's forward pass.
// Loads weights from /assets/score_model.json (exported from the trained
// PyTorch model).  No deep-learning library needed — the model is small.
// ============================================================================

import { silu } from './math.js';

/**
 * Load the model from JSON.
 */
export async function loadScoreModel(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch score model: ' + res.status);
  const json = await res.json();
  return makeScoreModel(json);
}

/**
 * Construct an inference function from a JSON spec.
 *
 * The architecture matches simulations/score_model.py:
 *   Time embedding:  t -> [sin(2π·t·f_k), cos(2π·t·f_k)] for 16 freqs (32 dims)
 *                    -> Linear(32, 128) -> SiLU -> Linear(128, 128)
 *   Main net:       concat([x (2 dims), t_emb (128)]) = 130 dims
 *                   -> Linear(130, 128) -> SiLU
 *                   -> Linear(128, 128) -> SiLU
 *                   -> Linear(128, 128) -> SiLU
 *                   -> Linear(128, 2)
 *
 * The returned function takes ({x, y}, t) and returns {sx, sy} — the predicted
 * score at that point at that noise level.
 */
export function makeScoreModel(json) {
  const W = json.weights;
  const freqs = json.arch.freqs;

  // Helper: dense layer y = W @ x + b  (W shape [out, in], x shape [in])
  function dense(weight, bias, x) {
    const out = new Float32Array(bias.length);
    const inDim = x.length;
    const outDim = bias.length;
    for (let i = 0; i < outDim; i++) {
      let s = bias[i];
      const row = weight[i];
      for (let j = 0; j < inDim; j++) s += row[j] * x[j];
      out[i] = s;
    }
    return out;
  }

  // Helper: in-place SiLU
  function applySilu(arr) {
    for (let i = 0; i < arr.length; i++) arr[i] = silu(arr[i]);
    return arr;
  }

  // Time embedding: returns 32-dim vector [sin(2π·t·f), cos(2π·t·f)]
  function timeEmbed(t) {
    const out = new Float32Array(32);
    for (let i = 0; i < 16; i++) {
      const angle = 2 * Math.PI * t * freqs[i];
      out[i] = Math.sin(angle);
      out[i + 16] = Math.cos(angle);
    }
    return out;
  }

  // Look up weight tensors.  Keys from PyTorch's named_parameters().
  // t_proj is nn.Sequential(Linear(32,128), SiLU, Linear(128,128))
  // .0 and .2 are the Linears.
  const tProj0_w = W['t_proj.0.weight'];   // [128, 32]
  const tProj0_b = W['t_proj.0.bias'];     // [128]
  const tProj2_w = W['t_proj.2.weight'];   // [128, 128]
  const tProj2_b = W['t_proj.2.bias'];     // [128]
  const net0_w = W['net.0.weight'];        // [128, 130]
  const net0_b = W['net.0.bias'];          // [128]
  const net2_w = W['net.2.weight'];        // [128, 128]
  const net2_b = W['net.2.bias'];          // [128]
  const net4_w = W['net.4.weight'];        // [128, 128]
  const net4_b = W['net.4.bias'];          // [128]
  const net6_w = W['net.6.weight'];        // [2, 128]
  const net6_b = W['net.6.bias'];          // [2]

  /**
   * Run the model at a single point.
   * @param {{x, y}} pt position in data space
   * @param {number} t noise level in [0, 1]
   * @returns {{sx, sy}} score vector at (pt, t)
   */
  function forward(pt, t) {
    // Time embedding -> projection
    const te = timeEmbed(t);
    const h1 = applySilu(dense(tProj0_w, tProj0_b, te));
    const tEmb = dense(tProj2_w, tProj2_b, h1);          // [128]

    // Concat [x.x, x.y, tEmb...] -> 130 dims
    const input = new Float32Array(130);
    input[0] = pt.x;
    input[1] = pt.y;
    for (let i = 0; i < 128; i++) input[2 + i] = tEmb[i];

    let h = applySilu(dense(net0_w, net0_b, input));
    h = applySilu(dense(net2_w, net2_b, h));
    h = applySilu(dense(net4_w, net4_b, h));
    const out = dense(net6_w, net6_b, h);    // [2]

    return { sx: out[0], sy: out[1] };
  }

  /**
   * VP schedule (must match PyTorch).
   */
  const beta_min = json.arch.schedule.beta_min;
  const beta_max = json.arch.schedule.beta_max;
  function beta(t) { return beta_min + t * (beta_max - beta_min); }
  function alphaBar(t) {
    const integ = beta_min * t + 0.5 * (beta_max - beta_min) * t * t;
    return Math.exp(-0.5 * integ);
  }
  function sigma(t) {
    const ab = alphaBar(t);
    return Math.sqrt(Math.max(0, 1 - ab * ab));
  }

  return { forward, beta, alphaBar, sigma };
}

/**
 * Take one reverse-SDE step (Euler-Maruyama) on a batch of points.
 * @param {{x,y}[]} pts mutated in place
 * @param {number} t current time (in [0,1], decreasing)
 * @param {number} dt step size (negative)
 * @param {Function} gauss zero-mean unit-variance Gaussian sampler
 * @param {object} model the model returned from makeScoreModel
 */
export function reverseSdeStep(pts, t, dt, gauss, model) {
  const bt = model.beta(t);
  for (let i = 0; i < pts.length; i++) {
    const s = model.forward(pts[i], t);
    // drift = -0.5 * beta * x - beta * score
    const driftX = -0.5 * bt * pts[i].x - bt * s.sx;
    const driftY = -0.5 * bt * pts[i].y - bt * s.sy;
    pts[i].x += driftX * dt;
    pts[i].y += driftY * dt;
    // diffusion (note: dt is negative, so use -dt for variance)
    const noiseScale = Math.sqrt(bt * (-dt));
    pts[i].x += noiseScale * gauss();
    pts[i].y += noiseScale * gauss();
  }
}
