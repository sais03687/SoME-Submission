// ============================================================================
// Math primitives shared across sandboxes.
// All functions are pure; no global state except seeded RNG instances.
// ============================================================================

/**
 * Mulberry32 PRNG. Tiny, fast, deterministic — perfect for reproducible sims.
 * @param {number} seed
 * @returns {() => number} a function returning uniform[0,1)
 */
export function makeRng(seed) {
  let s = seed >>> 0;
  return function () {
    s |= 0; s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Box-Muller Gaussian sampler with a stash for the unused second value.
 */
export function makeGaussian(rng) {
  let stashed = null;
  return function () {
    if (stashed !== null) {
      const v = stashed;
      stashed = null;
      return v;
    }
    let u1, u2;
    do { u1 = rng(); } while (u1 === 0);
    u2 = rng();
    const mag = Math.sqrt(-2.0 * Math.log(u1));
    const z0 = mag * Math.cos(2.0 * Math.PI * u2);
    const z1 = mag * Math.sin(2.0 * Math.PI * u2);
    stashed = z1;
    return z0;
  };
}

/**
 * Choose +1 or -1 with equal probability.
 */
export function rademacher(rng) {
  return rng() < 0.5 ? -1 : 1;
}

/**
 * Heat kernel: probability density for a 1D Brownian motion at time t.
 * @param {number} x
 * @param {number} t
 * @param {number} D diffusion constant (D=1/2 gives standard Brownian)
 */
export function heatKernel(x, t, D = 0.5) {
  if (t <= 0) t = 1e-6;
  return Math.exp(-x * x / (4 * D * t)) / Math.sqrt(4 * Math.PI * D * t);
}

/**
 * Gaussian density with given mean and variance.
 */
export function gaussianPdf(x, mean, variance) {
  return Math.exp(-(x - mean) * (x - mean) / (2 * variance)) /
         Math.sqrt(2 * Math.PI * variance);
}

/**
 * Numerical derivative.
 */
export function deriv(f, x, h = 1e-3) {
  return (f(x + h) - f(x - h)) / (2 * h);
}

/**
 * Numerical second derivative.
 */
export function deriv2(f, x, h = 1e-3) {
  return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h);
}

/**
 * Clamp x to [a, b].
 */
export function clamp(x, a, b) {
  return Math.max(a, Math.min(b, x));
}

/**
 * Linear interpolation.
 */
export function lerp(a, b, t) { return a + (b - a) * t; }

/**
 * Linear map from [aLo, aHi] to [bLo, bHi].
 */
export function remap(x, aLo, aHi, bLo, bHi) {
  return bLo + (bHi - bLo) * (x - aLo) / (aHi - aLo);
}

/**
 * Build an equispaced array.
 */
export function linspace(a, b, n) {
  const arr = new Float64Array(n);
  if (n === 1) { arr[0] = a; return arr; }
  const step = (b - a) / (n - 1);
  for (let i = 0; i < n; i++) arr[i] = a + i * step;
  return arr;
}

/**
 * Histogram into bins.
 * @param {Float64Array} values
 * @param {number} loEdge
 * @param {number} hiEdge
 * @param {number} nBins
 * @returns {{counts: Int32Array, edges: Float64Array, density: Float64Array}}
 */
export function histogram(values, loEdge, hiEdge, nBins) {
  const counts = new Int32Array(nBins);
  const edges = linspace(loEdge, hiEdge, nBins + 1);
  const w = (hiEdge - loEdge) / nBins;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v < loEdge || v >= hiEdge) continue;
    const b = Math.floor((v - loEdge) / w);
    if (b >= 0 && b < nBins) counts[b]++;
  }
  const density = new Float64Array(nBins);
  const total = values.length;
  if (total > 0) {
    for (let i = 0; i < nBins; i++) density[i] = counts[i] / (total * w);
  }
  return { counts, edges, density };
}

/**
 * Single Brownian step (Euler-Maruyama, 1D).
 */
export function brownianStep(x, dt, gauss) {
  return x + Math.sqrt(dt) * gauss();
}

/**
 * SiLU activation: x * sigmoid(x).
 */
export function silu(x) {
  return x / (1 + Math.exp(-x));
}

/**
 * 2D vector helpers — kept inline since they're trivial but readable.
 */
export const v2 = {
  make: (x, y) => ({ x, y }),
  add: (a, b) => ({ x: a.x + b.x, y: a.y + b.y }),
  sub: (a, b) => ({ x: a.x - b.x, y: a.y - b.y }),
  scale: (a, k) => ({ x: a.x * k, y: a.y * k }),
  dot: (a, b) => a.x * b.x + a.y * b.y,
  norm: (a) => Math.sqrt(a.x * a.x + a.y * a.y),
};
