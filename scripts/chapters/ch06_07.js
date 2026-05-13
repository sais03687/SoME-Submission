// ============================================================================
// Chapter 6 — Quantum shadow.  Wave packet evolves under Schrödinger; slider
// rotates time into the complex plane; reader watches it become diffusion.
// Chapter 7 — Langevin.  2D potential landscape with particles.
// ============================================================================

import { makeRng, makeGaussian, linspace, clamp } from '../core/math.js';
import { setupCanvas, makeTransform, drawAxes, drawCurve, clearCanvas, dot,
         arrow, drawLabel, drawAxisLabels, makeResponsive } from '../core/canvas.js';
import { slider, presetRow, actionButton, readout,
         makeSandbox, addControls, setCaption, makeVisibilityLoop } from '../core/ui.js';

// ----------------------------------------------------------------------------
// Chapter 6 — Quantum shadow with Wick rotation
// ----------------------------------------------------------------------------
export function initQuantumSandbox(rootId) {
  const root = document.getElementById(rootId);
  if (!root) return;
  const { canvas, controlsRow, captionEl } = makeSandbox(root,
    'rotate time into the imaginary plane — the wave becomes a diffusing bump');

  // State: t in [0, 1], theta in [0, pi/2] (theta = pi/2 -> imaginary time)
  const state = {
    t: 0.3,
    theta: 0,
    k: 6.0,     // carrier wavenumber
  };

  let renderCtx = null, cssW = 0, cssH = 0;
  makeResponsive(canvas, 1.6, (ctx, w, h) => { renderCtx = ctx; cssW = w; cssH = h; });

  const tSlider = slider({
    label: 'time t',
    min: 0.0, max: 1.0, step: 0.01, value: 0.3,
    format: v => v.toFixed(2),
    onChange: v => { state.t = v; },
  });
  const thetaSlider = slider({
    label: 'rotation angle θ',
    min: 0, max: Math.PI / 2, step: 0.01, value: 0,
    format: v => (v / Math.PI).toFixed(2) + 'π',
    onChange: v => { state.theta = v; },
  });
  const kSlider = slider({
    label: 'wavenumber k',
    min: 1, max: 12, step: 0.1, value: 6.0,
    format: v => v.toFixed(1),
    onChange: v => { state.k = v; },
  });
  addControls(controlsRow, tSlider, thetaSlider, kSlider);

  setCaption(captionEl,
    `On the left: a wave packet ψ(x,t) evolving under the free Schrödinger ` +
    `equation.  The violet curve is Re(ψ); its envelope is the magnitude.  ` +
    `On the right: the analogous heat equation solution for the same initial ` +
    `Gaussian — pure smoothing, no oscillation.  Drag θ from 0 to π/2: time ` +
    `rotates from real to imaginary, and the wave packet morphs <em>continuously</em> ` +
    `into the diffusion bump.  The equations are the same up to one substitution.`);

  // Analytic free wave packet (Gaussian envelope, carrier wave)
  // Under t -> -i τ, oscillation factor exp(-i ω t) becomes exp(-ω τ) — decay.
  // For visualisation we just interpolate continuously between oscillating and smoothing.
  function evalSchrodinger(x, t, theta) {
    // sigma^2 grows like 1 + (effective_t)^2 where effective_t mixes real and imaginary
    // In a clean approximation, the magnitude broadens as a Gaussian with
    // sigma^2(t) = sigma_0^2 + something*t.  We make this depend on cos(theta) for
    // the oscillating part and sin(theta) for the smoothing part.
    const sigma0sq = 0.18;
    // Real-time spreading: sigma2 grows quadratically due to dispersion
    const realSpread = 0.5 * t * t;
    // Imaginary-time spreading: sigma2 grows linearly (heat eqn)
    const imagSpread = 1.5 * t;
    const sigma2 = sigma0sq + Math.cos(theta) ** 2 * realSpread
                            + Math.sin(theta) ** 2 * imagSpread;
    const envelope = Math.exp(-x * x / (2 * sigma2)) / Math.sqrt(2 * Math.PI * sigma2);
    // Carrier wave: oscillation amplitude scales with cos(theta).  At theta = pi/2 it's gone.
    const omega = 0.5 * state.k * state.k;
    const carrier = Math.cos(state.k * x - omega * t);
    const oscBlend = Math.cos(theta);
    return {
      re: envelope * (oscBlend * carrier + (1 - oscBlend)),
      mag: envelope,
    };
  }

  function evalHeat(x, t) {
    const sigma2 = 0.18 + 1.5 * t;
    return Math.exp(-x * x / (2 * sigma2)) / Math.sqrt(2 * Math.PI * sigma2);
  }

  makeVisibilityLoop(root, () => {
    if (!renderCtx) return;
    const ctx = renderCtx;
    clearCanvas(ctx, cssW, cssH, '#0a0e1a');

    const splitX = cssW / 2;

    // ---- Left half: quantum side ----
    {
      const world = { xMin: -3.5, xMax: 3.5, yMin: -1.6, yMax: 1.6 };
      const screen = { x: 50, y: 28, width: splitX - 70, height: cssH - 76 };
      const T = makeTransform(screen, world);
      drawAxes(ctx, T, world);
      drawAxisLabels(ctx, T, world, screen, {
        xLabel: 'position  x',
        yLabel: 'Re(ψ)',
      });

      const xs = linspace(-3.5, 3.5, 240);
      const reYs = new Float64Array(xs.length);
      const envYs = new Float64Array(xs.length);
      for (let i = 0; i < xs.length; i++) {
        const e = evalSchrodinger(xs[i], state.t, state.theta);
        reYs[i] = e.re * 1.5;       // scale for display
        envYs[i] = e.mag * 1.5;
      }
      // Envelope outline
      drawCurve(ctx, T, xs, envYs, '#b084eb', 1, 0.4);
      const negEnv = new Float64Array(envYs.length);
      for (let i = 0; i < envYs.length; i++) negEnv[i] = -envYs[i];
      drawCurve(ctx, T, xs, negEnv, '#b084eb', 1, 0.4);
      // Main curve
      drawCurve(ctx, T, xs, reYs, '#b084eb', 2.5, 0.95);

      drawLabel(ctx, 'Schrödinger side', 30, 18,
        { color: '#b084eb', font: '12px "JetBrains Mono"', align: 'left', baseline: 'top' });
      drawLabel(ctx, `θ = ${(state.theta / Math.PI).toFixed(2)}π`,
        splitX - 24, 18,
        { color: '#f4d35e', font: '12px "JetBrains Mono"', align: 'right', baseline: 'top' });
    }

    // Divider
    ctx.strokeStyle = '#1f2735';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(splitX, 20);
    ctx.lineTo(splitX, cssH - 30);
    ctx.stroke();

    // ---- Right half: heat side (the reference) ----
    {
      const world = { xMin: -3.5, xMax: 3.5, yMin: -1.6, yMax: 1.6 };
      const screen = { x: splitX + 50, y: 28, width: splitX - 70, height: cssH - 76 };
      const T = makeTransform(screen, world);
      drawAxes(ctx, T, world);
      drawAxisLabels(ctx, T, world, screen, {
        xLabel: 'position  x',
        yLabel: 'density  p(x, t)',
      });

      const xs = linspace(-3.5, 3.5, 240);
      const ys = new Float64Array(xs.length);
      for (let i = 0; i < xs.length; i++) ys[i] = evalHeat(xs[i], state.t) * 1.5;
      drawCurve(ctx, T, xs, ys, '#5bc5f2', 2.5, 0.95);

      drawLabel(ctx, 'heat side', splitX + 30, 18,
        { color: '#5bc5f2', font: '12px "JetBrains Mono"', align: 'left', baseline: 'top' });
      drawLabel(ctx, `t = ${state.t.toFixed(2)}`,
        cssW - 24, 18,
        { color: '#5bc5f2', font: '12px "JetBrains Mono"', align: 'right', baseline: 'top' });
    }

    // Bottom caveat
    drawLabel(ctx,
      'Analogy, not identity.  Quantum amplitudes are not ordinary probabilities.',
      cssW / 2, cssH - 12,
      { color: '#e07856', font: '11px "JetBrains Mono"', align: 'center' });
  });
}

// ----------------------------------------------------------------------------
// Chapter 7 — Langevin dynamics on a 2D potential
// ----------------------------------------------------------------------------
const POTENTIALS = [
  {
    name: 'double well',
    V: (x, y) => (x * x - 1) ** 2 + 0.5 * y * y,
    grad: (x, y) => ({ x: 4 * x * (x * x - 1), y: y }),
    range: { xMin: -2.2, xMax: 2.2, yMin: -1.6, yMax: 1.6 },
  },
  {
    name: 'single bowl',
    V: (x, y) => 0.5 * (x * x + y * y),
    grad: (x, y) => ({ x: x, y: y }),
    range: { xMin: -2.2, xMax: 2.2, yMin: -1.6, yMax: 1.6 },
  },
  {
    name: 'four valleys',
    V: (x, y) => 0.5 * Math.sin(2 * x) * Math.sin(2 * y) + 0.1 * (x * x + y * y),
    grad: (x, y) => ({
      x: Math.cos(2 * x) * Math.sin(2 * y) + 0.2 * x,
      y: Math.sin(2 * x) * Math.cos(2 * y) + 0.2 * y,
    }),
    range: { xMin: -2.4, xMax: 2.4, yMin: -1.8, yMax: 1.8 },
  },
];

export function initLangevinSandbox(rootId) {
  const root = document.getElementById(rootId);
  if (!root) return;
  const { canvas, controlsRow, captionEl } = makeSandbox(root,
    'drift biases motion downhill; noise lets particles cross ridges');

  const state = {
    potIdx: 0,
    beta: 4,      // inverse temperature
    nParticles: 200,
    particles: [],
    rng: null, gauss: null,
    showField: true,
    running: true,
    dt: 0.01,
  };

  function reset() {
    const p = POTENTIALS[state.potIdx];
    state.rng = makeRng(99);
    state.gauss = makeGaussian(state.rng);
    state.particles = [];
    for (let i = 0; i < state.nParticles; i++) {
      // uniform over the visible region
      state.particles.push({
        x: p.range.xMin + state.rng() * (p.range.xMax - p.range.xMin),
        y: p.range.yMin + state.rng() * (p.range.yMax - p.range.yMin),
      });
    }
  }
  reset();

  function step(dt) {
    const p = POTENTIALS[state.potIdx];
    const noiseScale = Math.sqrt(2 * dt / state.beta);
    for (const pt of state.particles) {
      const g = p.grad(pt.x, pt.y);
      pt.x += -g.x * dt + noiseScale * state.gauss();
      pt.y += -g.y * dt + noiseScale * state.gauss();
      pt.x = clamp(pt.x, p.range.xMin, p.range.xMax);
      pt.y = clamp(pt.y, p.range.yMin, p.range.yMax);
    }
  }

  let renderCtx = null, cssW = 0, cssH = 0;
  // Pre-rendered contour image cache (one per potential, computed when first needed)
  const contourCaches = new Map();

  function buildContours(potIdx, w, h) {
    const off = document.createElement('canvas');
    off.width = w; off.height = h;
    const cctx = off.getContext('2d');
    const p = POTENTIALS[potIdx];
    const world = p.range;
    const T = makeTransform({ x: 0, y: 0, width: w, height: h }, world);

    // Sample V on a grid; render as semi-transparent intensity bands
    const nx = 240, ny = 160;
    const dx = (world.xMax - world.xMin) / nx;
    const dy = (world.yMax - world.yMin) / ny;
    const img = cctx.createImageData(w, h);
    // For each pixel compute V; render via a colormap
    for (let py = 0; py < h; py++) {
      for (let px = 0; px < w; px++) {
        const wc = T.fromScreen(px, py);
        const v = p.V(wc.x, wc.y);
        // V mapped to brightness 0..1 (low V -> brighter orange)
        const vClamp = clamp(v, 0, 4);
        // Render as a faint orange tint
        const intensity = 1 - vClamp / 4;
        const r = Math.round(245 * intensity * 0.35);
        const g = Math.round(166 * intensity * 0.35);
        const b = Math.round(35  * intensity * 0.35);
        const idx = (py * w + px) * 4;
        img.data[idx] = r; img.data[idx+1] = g; img.data[idx+2] = b;
        img.data[idx+3] = 90;
      }
    }
    cctx.putImageData(img, 0, 0);

    // Overlay contour curves at fixed V levels using marching squares (sparse)
    const levels = [0.05, 0.3, 0.7, 1.2, 1.8, 2.6, 3.4];
    cctx.strokeStyle = '#f5a623';
    cctx.lineWidth = 1.0;
    cctx.globalAlpha = 0.5;
    // Build a V grid at lower resolution
    const gridNx = 200, gridNy = 130;
    const gridDx = (world.xMax - world.xMin) / gridNx;
    const gridDy = (world.yMax - world.yMin) / gridNy;
    const V = new Float64Array(gridNx * gridNy);
    for (let j = 0; j < gridNy; j++) {
      for (let i = 0; i < gridNx; i++) {
        const x = world.xMin + (i + 0.5) * gridDx;
        const y = world.yMin + (j + 0.5) * gridDy;
        V[j * gridNx + i] = p.V(x, y);
      }
    }
    for (const L of levels) {
      // Simple marching squares: emit a line segment in each cell where V crosses L
      cctx.beginPath();
      for (let j = 0; j < gridNy - 1; j++) {
        for (let i = 0; i < gridNx - 1; i++) {
          const v00 = V[j * gridNx + i];
          const v10 = V[j * gridNx + (i + 1)];
          const v01 = V[(j + 1) * gridNx + i];
          const v11 = V[(j + 1) * gridNx + (i + 1)];
          const corners = [v00, v10, v11, v01];
          // 4-bit index of corners above the level
          let idx = 0;
          if (v00 > L) idx |= 1;
          if (v10 > L) idx |= 2;
          if (v11 > L) idx |= 4;
          if (v01 > L) idx |= 8;
          if (idx === 0 || idx === 15) continue;
          // Bilerp the two intersections — for simplicity just use a centred segment
          const cx = world.xMin + (i + 0.5) * gridDx;
          const cy = world.yMin + (j + 0.5) * gridDy;
          // skip — just draw a dot at the centre to indicate transition
          const sp = T.toScreen(cx, cy);
          cctx.moveTo(sp.x, sp.y);
          cctx.lineTo(sp.x + 0.5, sp.y + 0.5);
        }
      }
      cctx.stroke();
    }
    cctx.globalAlpha = 1;

    return off;
  }

  makeResponsive(canvas, 1.65, (ctx, w, h) => {
    renderCtx = ctx; cssW = w; cssH = h;
    contourCaches.clear();
  });

  // Controls
  const potRow = presetRow({
    labels: POTENTIALS.map(p => p.name),
    active: 0,
    onChange: (i) => { state.potIdx = i; reset(); contourCaches.clear(); },
  });
  const betaSlider = slider({
    label: 'inverse temperature β',
    min: 0.5, max: 25, step: 0.5, value: 4,
    format: v => v.toFixed(1),
    onChange: v => { state.beta = v; },
  });
  const nSlider = slider({
    label: 'particles',
    min: 20, max: 800, step: 20, value: 200,
    format: v => v.toFixed(0),
    onChange: v => { state.nParticles = v; reset(); },
  });
  const fieldBtn = actionButton('toggle field arrows',
    () => { state.showField = !state.showField; });
  const pauseBtn = actionButton('pause / resume',
    () => { state.running = !state.running; });
  const resetBtn = actionButton('reset', reset);
  addControls(controlsRow, betaSlider, nSlider, fieldBtn, pauseBtn, resetBtn);
  root.appendChild(potRow.element);

  setCaption(captionEl,
    `Each particle obeys dX = −∇V dt + √(2β⁻¹) dW.  The orange tint shows ` +
    `the landscape's elevation; arrows show the drift direction (−∇V).  ` +
    `Cooling (β → ∞) freezes particles in the valleys.  Heating (β → 0) ` +
    `washes them across the whole landscape.  Watch barrier crossings happen ` +
    `at intermediate β: rare, but real.`);

  makeVisibilityLoop(root, () => {
    if (!renderCtx) return;
    if (state.running) {
      for (let s = 0; s < 2; s++) step(state.dt);
    }
    const ctx = renderCtx;
    clearCanvas(ctx, cssW, cssH, '#0a0e1a');

    const p = POTENTIALS[state.potIdx];
    const T = makeTransform({ x: 0, y: 0, width: cssW, height: cssH }, p.range);

    // Background (contour tint)
    let contourImg = contourCaches.get(state.potIdx);
    if (!contourImg) {
      contourImg = buildContours(state.potIdx, cssW, cssH);
      contourCaches.set(state.potIdx, contourImg);
    }
    ctx.drawImage(contourImg, 0, 0);

    drawAxes(ctx, T, p.range);

    // Overlay axis labels in corners (canvas is fully filled with potential tint)
    drawLabel(ctx, 'position x →', cssW - 8, cssH - 6,
      { color: '#9da8c2', font: '11px "JetBrains Mono"', align: 'right', baseline: 'bottom' });
    drawLabel(ctx, '↑ position y', 8, 8,
      { color: '#9da8c2', font: '11px "JetBrains Mono"', align: 'left', baseline: 'top' });
    drawLabel(ctx, 'brighter orange = valley (low V),  arrows = −∇V (drift downhill)',
      cssW / 2, 8,
      { color: '#9da8c2', font: '11px "JetBrains Mono"', align: 'center', baseline: 'top' });

    // Field arrows
    if (state.showField) {
      for (let ix = 0; ix < 14; ix++) {
        for (let iy = 0; iy < 9; iy++) {
          const x = p.range.xMin + (ix + 0.5) / 14 * (p.range.xMax - p.range.xMin);
          const y = p.range.yMin + (iy + 0.5) / 9 * (p.range.yMax - p.range.yMin);
          const g = p.grad(x, y);
          const mag = Math.hypot(g.x, g.y);
          if (mag < 0.05) continue;
          const len = 0.18;
          const vx = -g.x / mag * len;
          const vy = -g.y / mag * len;
          const s1 = T.toScreen(x, y);
          const s2 = T.toScreen(x + vx, y + vy);
          arrow(ctx, s1.x, s1.y, s2.x, s2.y, '#f5a623',
            { lineWidth: 1.0, headLen: 4, alpha: 0.5 });
        }
      }
    }

    // Particles
    for (const pt of state.particles) {
      const sp = T.toScreen(pt.x, pt.y);
      dot(ctx, sp.x, sp.y, 2.2, '#f4f6fb', 0.85);
    }

    // Label
    drawLabel(ctx, `β = ${state.beta.toFixed(1)}`,
      cssW - 12, 16,
      { color: '#f4d35e', font: '13px "JetBrains Mono"', align: 'right', baseline: 'top' });
  });
}
