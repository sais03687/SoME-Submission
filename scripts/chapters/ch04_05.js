// ============================================================================
// Chapter 4 — Laplacian generator.  Pick f(x), drag x_0 and h, watch the
// symmetric-average gap come out to ½ h² f''(x_0).  This is the IP scene.
// Chapter 5 — Heat equation.  Real-time particle sim with analytic kernel overlay.
// ============================================================================

import { makeRng, makeGaussian, deriv2, linspace, histogram, gaussianPdf, heatKernel }
  from '../core/math.js';
import { setupCanvas, makeTransform, drawAxes, drawCurve, drawHistogram,
         clearCanvas, dot, arrow, drawLabel, drawAxisLabels, makeResponsive } from '../core/canvas.js';
import { slider, presetRow, actionButton, readout,
         makeSandbox, addControls, setCaption, makeVisibilityLoop } from '../core/ui.js';

// ----------------------------------------------------------------------------
// Chapter 4 — Laplacian generator
// ----------------------------------------------------------------------------
const FUNCTIONS = [
  { name: 'sin·sloped',  f: (x) => 0.6 * Math.sin(0.9 * x) + 0.15 * x * x,
    fpp: (x) => -0.486 * Math.sin(0.9 * x) + 0.30 },
  { name: 'parabola',    f: (x) => 0.18 * x * x - 0.6,
    fpp: (x) => 0.36 },
  { name: 'flat',        f: (x) => 0.5 * x - 0.3,
    fpp: (x) => 0 },
  { name: 'bump',        f: (x) => 1.5 * Math.exp(-x * x / 2),
    fpp: (x) => 1.5 * Math.exp(-x * x / 2) * (x * x - 1) },
];

export function initLaplacianSandbox(rootId) {
  const root = document.getElementById(rootId);
  if (!root) return;
  const { canvas, controlsRow, captionEl } = makeSandbox(root,
    'first-order randomness cancels — curvature is what remains');

  const state = {
    fnIdx: 0,
    x0: 1.4,
    h: 1.0,
  };

  let renderCtx = null, cssW = 0, cssH = 0;
  makeResponsive(canvas, 1.7, (ctx, w, h) => { renderCtx = ctx; cssW = w; cssH = h; });

  const presets = presetRow({
    labels: FUNCTIONS.map(f => f.name),
    active: 0,
    onChange: (i) => { state.fnIdx = i; },
  });
  const x0Slider = slider({
    label: 'particle position x₀',
    min: -4, max: 4, step: 0.05, value: 1.4,
    format: (v) => v.toFixed(2),
    onChange: (v) => { state.x0 = v; },
  });
  const hSlider = slider({
    label: 'step size h',
    min: 0.05, max: 2.0, step: 0.01, value: 1.0,
    format: (v) => v.toFixed(2),
    onChange: (v) => { state.h = v; },
  });
  addControls(controlsRow, x0Slider, hSlider);
  root.appendChild(presets.element);

  const ro = readout([
    { key: 'avg',  label: '½ (f(x₀+h)+f(x₀-h))',     format: v => v.toFixed(4) },
    { key: 'f0',   label: 'f(x₀)',                    format: v => v.toFixed(4) },
    { key: 'gap',  label: 'Δ (average − f(x₀))',      format: v => v.toFixed(4) },
    { key: 'pred', label: '½ h² f″(x₀)  predicted',   format: v => v.toFixed(4) },
  ]);
  root.appendChild(ro.element);

  setCaption(captionEl,
    `Pick a function; drag the particle's position x₀ and the step size h.  ` +
    `The blue dots are where the particle ends up after a ±h step.  The orange ` +
    `midpoint is their average f-value.  Compare it to f(x₀) (white).  The gap ` +
    `is almost exactly ½ h² f″(x₀) — pure curvature, no slope.  Drag x₀ to a flat ` +
    `region: the gap stays at zero <em>regardless of how steep the function is</em>.`);

  makeVisibilityLoop(root, () => {
    if (!renderCtx) return;
    const ctx = renderCtx;
    clearCanvas(ctx, cssW, cssH, '#0a0e1a');

    const fn = FUNCTIONS[state.fnIdx];
    const world = { xMin: -4.2, xMax: 4.2, yMin: -2.2, yMax: 2.6 };
    const screen = { x: 60, y: 18, width: cssW - 76, height: cssH - 60 };
    const T = makeTransform(screen, world);
    drawAxes(ctx, T, world);
    drawAxisLabels(ctx, T, world, screen, {
      xLabel: 'particle position  x',
      yLabel: 'value at that position  f(x)',
      xTicks: [{ at: -2, label: '−2' }, { at: 2, label: '2' }],
      yTicks: [{ at: -1, label: '−1' }, { at: 1, label: '1' }, { at: 2, label: '2' }],
    });

    // Draw f(x)
    const xs = linspace(world.xMin, world.xMax, 200);
    const ys = new Float64Array(xs.length);
    for (let i = 0; i < xs.length; i++) ys[i] = fn.f(xs[i]);
    drawCurve(ctx, T, xs, ys, '#5bc5f2', 2.5, 0.9);

    // Particle at x_0
    const x0 = state.x0;
    const y0 = fn.f(x0);
    const p0 = T.toScreen(x0, y0);
    const p0base = T.toScreen(x0, 0);

    // Dashed line from axis up to f(x_0)
    ctx.strokeStyle = '#9da8c2';
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.4;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(p0base.x, p0base.y);
    ctx.lineTo(p0.x, p0.y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    dot(ctx, p0.x, p0.y, 5, '#f4f6fb');
    dot(ctx, p0base.x, p0base.y, 3.5, '#f4f6fb', 0.7);

    // ±h step positions
    const h = state.h;
    const xL = x0 - h, xR = x0 + h;
    const yL = fn.f(xL), yR = fn.f(xR);
    const pL = T.toScreen(xL, yL);
    const pR = T.toScreen(xR, yR);
    dot(ctx, pL.x, pL.y, 4, '#5bc5f2');
    dot(ctx, pR.x, pR.y, 4, '#5bc5f2');

    // Step arrows along the x-axis
    const pLbase = T.toScreen(xL, 0);
    const pRbase = T.toScreen(xR, 0);
    arrow(ctx, p0base.x, p0base.y + 12, pLbase.x, p0base.y + 12, '#f5a623',
      { lineWidth: 1.5, headLen: 5 });
    arrow(ctx, p0base.x, p0base.y + 12, pRbase.x, p0base.y + 12, '#f5a623',
      { lineWidth: 1.5, headLen: 5 });
    drawLabel(ctx, '−h', (p0base.x + pLbase.x) / 2, p0base.y + 24,
      { color: '#f5a623', font: '11px "JetBrains Mono"', align: 'center' });
    drawLabel(ctx, '+h', (p0base.x + pRbase.x) / 2, p0base.y + 24,
      { color: '#f5a623', font: '11px "JetBrains Mono"', align: 'center' });

    // Chord between the two f-values; midpoint
    ctx.strokeStyle = '#f5a623';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(pL.x, pL.y);
    ctx.lineTo(pR.x, pR.y);
    ctx.stroke();
    const midY = 0.5 * (yL + yR);
    const pMid = T.toScreen(x0, midY);
    dot(ctx, pMid.x, pMid.y, 5, '#f5a623');

    // Vertical bracket showing the gap Δ
    const gap = midY - y0;
    if (Math.abs(gap) > 1e-4) {
      ctx.strokeStyle = '#f4d35e';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(p0.x + 4, p0.y);
      ctx.lineTo(pMid.x + 4, pMid.y);
      ctx.stroke();
      drawLabel(ctx, 'Δ', p0.x + 12, (p0.y + pMid.y) / 2,
        { color: '#f4d35e', font: '13px "JetBrains Mono"', align: 'left' });
    }

    // Readout
    const avg = 0.5 * (yL + yR);
    const pred = 0.5 * h * h * fn.fpp(x0);
    ro.update({ avg, f0: y0, gap, pred });

    // Sub-label
    drawLabel(ctx, fn.name, cssW - 10, 16,
      { color: '#9da8c2', font: '11px "JetBrains Mono"', align: 'right', baseline: 'top' });
  });
}

// ----------------------------------------------------------------------------
// Chapter 5 — Heat equation: live particle sim + analytic kernel overlay
// ----------------------------------------------------------------------------
export function initHeatSandbox(rootId) {
  const root = document.getElementById(rootId);
  if (!root) return;
  const { canvas, controlsRow, captionEl } = makeSandbox(root,
    'a cloud of randomly-wandering particles follows the heat equation exactly');

  const state = {
    nParticles: 2000,
    D: 0.5,
    particles: null,
    t: 0,
    initDist: 'delta',   // 'delta' | 'gauss' | 'two'
    rng: null, gauss: null,
    running: true,
    paintMode: false,
  };

  function reset() {
    state.t = 0;
    const n = state.nParticles;
    const arr = new Float64Array(n);
    state.rng = makeRng(123);
    state.gauss = makeGaussian(state.rng);
    if (state.initDist === 'delta') {
      // all at zero
    } else if (state.initDist === 'gauss') {
      for (let i = 0; i < n; i++) arr[i] = state.gauss() * 0.8;
    } else if (state.initDist === 'two') {
      for (let i = 0; i < n; i++) arr[i] = (state.rng() < 0.5 ? -2 : 2) + 0.3 * state.gauss();
    }
    state.particles = arr;
  }
  reset();

  function step(dt) {
    const sqrtDt = Math.sqrt(2 * state.D * dt);
    const p = state.particles;
    const g = state.gauss;
    for (let i = 0; i < p.length; i++) p[i] += sqrtDt * g();
    state.t += dt;
  }

  let renderCtx = null, cssW = 0, cssH = 0;
  makeResponsive(canvas, 1.7, (ctx, w, h) => { renderCtx = ctx; cssW = w; cssH = h; });

  // Controls
  const nSlider = slider({
    label: 'particles',
    min: 200, max: 8000, step: 100, value: 2000,
    format: v => v.toFixed(0),
    onChange: v => { state.nParticles = v; reset(); },
  });
  const dSlider = slider({
    label: 'diffusion D',
    min: 0.1, max: 2.0, step: 0.05, value: 0.5,
    format: v => v.toFixed(2),
    onChange: v => { state.D = v; },
  });
  const initRow = presetRow({
    labels: ['start: delta', 'start: Gaussian', 'start: two peaks'],
    active: 0,
    onChange: (i) => {
      state.initDist = ['delta', 'gauss', 'two'][i];
      reset();
    },
  });
  const resetBtn = actionButton('reset', reset);
  const pauseBtn = actionButton('pause / resume',
    () => { state.running = !state.running; });
  addControls(controlsRow, nSlider, dSlider, resetBtn, pauseBtn);
  root.appendChild(initRow.element);

  setCaption(captionEl,
    `Each particle takes its own Brownian step.  The blue histogram is their ` +
    `live density.  The gold curve is the analytic heat-equation prediction ` +
    `p(x,t) = (4πDt)<sup>−½</sup> e<sup>−x²/(4Dt)</sup>.  When the initial ` +
    `state is more complex (two peaks), both bumps spread and merge — and the ` +
    `equation predicts that too, by linear superposition.`);

  makeVisibilityLoop(root, () => {
    if (!renderCtx) return;
    if (state.running) step(0.012);
    const ctx = renderCtx;
    clearCanvas(ctx, cssW, cssH, '#0a0e1a');

    // World coords
    const maxX = 6;
    const world = { xMin: -maxX, xMax: maxX, yMin: -0.05, yMax: 0.55 };
    const screen = { x: 60, y: 18, width: cssW - 76, height: cssH - 60 };
    const T = makeTransform(screen, world);
    drawAxes(ctx, T, world);
    drawAxisLabels(ctx, T, world, screen, {
      xLabel: 'particle position  x',
      yLabel: 'density  p(x, t)',
      xTicks: [{ at: -4, label: '−4' }, { at: -2, label: '−2' },
               { at: 2,  label: '2'  }, { at: 4,  label: '4' }],
      yTicks: [{ at: 0.2, label: '0.2' }, { at: 0.4, label: '0.4' }],
    });

    // Particle density histogram
    const nBins = 80;
    const hist = histogram(state.particles, -maxX, maxX, nBins);
    drawHistogram(ctx, T, hist.edges, hist.density, '#5bc5f2', 0.5, 1);

    // Analytic prediction
    const xs = linspace(-maxX, maxX, 250);
    const ys = new Float64Array(xs.length);
    if (state.initDist === 'delta') {
      for (let i = 0; i < xs.length; i++) ys[i] = heatKernel(xs[i], state.t, state.D);
    } else if (state.initDist === 'gauss') {
      // Initial variance = 0.8² = 0.64; at time t, variance = 0.64 + 2 D t
      const var0 = 0.64;
      for (let i = 0; i < xs.length; i++) ys[i] = gaussianPdf(xs[i], 0, var0 + 2 * state.D * state.t);
    } else if (state.initDist === 'two') {
      // Two Gaussians initially at ±2 with var=0.09 each
      const var0 = 0.09;
      const v = var0 + 2 * state.D * state.t;
      for (let i = 0; i < xs.length; i++) {
        ys[i] = 0.5 * (gaussianPdf(xs[i], -2, v) + gaussianPdf(xs[i], 2, v));
      }
    }
    drawCurve(ctx, T, xs, ys, '#f4d35e', 2.5, 0.95);

    // t label
    drawLabel(ctx, `t = ${state.t.toFixed(2)}`,
      cssW - 10, 16, { color: '#f4d35e', font: '13px "JetBrains Mono"',
                       align: 'right', baseline: 'top' });
    drawLabel(ctx, `D = ${state.D.toFixed(2)}, N = ${state.particles.length}`,
      cssW - 10, 32, { color: '#9da8c2', font: '11px "JetBrains Mono"',
                       align: 'right', baseline: 'top' });
  });
}
