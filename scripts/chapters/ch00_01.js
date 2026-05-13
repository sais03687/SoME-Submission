// ============================================================================
// Chapter 0 — Hook.  A jittering speck plus a "spawn more" button.
// Chapter 1 — Random walks: live histogram + Gaussian + variance bands.
// ============================================================================

import { makeRng, makeGaussian, gaussianPdf, linspace, histogram, rademacher } from '../core/math.js';
import { setupCanvas, makeTransform, drawAxes, drawCurve, drawHistogram,
         clearCanvas, dot, drawLabel, drawAxisLabels, makeResponsive } from '../core/canvas.js';
import { slider, presetRow, actionButton, readout,
         makeSandbox, addControls, setCaption, makeVisibilityLoop } from '../core/ui.js';

// ----------------------------------------------------------------------------
// Chapter 0 — Hook: the speck and the cloud
// ----------------------------------------------------------------------------
export function initHookSandbox(rootId) {
  const root = document.getElementById(rootId);
  if (!root) return;
  const { canvas, controlsRow, captionEl } = makeSandbox(root,
    'one hiker — many hikers');

  // State
  const state = {
    hikers: [],      // {x, y, vx, vy, trail: []}
    nTarget: 1,
    rng: makeRng(7),
    gauss: null,
  };
  state.gauss = makeGaussian(state.rng);

  function spawnTo(n) {
    while (state.hikers.length < n) {
      state.hikers.push({
        x: 0, y: 0,
        trail: [],
      });
    }
    if (state.hikers.length > n) state.hikers.length = n;
  }

  function step(dt) {
    const stepScale = Math.sqrt(dt) * 0.4;
    for (let i = 0; i < state.hikers.length; i++) {
      const h = state.hikers[i];
      h.x += stepScale * state.gauss();
      h.y += stepScale * state.gauss();
      if (state.hikers.length <= 30) {
        h.trail.push({ x: h.x, y: h.y });
        if (h.trail.length > 80) h.trail.shift();
      }
    }
  }

  let renderCtx = null;
  let cssW = 0, cssH = 0;

  makeResponsive(canvas, 1.6, (ctx, w, h) => {
    renderCtx = ctx;
    cssW = w; cssH = h;
  });

  // Controls
  const nSlider = slider({
    label: 'hikers',
    min: 1, max: 1000, step: 1, value: 1,
    format: (v) => v.toFixed(0),
    onChange: (v) => { state.nTarget = v; spawnTo(v); },
  });
  const resetBtn = actionButton('reset', () => {
    state.hikers = [];
    spawnTo(state.nTarget);
  });
  addControls(controlsRow, nSlider, resetBtn);

  setCaption(captionEl,
    `Start with one hiker.  Watch them wander.  Slide the count up — at 10, ` +
    `20, 100, 1000 — and notice the cloud taking shape.  Individual paths are ` +
    `unpredictable; the cloud is not.`);

  spawnTo(1);

  makeVisibilityLoop(root, () => {
    if (!renderCtx) return;
    step(1/60);

    const ctx = renderCtx;
    clearCanvas(ctx, cssW, cssH, '#0a0e1a');

    const world = { xMin: -4, xMax: 4, yMin: -2.5, yMax: 2.5 };
    const screen = { x: 0, y: 0, width: cssW, height: cssH };
    const T = makeTransform(screen, world);
    drawAxes(ctx, T, world);
    // Overlay-style labels — the cloud takes the whole canvas
    drawLabel(ctx, 'horizontal position →', cssW - 8, cssH - 6,
      { color: '#9da8c2', font: '11px "JetBrains Mono"', align: 'right', baseline: 'bottom' });
    drawLabel(ctx, '↑ vertical position', 8, 8,
      { color: '#9da8c2', font: '11px "JetBrains Mono"', align: 'left', baseline: 'top' });

    // Trails (only when few)
    if (state.hikers.length <= 30) {
      for (const h of state.hikers) {
        if (h.trail.length < 2) continue;
        ctx.strokeStyle = '#9da8c2';
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.35;
        ctx.beginPath();
        const p0 = T.toScreen(h.trail[0].x, h.trail[0].y);
        ctx.moveTo(p0.x, p0.y);
        for (let k = 1; k < h.trail.length; k++) {
          const p = T.toScreen(h.trail[k].x, h.trail[k].y);
          ctx.lineTo(p.x, p.y);
        }
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    // Hikers
    const r = state.hikers.length > 200 ? 1.5 : state.hikers.length > 50 ? 2 : 3;
    const alpha = state.hikers.length > 200 ? 0.4 : state.hikers.length > 50 ? 0.6 : 0.9;
    for (const h of state.hikers) {
      const p = T.toScreen(h.x, h.y);
      dot(ctx, p.x, p.y, r, '#f4f6fb', alpha);
    }
  });
}

// ----------------------------------------------------------------------------
// Chapter 1 — Random walk: live histogram emerging from many walkers
// ----------------------------------------------------------------------------
export function initRandomWalkSandbox(rootId) {
  const root = document.getElementById(rootId);
  if (!root) return;
  const { canvas, controlsRow, captionEl } = makeSandbox(root,
    'the cloud has a shape — and that shape is predictable');

  const state = {
    nWalkers: 200,
    nSteps: 80,
    rng: makeRng(42),
    walkers: [],   // each walker is {pos, steps_done}
    stepIdx: 0,
  };

  function reset() {
    state.walkers = [];
    for (let i = 0; i < state.nWalkers; i++) {
      state.walkers.push({ pos: 0 });
    }
    state.stepIdx = 0;
  }
  reset();

  function step() {
    if (state.stepIdx >= state.nSteps) return;
    for (let i = 0; i < state.walkers.length; i++) {
      state.walkers[i].pos += rademacher(state.rng);
    }
    state.stepIdx++;
  }

  let renderCtx = null, cssW = 0, cssH = 0;
  makeResponsive(canvas, 2.0, (ctx, w, h) => { renderCtx = ctx; cssW = w; cssH = h; });

  // Controls
  const nSlider = slider({
    label: 'walkers (N)',
    min: 1, max: 5000, step: 1, value: 200,
    format: (v) => v.toFixed(0),
    onChange: (v) => { state.nWalkers = v; reset(); },
  });
  const stepsSlider = slider({
    label: 'steps (n)',
    min: 1, max: 200, step: 1, value: 80,
    format: (v) => v.toFixed(0),
    onChange: (v) => { state.nSteps = v; reset(); },
  });
  const showGaussian = { value: true };
  const gaussBtn = actionButton('toggle Gaussian overlay', () => {
    showGaussian.value = !showGaussian.value;
  });
  const resetBtn = actionButton('reset', reset);
  addControls(controlsRow, nSlider, stepsSlider, gaussBtn, resetBtn);

  const ro = readout([
    { key: 'mean', label: 'mean', format: (v) => v.toFixed(2) },
    { key: 'variance', label: 'variance', format: (v) => v.toFixed(2) },
    { key: 'predict', label: 'predicted variance (=n)', format: (v) => v.toFixed(2) },
  ]);
  root.appendChild(ro.element);

  setCaption(captionEl,
    `Each walker takes one step left or right per tick.  After enough steps ` +
    `the cloud of final positions looks like a bell curve.  Compare the ` +
    `<em>measured</em> variance against the prediction <em>n</em> — they ` +
    `match because variances add for independent ±1 steps.`);

  let frame = 0;

  makeVisibilityLoop(root, () => {
    if (!renderCtx) return;
    // Slow down: one step every 6 frames so people can watch the histogram fill
    frame++;
    if (frame % 4 === 0) step();
    if (state.stepIdx >= state.nSteps) {
      // auto-restart after a short pause
      if (frame % 240 === 0) reset();
    }

    const ctx = renderCtx;
    clearCanvas(ctx, cssW, cssH, '#0a0e1a');

    const maxRange = Math.max(10, Math.sqrt(state.nSteps) * 3.2);
    const world = { xMin: -maxRange, xMax: maxRange, yMin: -0.05, yMax: 0.5 };
    // Plot rectangle with margins for axis labels (left for y-title, bottom for x-title)
    const screen = { x: 64, y: 18, width: cssW - 78, height: cssH - 60 };
    const T = makeTransform(screen, world);
    drawAxes(ctx, T, world);
    drawAxisLabels(ctx, T, world, screen, {
      xLabel: 'final position  X_n  (east-steps minus west-steps)',
      yLabel: 'fraction of walkers (density)',
      xTicks: [
        { at: -Math.floor(maxRange / 2), label: String(-Math.floor(maxRange / 2)) },
        { at: 0, label: '0' },
        { at:  Math.floor(maxRange / 2), label: String( Math.floor(maxRange / 2)) },
      ],
      yTicks: [
        { at: 0, label: '0' },
        { at: 0.2, label: '0.2' },
        { at: 0.4, label: '0.4' },
      ],
    });

    // Histogram
    const positions = new Float64Array(state.walkers.length);
    for (let i = 0; i < state.walkers.length; i++) positions[i] = state.walkers[i].pos;
    const nBins = 50;
    const binW = (2 * maxRange) / nBins;
    const hist = histogram(positions, -maxRange, maxRange, nBins);
    drawHistogram(ctx, T, hist.edges, hist.density, '#5bc5f2', 0.55, 1);

    // Gaussian overlay (predicted)
    if (showGaussian.value && state.stepIdx > 0) {
      const variance = state.stepIdx;  // theoretical variance for symmetric ±1 walk
      const xs = linspace(-maxRange, maxRange, 200);
      const ys = new Float64Array(xs.length);
      for (let i = 0; i < xs.length; i++) ys[i] = gaussianPdf(xs[i], 0, variance);
      drawCurve(ctx, T, xs, ys, '#f4d35e', 2.5, 0.9);

      // Variance bands: ±√n
      const sd = Math.sqrt(variance);
      for (const k of [1, 2]) {
        for (const sign of [-1, 1]) {
          const xpos = sign * k * sd;
          const top = T.toScreen(xpos, 0.45);
          const bot = T.toScreen(xpos, 0);
          ctx.strokeStyle = '#9da8c2';
          ctx.lineWidth = 1;
          ctx.globalAlpha = 0.4;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(top.x, top.y);
          ctx.lineTo(bot.x, bot.y);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.globalAlpha = 1;
        }
      }
      // Label the ±√n band
      const labP = T.toScreen(sd, 0.46);
      drawLabel(ctx, '√n', labP.x, labP.y,
        { color: '#9da8c2', font: '11px "JetBrains Mono"', align: 'center' });
    }

    // Stats readout
    let mean = 0, vsum = 0;
    for (let i = 0; i < positions.length; i++) mean += positions[i];
    mean /= positions.length;
    for (let i = 0; i < positions.length; i++) vsum += (positions[i] - mean) ** 2;
    const variance = vsum / positions.length;
    ro.update({ mean, variance, predict: state.stepIdx });

    // Progress indicator (small)
    drawLabel(ctx, `step ${state.stepIdx}/${state.nSteps}`, cssW - 10, 16,
      { color: '#6b7280', font: '11px "JetBrains Mono"', align: 'right', baseline: 'top' });
  });
}
