// ============================================================================
// Chapter 2 — Scaling limit.  Slider for scaling exponent α.  The reader
// adjusts how step-size depends on step-count and watches three regimes.
// Chapter 3 — Brownian roughness.  Click-to-zoom into a precomputed path.
// ============================================================================

import { makeRng, makeGaussian, rademacher, linspace } from '../core/math.js';
import { setupCanvas, makeTransform, drawAxes, drawCurve,
         clearCanvas, dot, drawLabel, drawAxisLabels, makeResponsive } from '../core/canvas.js';
import { slider, presetRow, actionButton, readout,
         makeSandbox, addControls, setCaption, makeVisibilityLoop } from '../core/ui.js';

// Round step to one of {1, 2, 5} × 10^k for clean tick labels
function niceStep(rawStep) {
  if (rawStep <= 0) return 1;
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const norm = rawStep / mag;
  if (norm < 1.5) return mag;
  if (norm < 3.5) return 2 * mag;
  if (norm < 7.5) return 5 * mag;
  return 10 * mag;
}

// ----------------------------------------------------------------------------
// Chapter 2 — Scaling limit
// ----------------------------------------------------------------------------
export function initScalingLimitSandbox(rootId) {
  const root = document.getElementById(rootId);
  if (!root) return;
  const { canvas, controlsRow, captionEl } = makeSandbox(root,
    'which scaling survives?');

  const state = {
    alpha: 0.5,
    nSteps: 1000,
    seed: 314,
    // Pre-generate a fixed base ±1 sequence; rescaling visualises different
    // limits of the SAME underlying walk.
    baseSteps: null,
  };

  function regenerateBase() {
    const rng = makeRng(state.seed);
    const N = 65536;
    const s = new Int8Array(N);
    for (let i = 0; i < N; i++) s[i] = rademacher(rng);
    state.baseSteps = s;
  }
  regenerateBase();

  function rescaledPath(alpha, nSteps) {
    const N = state.baseSteps.length;
    const stride = Math.max(1, Math.floor(N / nSteps));
    const cum = new Float64Array(nSteps + 1);
    cum[0] = 0;
    let total = 0;
    for (let i = 0; i < nSteps; i++) {
      total += state.baseSteps[i * stride];
      cum[i + 1] = total;
    }
    const T = 1.0;
    const dt = T / nSteps;
    // Position at step k is rescaled cumulative: (dt^α) * sum_of_steps
    const scale = Math.pow(dt, alpha);
    for (let i = 0; i <= nSteps; i++) cum[i] *= scale;
    return cum;
  }

  let renderCtx = null, cssW = 0, cssH = 0;
  makeResponsive(canvas, 1.9, (ctx, w, h) => { renderCtx = ctx; cssW = w; cssH = h; });

  // Controls
  const alphaSlider = slider({
    label: 'scaling exponent α',
    min: 0, max: 1, step: 0.02, value: 0.5,
    format: (v) => v.toFixed(2),
    onChange: (v) => { state.alpha = v; },
  });
  const stepsSlider = slider({
    label: 'steps (n)',
    min: 100, max: 8000, step: 100, value: 1000,
    format: (v) => v.toFixed(0),
    onChange: (v) => { state.nSteps = v; },
  });
  const presets = presetRow({
    labels: ['α = 1.0 (vanishes)', 'α = 0.5 (Brownian)', 'α = 0.0 (blows up)'],
    active: 1,
    onChange: (i) => {
      const vals = [1.0, 0.5, 0.0];
      state.alpha = vals[i];
      alphaSlider.set(vals[i]);
    },
  });
  addControls(controlsRow, alphaSlider, stepsSlider, presets);

  setCaption(captionEl,
    `Drag the exponent α and watch what happens to the path.  Each step ` +
    `is rescaled by Δt<sup>α</sup>.  Three regimes: too small a scaling ` +
    `(α near 1) and the path flattens.  Too large (α near 0) and it ` +
    `explodes.  Exactly one value in between — α = ½ — gives a path that ` +
    `has the same character at every level of refinement.  That's Brownian motion.`);

  makeVisibilityLoop(root, () => {
    if (!renderCtx) return;
    const ctx = renderCtx;
    clearCanvas(ctx, cssW, cssH, '#0a0e1a');

    const path = rescaledPath(state.alpha, state.nSteps);

    // Auto-determine y-range from the data, but clamp so we always see something
    let yLo = -1, yHi = 1;
    for (let i = 0; i < path.length; i++) {
      if (path[i] < yLo) yLo = path[i];
      if (path[i] > yHi) yHi = path[i];
    }
    const pad = Math.max(0.3, (yHi - yLo) * 0.15);
    yLo -= pad; yHi += pad;

    const world = { xMin: 0, xMax: 1, yMin: yLo, yMax: yHi };
    const screen = { x: 60, y: 18, width: cssW - 76, height: cssH - 60 };
    const T = makeTransform(screen, world);
    drawAxes(ctx, T, world);
    // Pick 3 round-ish y-ticks that span the current y range
    const yMid = (yLo + yHi) / 2;
    const yTickStep = niceStep((yHi - yLo) / 2);
    const yTickVals = [
      Math.round(yMid / yTickStep) * yTickStep - yTickStep,
      Math.round(yMid / yTickStep) * yTickStep,
      Math.round(yMid / yTickStep) * yTickStep + yTickStep,
    ].filter(v => v >= yLo && v <= yHi);
    drawAxisLabels(ctx, T, world, screen, {
      xLabel: 'time  t  ∈ [0, 1]',
      yLabel: `rescaled position  W(t) = Δt^${state.alpha.toFixed(2)} · (sum of ±1 steps)`,
      xTicks: [{ at: 0, label: '0' }, { at: 0.5, label: '0.5' }, { at: 1, label: '1' }],
      yTicks: yTickVals.map(v => ({ at: v, label: v.toFixed(1) })),
    });

    // Color depends on which regime we're in
    let color = '#f4f6fb';
    if (state.alpha > 0.65) color = '#e07856';
    else if (state.alpha < 0.35) color = '#e07856';
    else if (Math.abs(state.alpha - 0.5) < 0.07) color = '#f4d35e';

    // Draw the path
    const xs = linspace(0, 1, path.length);
    drawCurve(ctx, T, xs, path, color, 1.6, 0.92);

    // Label diagnostics
    drawLabel(ctx, `α = ${state.alpha.toFixed(2)}`,
      cssW - 10, 16, { color, font: '13px "JetBrains Mono"', align: 'right', baseline: 'top' });
    drawLabel(ctx, `displacement at t=1:  ${path[path.length-1].toFixed(3)}`,
      cssW - 10, 32, { color: '#9da8c2', font: '11px "JetBrains Mono"',
                       align: 'right', baseline: 'top' });

    // Verdict
    let verdict;
    if (Math.abs(state.alpha - 0.5) < 0.05) verdict = 'survives  ✓';
    else if (state.alpha > 0.5) verdict = 'shrinks  →  flat';
    else verdict = 'explodes  →  diverges';
    drawLabel(ctx, verdict,
      cssW - 10, 50, { color, font: '12px "JetBrains Mono"', align: 'right', baseline: 'top' });
  });
}

// ----------------------------------------------------------------------------
// Chapter 3 — Brownian roughness.  Repeated click-to-zoom.
// ----------------------------------------------------------------------------
export function initRoughnessSandbox(rootId) {
  const root = document.getElementById(rootId);
  if (!root) return;
  const { canvas, controlsRow, captionEl } = makeSandbox(root,
    'zoom in.  it never gets any smoother.');

  // Pre-compute a single very-high-resolution Brownian path
  const N_FINE = 200000;
  const T_END = 1.0;
  const dt = T_END / N_FINE;
  const fullPath = new Float64Array(N_FINE + 1);
  {
    const rng = makeRng(2718);
    const g = makeGaussian(rng);
    let x = 0;
    fullPath[0] = 0;
    const stepScale = Math.sqrt(dt);
    for (let i = 1; i <= N_FINE; i++) {
      x += stepScale * g();
      fullPath[i] = x;
    }
  }

  // View state: window into the path
  const state = {
    tCentre: 0.5,
    halfWidth: 0.5,   // initially see the whole path [0, 1]
    showTangent: false,
    tangentH: 0.05,
  };

  let renderCtx = null, cssW = 0, cssH = 0;
  makeResponsive(canvas, 1.7, (ctx, w, h) => { renderCtx = ctx; cssW = w; cssH = h; });

  // Build controls
  const zoomBtn = actionButton('zoom in 5×', () => {
    state.halfWidth = Math.max(0.000001, state.halfWidth / 5);
  });
  const zoomOutBtn = actionButton('zoom out 5×', () => {
    state.halfWidth = Math.min(0.5, state.halfWidth * 5);
  });
  const resetBtn = actionButton('reset', () => {
    state.tCentre = 0.5;
    state.halfWidth = 0.5;
  });
  const tangentBtn = actionButton('toggle tangent estimates', () => {
    state.showTangent = !state.showTangent;
  });
  const tangentSlider = slider({
    label: 'tangent window h',
    min: 0.0001, max: 0.5, step: 0.0001, value: 0.05,
    format: (v) => v < 0.001 ? v.toExponential(1) : v.toFixed(3),
    onChange: (v) => { state.tangentH = v; },
  });
  addControls(controlsRow, zoomBtn, zoomOutBtn, resetBtn, tangentBtn);
  controlsRow.appendChild(tangentSlider.element);

  // Click to recentre
  canvas.addEventListener('click', (ev) => {
    const r = canvas.getBoundingClientRect();
    const px = ev.clientX - r.left;
    const screenLeft = 40, screenRight = cssW - 20;
    if (px < screenLeft || px > screenRight) return;
    const fracX = (px - screenLeft) / (screenRight - screenLeft);
    const tLo = state.tCentre - state.halfWidth;
    const tHi = state.tCentre + state.halfWidth;
    state.tCentre = tLo + fracX * (tHi - tLo);
    // Also zoom in 3x on click
    state.halfWidth = Math.max(0.000001, state.halfWidth / 3);
  });

  setCaption(captionEl,
    `Click anywhere on the path to zoom in 3×.  The path stays jagged at every ` +
    `scale.  Toggle "tangent estimates" to see secant lines drawn over a window ` +
    `of width h on either side of the centre.  Drag h smaller — the secant ` +
    `direction <em>refuses to settle</em>.  That's why Brownian motion has no ` +
    `ordinary derivative.`);

  makeVisibilityLoop(root, () => {
    if (!renderCtx) return;
    const ctx = renderCtx;
    clearCanvas(ctx, cssW, cssH, '#0a0e1a');

    const tLo = Math.max(0, state.tCentre - state.halfWidth);
    const tHi = Math.min(1, state.tCentre + state.halfWidth);
    const iLo = Math.floor(tLo * N_FINE);
    const iHi = Math.min(N_FINE, Math.ceil(tHi * N_FINE));

    // Find y-range in window
    let yLo = Infinity, yHi = -Infinity;
    for (let i = iLo; i <= iHi; i++) {
      const v = fullPath[i];
      if (v < yLo) yLo = v;
      if (v > yHi) yHi = v;
    }
    const pad = Math.max(0.05, (yHi - yLo) * 0.12);
    yLo -= pad; yHi += pad;

    const world = { xMin: tLo, xMax: tHi, yMin: yLo, yMax: yHi };
    const screen = { x: 60, y: 18, width: cssW - 76, height: cssH - 60 };
    const T = makeTransform(screen, world);
    drawAxes(ctx, T, world);

    // Dynamic ticks adjusting to the current zoom level
    const xStep = niceStep((tHi - tLo) / 4);
    const xFirst = Math.ceil(tLo / xStep) * xStep;
    const xTickVals = [];
    for (let v = xFirst; v <= tHi + 1e-9; v += xStep) xTickVals.push(v);

    const yStep = niceStep((yHi - yLo) / 3);
    const yMid = (yLo + yHi) / 2;
    const yTickVals = [
      Math.round(yMid / yStep) * yStep - yStep,
      Math.round(yMid / yStep) * yStep,
      Math.round(yMid / yStep) * yStep + yStep,
    ].filter(v => v >= yLo && v <= yHi);

    // Format ticks: when zoomed in tight, use enough decimals to distinguish them
    const xDec = xStep < 0.01 ? 4 : xStep < 0.1 ? 3 : xStep < 1 ? 2 : 1;
    const yDec = yStep < 0.1 ? 2 : 1;

    drawAxisLabels(ctx, T, world, screen, {
      xLabel: 'time  t',
      yLabel: 'Brownian path  W(t)',
      xTicks: xTickVals.map(v => ({ at: v, label: v.toFixed(xDec) })),
      yTicks: yTickVals.map(v => ({ at: v, label: v.toFixed(yDec) })),
    });

    // Decimate the path data for drawing efficiency
    const segCount = Math.min(iHi - iLo, 2000);
    const stride = Math.max(1, Math.floor((iHi - iLo) / segCount));
    ctx.strokeStyle = '#f4f6fb';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    const p0 = T.toScreen(tLo, fullPath[iLo]);
    ctx.moveTo(p0.x, p0.y);
    for (let i = iLo + stride; i <= iHi; i += stride) {
      const p = T.toScreen(i / N_FINE, fullPath[i]);
      ctx.lineTo(p.x, p.y);
    }
    ctx.stroke();

    // Centre indicator
    const tCentreI = Math.floor(state.tCentre * N_FINE);
    const centreVal = fullPath[Math.min(tCentreI, N_FINE)];
    const cp = T.toScreen(state.tCentre, centreVal);
    dot(ctx, cp.x, cp.y, 4, '#f4d35e');

    // Tangent estimates: draw secants for three values of h around tangentH
    if (state.showTangent) {
      const hVals = [state.tangentH, state.tangentH / 4, state.tangentH / 20];
      const colors = ['#e07856', '#f5a623', '#f4d35e'];
      for (let k = 0; k < hVals.length; k++) {
        const h = hVals[k];
        const t1 = Math.max(0, state.tCentre - h);
        const t2 = Math.min(1, state.tCentre + h);
        const i1 = Math.floor(t1 * N_FINE);
        const i2 = Math.min(N_FINE, Math.floor(t2 * N_FINE));
        const v1 = fullPath[i1];
        const v2 = fullPath[i2];
        const slope = (v2 - v1) / (t2 - t1);
        // Extend the line from t1 to t2
        const lineLeft = T.toScreen(t1, centreVal + slope * (t1 - state.tCentre));
        const lineRight = T.toScreen(t2, centreVal + slope * (t2 - state.tCentre));
        ctx.strokeStyle = colors[k];
        ctx.lineWidth = 1.6;
        ctx.globalAlpha = 0.85;
        ctx.beginPath();
        ctx.moveTo(lineLeft.x, lineLeft.y);
        ctx.lineTo(lineRight.x, lineRight.y);
        ctx.stroke();
        ctx.globalAlpha = 1;
        drawLabel(ctx,
          `h=${h < 0.001 ? h.toExponential(1) : h.toFixed(3)}: slope ${slope.toFixed(1)}`,
          12, 32 + k * 16, { color: colors[k], align: 'left', baseline: 'top' });
      }
    }

    drawLabel(ctx,
      `view: [${tLo.toFixed(6)}, ${tHi.toFixed(6)}]  (width ${(2*state.halfWidth).toExponential(1)})`,
      cssW - 10, 16,
      { color: '#9da8c2', font: '11px "JetBrains Mono"', align: 'right', baseline: 'top' });
  });
}
