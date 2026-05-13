// ============================================================================
// Chapter 8 — Optimization.  GD, SGD, and Langevin race down a loss landscape.
// Chapter 9 — Score field.  Real trained denoising network on two-moons.
// ============================================================================

import { makeRng, makeGaussian, clamp, linspace } from '../core/math.js';
import { setupCanvas, makeTransform, drawAxes, clearCanvas, dot, arrow,
         drawLabel, drawAxisLabels, makeResponsive } from '../core/canvas.js';
import { slider, presetRow, actionButton, toggleButton, readout,
         makeSandbox, addControls, setCaption, makeVisibilityLoop } from '../core/ui.js';
import { loadScoreModel, reverseSdeStep } from '../core/score_net.js';

// ----------------------------------------------------------------------------
// Chapter 8 — Optimization race
// ----------------------------------------------------------------------------
// Loss landscape: asymmetric two-well-with-narrow-sharp basin.
// L(x, y) = 0.5*(x-1)^2 + 0.4*y^2  preferred wide global min at (1, 0)
//        + 4 * exp(-((x+1.2)^2 + y^2) / 0.05)  narrow sharp min around (-1.2, 0) — local
// The narrow basin has steep walls (easy for GD to fall into and get stuck).
// Asymmetric topology: narrow-and-deep on the left, wide-and-shallow on the
// right.  Walkers start above on the shoulder so they have to commit to one.
function losses(landscape) {
  if (landscape === 'sharp_vs_wide') {
    return {
      // Three components:
      //   bowl    -- a gentle ambient parabola that gives the whole landscape
      //              a "downhill" feel toward y=0
      //   narrow  -- a narrow Gaussian well at (-1.1, 0).  Width 0.18 in
      //              x, depth 1.9.  Steep walls → GD's gradient near the
      //              surface points hard into this well, so it falls in
      //              and parks.  But this is the WORSE (less deep) of the
      //              two minima — the local trap.
      //   wide    -- a broader Gaussian well at (1.1, 0).  Width 0.70,
      //              depth 2.6.  The TRUE global minimum: deeper but with
      //              a shallower approach gradient that GD doesn't follow
      //              from the starting point.
      //
      // The narrow well's steeper walls are what fool plain GD: from the
      // starting shoulder, the immediate gradient points strongly into the
      // narrow well, even though the wide well would ultimately be deeper.
      // Langevin's noise lets it escape the narrow trap and find the deeper
      // wide basin — the entire pedagogical point.
      L: (x, y) => 0.15 * (y * y) + 0.05 * (x * x)
                   - 1.9 * Math.exp(-(((x + 1.1) ** 2) / 0.18 + (y * y) / 0.18))
                   - 2.6 * Math.exp(-(((x - 1.1) ** 2) / 0.70 + (y * y) / 0.40)),
      grad: (x, y) => {
        const eN = Math.exp(-(((x + 1.1) ** 2) / 0.18 + (y * y) / 0.18));
        const eW = Math.exp(-(((x - 1.1) ** 2) / 0.70 + (y * y) / 0.40));
        return {
          x: 0.10 * x
             + 1.9 * eN * (2 * (x + 1.1) / 0.18)
             + 2.6 * eW * (2 * (x - 1.1) / 0.70),
          y: 0.30 * y
             + 1.9 * eN * (2 * y / 0.18)
             + 2.6 * eW * (2 * y / 0.40),
        };
      },
      range: { xMin: -2.4, xMax: 2.4, yMin: -1.5, yMax: 1.5 },
      mins: [{ x: 1.1, y: 0, label: 'wide global min' },
             { x: -1.1, y: 0, label: 'narrow local min' }],
      // Walkers start above, midway between the two basins, so they have to
      // pick one when descending.  GD goes deterministically into whichever
      // wins the gradient race; SGD wobbles slightly; Langevin can escape.
      start: { x: -0.55, y: 1.15 },
    };
  }
  return {
    L: (x, y) => 0.5 * x * x + 0.4 * y * y,
    grad: (x, y) => ({ x: x, y: 0.8 * y }),
    range: { xMin: -2.5, xMax: 2.5, yMin: -1.5, yMax: 1.5 },
    mins: [{ x: 0, y: 0, label: 'global min' }],
    start: { x: -1.7, y: 1.1 },
  };
}

export function initOptimizationSandbox(rootId) {
  const root = document.getElementById(rootId);
  if (!root) return;
  const { canvas, controlsRow, captionEl } = makeSandbox(root,
    'three walkers descend a loss landscape — one of them escapes');

  const state = {
    lr: 0.04,                 // smaller learning rate -> sees the topology better
    sgdNoise: 0.6,            // visible wobble but not chaotic
    langevinTemp: 0.35,       // calibrated to escape the narrow basin in a few seconds
    runners: [],
    rng: null, gauss: null,
    running: true,
    landscape: 'sharp_vs_wide',
  };

  function reset() {
    state.rng = makeRng(11);
    state.gauss = makeGaussian(state.rng);
    const L = losses(state.landscape);
    const start = L.start;
    state.runners = [
      { kind: 'GD',   pos: { ...start }, color: '#3ddc97', label: 'GD',       trail: [], lossHistory: [] },
      { kind: 'SGD',  pos: { ...start }, color: '#f5a623', label: 'SGD',      trail: [], lossHistory: [] },
      { kind: 'LANG', pos: { ...start }, color: '#b084eb', label: 'Langevin', trail: [], lossHistory: [] },
    ];
    state.stepCount = 0;
  }
  reset();

  // Maximum number of steps kept in the loss-vs-time plot.  When we hit this,
  // we either auto-reset or scroll.  We auto-reset because a fresh demo is
  // friendlier than a partially-replaced one — the reader can scroll past or
  // pause if they want to stop the loop.
  const HISTORY_CAP = 700;

  // Fraction of the canvas height devoted to the landscape panel; the rest
  // (minus a small gutter) is the loss-vs-time plot.
  const LANDSCAPE_FRAC = 0.68;

  function step() {
    const L = losses(state.landscape);
    for (const r of state.runners) {
      const g = L.grad(r.pos.x, r.pos.y);
      let dx = -state.lr * g.x;
      let dy = -state.lr * g.y;
      if (r.kind === 'SGD') {
        // approximation: gradient with extra zero-mean noise
        dx += state.sgdNoise * state.lr * state.gauss();
        dy += state.sgdNoise * state.lr * state.gauss();
      } else if (r.kind === 'LANG') {
        const noiseScale = Math.sqrt(2 * state.lr * state.langevinTemp);
        dx += noiseScale * state.gauss();
        dy += noiseScale * state.gauss();
      }
      r.pos.x += dx;
      r.pos.y += dy;
      r.pos.x = clamp(r.pos.x, L.range.xMin, L.range.xMax);
      r.pos.y = clamp(r.pos.y, L.range.yMin, L.range.yMax);
      r.trail.push({ x: r.pos.x, y: r.pos.y });
      if (r.trail.length > 250) r.trail.shift();
      r.lossHistory.push(L.L(r.pos.x, r.pos.y));
    }
    state.stepCount++;
    if (state.stepCount >= HISTORY_CAP) reset();
  }

  let renderCtx = null, cssW = 0, cssH = 0;
  let bgCache = null;

  function renderBg(w, h) {
    const L = losses(state.landscape);
    const off = document.createElement('canvas');
    off.width = w; off.height = h;
    const cctx = off.getContext('2d');
    const T = makeTransform({ x: 0, y: 0, width: w, height: h }, L.range);
    const img = cctx.createImageData(w, h);

    // First pass: compute V at every pixel for contrast normalisation.
    const vGrid = new Float64Array(w * h);
    let vmin = Infinity, vmax = -Infinity;
    for (let py = 0; py < h; py++) {
      for (let px = 0; px < w; px++) {
        const wc = T.fromScreen(px, py);
        const v = L.L(wc.x, wc.y);
        vGrid[py * w + px] = v;
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
      }
    }
    const range = vmax - vmin || 1;

    // Second pass: paint.  We apply a gamma curve (t^0.55) so the deep
    // basins look distinctly "well-like" instead of getting compressed
    // against the bottom of the colour ramp.  Dark navy for high loss,
    // bright teal for low loss.
    const ai   = [0x3d, 0xdc, 0x97];   // low loss
    const dark = [0x0a, 0x10, 0x1c];   // high loss
    for (let py = 0; py < h; py++) {
      for (let px = 0; px < w; px++) {
        const v = vGrid[py * w + px];
        const tNorm = (v - vmin) / range;       // 0 at minimum, 1 at peak
        const t = Math.pow(tNorm, 0.55);        // gamma: emphasises low end
        const r = Math.round(ai[0] * (1 - t) + dark[0] * t);
        const g = Math.round(ai[1] * (1 - t) + dark[1] * t);
        const b = Math.round(ai[2] * (1 - t) + dark[2] * t);
        const idx = (py * w + px) * 4;
        img.data[idx]   = r;
        img.data[idx+1] = g;
        img.data[idx+2] = b;
        img.data[idx+3] = 230;          // mostly opaque so basins read clearly
      }
    }
    cctx.putImageData(img, 0, 0);

    // Overlay iso-contour lines.  We pick 8 levels spaced linearly between
    // vmin and vmax, then for each pixel check if its level differs from
    // its right or bottom neighbour — if so we're at a level crossing.
    const nLevels = 8;
    const levelStep = range / nLevels;
    const contourImg = cctx.getImageData(0, 0, w, h);
    for (let py = 0; py < h - 1; py++) {
      for (let px = 0; px < w - 1; px++) {
        const v00 = vGrid[py * w + px];
        const v10 = vGrid[py * w + (px + 1)];
        const v01 = vGrid[(py + 1) * w + px];
        const l00 = Math.floor((v00 - vmin) / levelStep);
        const l10 = Math.floor((v10 - vmin) / levelStep);
        const l01 = Math.floor((v01 - vmin) / levelStep);
        if (l00 !== l10 || l00 !== l01) {
          const idx = (py * w + px) * 4;
          // Mix in a brighter teal hint to mark the contour.
          contourImg.data[idx]   = Math.min(255, contourImg.data[idx]   + 30);
          contourImg.data[idx+1] = Math.min(255, contourImg.data[idx+1] + 30);
          contourImg.data[idx+2] = Math.min(255, contourImg.data[idx+2] + 30);
        }
      }
    }
    cctx.putImageData(contourImg, 0, 0);
    return off;
  }

  // Aspect ratio 1.35: tall enough to fit both the landscape panel and the
  // loss-vs-time panel below it without either feeling cramped.
  makeResponsive(canvas, 1.35, (ctx, w, h) => {
    renderCtx = ctx; cssW = w; cssH = h;
    bgCache = renderBg(w, Math.round(h * LANDSCAPE_FRAC));
  });

  const lrSlider = slider({
    label: 'learning rate',
    min: 0.01, max: 0.2, step: 0.005, value: 0.04,
    format: v => v.toFixed(3),
    onChange: v => { state.lr = v; },
  });
  const sgdSlider = slider({
    label: 'SGD noise',
    min: 0, max: 2, step: 0.05, value: 0.6,
    format: v => v.toFixed(2),
    onChange: v => { state.sgdNoise = v; },
  });
  const tempSlider = slider({
    label: 'Langevin temp',
    min: 0.05, max: 2.5, step: 0.05, value: 0.35,
    format: v => v.toFixed(2),
    onChange: v => { state.langevinTemp = v; },
  });
  const pauseBtn = actionButton('pause / resume',
    () => { state.running = !state.running; });
  const resetBtn = actionButton('reset positions', reset);
  addControls(controlsRow, lrSlider, sgdSlider, tempSlider, pauseBtn, resetBtn);

  setCaption(captionEl,
    `<strong>Top panel:</strong> the loss landscape from above.  All three ` +
    `walkers start at the same spot on the shoulder, with a narrow-deep valley ` +
    `on the left and a wide-shallow valley on the right.  ` +
    `<strong>Bottom panel:</strong> each walker's loss value over time.  Watch ` +
    `for the moment <span style="color: #b084eb;">Langevin</span>'s curve jumps ` +
    `<em>up</em> (climbing the ridge between basins) and then drops <em>past</em> ` +
    `where <span style="color: #3ddc97;">GD</span> plateaued — that's the escape.  ` +
    `Plain GD (green) and SGD (orange) plateau at the narrow basin's depth and ` +
    `never improve from there.`);

  // ---- Drawing helpers for the loss-vs-time plot ------------------
  // We keep these inside the closure so they share state.runners.

  function drawLossPlot(ctx, px, py, plotW, plotH) {
    // Decide y-range from the data we have so far (with sensible padding)
    let lossMin = Infinity, lossMax = -Infinity;
    for (const r of state.runners) {
      for (let i = 0; i < r.lossHistory.length; i++) {
        const v = r.lossHistory[i];
        if (v < lossMin) lossMin = v;
        if (v > lossMax) lossMax = v;
      }
    }
    if (!isFinite(lossMin) || !isFinite(lossMax)) {
      // Nothing recorded yet — pick a stub range
      lossMin = -3; lossMax = 1;
    }
    // Pad
    const pad = Math.max(0.2, (lossMax - lossMin) * 0.08);
    lossMin -= pad; lossMax += pad;

    const xRange = HISTORY_CAP;     // steps shown on x-axis

    // Panel background
    ctx.fillStyle = '#070b15';
    ctx.fillRect(px, py, plotW, plotH);
    // Subtle border
    ctx.strokeStyle = '#1f2735';
    ctx.lineWidth = 1;
    ctx.strokeRect(px + 0.5, py + 0.5, plotW - 1, plotH - 1);

    // Coordinate transform (math y goes up; screen y goes down)
    function plotXY(stepIdx, lossVal) {
      const fx = stepIdx / xRange;
      const fy = (lossVal - lossMin) / (lossMax - lossMin);
      return {
        x: px + 36 + fx * (plotW - 46),         // leave 36px on left for y labels
        y: py + plotH - 26 - fy * (plotH - 38), // 26px on bottom, 12px on top
      };
    }

    // Reference horizontal lines: depths of the two minima, if we have them
    const L = losses(state.landscape);
    if (L.mins) {
      for (const m of L.mins) {
        const depth = L.L(m.x, m.y);
        const p = plotXY(0, depth);
        const pEnd = plotXY(xRange, depth);
        ctx.strokeStyle = '#1f2735';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(pEnd.x, pEnd.y);
        ctx.stroke();
        ctx.setLineDash([]);
        drawLabel(ctx, m.label.replace(' min', ''),
          pEnd.x - 4, p.y - 5,
          { color: '#6b7280', font: '10px "JetBrains Mono"',
            align: 'right', baseline: 'bottom' });
      }
    }

    // Numerical tick marks
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 1;
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.fillStyle = '#9da8c2';
    // Y-axis: 3 ticks (top, middle, bottom)
    const yTickVals = [lossMin, (lossMin + lossMax) / 2, lossMax];
    for (const yv of yTickVals) {
      const sp = plotXY(0, yv);
      ctx.beginPath();
      ctx.moveTo(sp.x, sp.y);
      ctx.lineTo(sp.x - 3, sp.y);
      ctx.stroke();
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(yv.toFixed(1), sp.x - 5, sp.y);
    }
    // X-axis: 5 ticks
    for (let k = 0; k <= 4; k++) {
      const sv = Math.round(k * xRange / 4);
      const sp = plotXY(sv, lossMin);
      ctx.beginPath();
      ctx.moveTo(sp.x, sp.y);
      ctx.lineTo(sp.x, sp.y + 3);
      ctx.stroke();
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(String(sv), sp.x, sp.y + 5);
    }

    // Each walker's loss curve
    for (const r of state.runners) {
      if (r.lossHistory.length < 2) continue;
      ctx.strokeStyle = r.color;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      const p0 = plotXY(0, r.lossHistory[0]);
      ctx.moveTo(p0.x, p0.y);
      for (let i = 1; i < r.lossHistory.length; i++) {
        const p = plotXY(i, r.lossHistory[i]);
        ctx.lineTo(p.x, p.y);
      }
      ctx.stroke();

      // Mark the current value with a dot
      const last = r.lossHistory.length - 1;
      const pCur = plotXY(last, r.lossHistory[last]);
      dot(ctx, pCur.x, pCur.y, 3, r.color);
    }

    // Axis labels
    drawLabel(ctx, 'loss',
      px + 4, py + 12,
      { color: '#6b7280', font: '10px "JetBrains Mono"', align: 'left', baseline: 'top' });
    drawLabel(ctx, 'steps →',
      px + plotW - 6, py + plotH - 6,
      { color: '#6b7280', font: '10px "JetBrains Mono"', align: 'right', baseline: 'bottom' });

    // Step count display
    drawLabel(ctx, `step ${state.stepCount}`,
      px + plotW - 6, py + 12,
      { color: '#9da8c2', font: '11px "JetBrains Mono"',
        align: 'right', baseline: 'top' });
  }

  makeVisibilityLoop(root, () => {
    if (!renderCtx) return;
    if (state.running) {
      for (let s = 0; s < 1; s++) step();
    }
    const ctx = renderCtx;
    clearCanvas(ctx, cssW, cssH, '#0a0e1a');

    const landscapeH = Math.round(cssH * LANDSCAPE_FRAC);
    const plotPadding = 10;
    const plotY = landscapeH + plotPadding;
    const plotH = cssH - plotY;

    // -------- Landscape panel (top) --------
    if (bgCache) ctx.drawImage(bgCache, 0, 0);

    const L = losses(state.landscape);
    const landscapeScreen = { x: 0, y: 0, width: cssW, height: landscapeH };
    const T = makeTransform(landscapeScreen, L.range);
    drawAxes(ctx, T, L.range);

    // Overlay-style axis labels (the canvas IS the visualization, no margin room)
    drawLabel(ctx, 'parameter θ₁ →', cssW - 8, landscapeH - 6,
      { color: '#9da8c2', font: '11px "JetBrains Mono"', align: 'right', baseline: 'bottom' });
    drawLabel(ctx, '↑ parameter θ₂', 8, 8,
      { color: '#9da8c2', font: '11px "JetBrains Mono"', align: 'left', baseline: 'top' });
    drawLabel(ctx, 'brighter = lower loss L(θ₁, θ₂)', cssW / 2, 8,
      { color: '#9da8c2', font: '11px "JetBrains Mono"', align: 'center', baseline: 'top' });

    // Mark minima.  Labels are placed well above the basin to keep them clear
    // of walkers (which settle inside) and trails (which loop around inside).
    // Different vertical offsets for the two markers — narrow basin label
    // goes higher than wide basin label so they don't pile up if both happen
    // to be near the top of the canvas.
    const minimaLabelOffsets = [
      { dy: -28 },   // first listed minimum (wide global)
      { dy: -22 },   // second listed minimum (narrow local)
    ];
    L.mins.forEach((m, idx) => {
      const p = T.toScreen(m.x, m.y);
      ctx.strokeStyle = '#9da8c2';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 9, 0, Math.PI * 2);
      ctx.stroke();
      const off = minimaLabelOffsets[idx] || { dy: -22 };
      drawLabel(ctx, m.label, p.x, p.y + off.dy,
        { color: '#9da8c2', font: '11px "JetBrains Mono"',
          align: 'center', baseline: 'bottom' });
    });

    // Trails first, so dots draw on top
    for (const r of state.runners) {
      if (!r.trail || r.trail.length < 2) continue;
      ctx.strokeStyle = r.color;
      ctx.lineWidth = 1.6;
      ctx.globalAlpha = 0.55;
      ctx.beginPath();
      const p0 = T.toScreen(r.trail[0].x, r.trail[0].y);
      ctx.moveTo(p0.x, p0.y);
      for (let k = 1; k < r.trail.length; k++) {
        const p = T.toScreen(r.trail[k].x, r.trail[k].y);
        ctx.lineTo(p.x, p.y);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Walker dots on landscape.  Each walker's label sits at a different
    // vertical offset from its dot so the three names don't pile up when
    // walkers cluster (which they do — GD and SGD park in the same basin).
    // Order: GD up-and-right, SGD down-and-right, Langevin up-and-left.
    const labelOffsets = {
      'GD':   { dx:  10, dy: -12, align: 'left',  baseline: 'middle' },
      'SGD':  { dx:  10, dy:  14, align: 'left',  baseline: 'middle' },
      'LANG': { dx: -10, dy: -12, align: 'right', baseline: 'middle' },
    };
    for (const r of state.runners) {
      const p = T.toScreen(r.pos.x, r.pos.y);
      dot(ctx, p.x, p.y, 5.5, r.color);
      const off = labelOffsets[r.kind];
      drawLabel(ctx, r.label, p.x + off.dx, p.y + off.dy,
        { color: r.color, font: '11px "JetBrains Mono"',
          align: off.align, baseline: off.baseline });
    }

    // -------- Loss-vs-time panel (bottom) --------
    drawLossPlot(ctx, 0, plotY, cssW, plotH);
  });
}

// ----------------------------------------------------------------------------
// Chapter 9 — Score field (THE CLIMAX)
// ----------------------------------------------------------------------------
export async function initScoreFieldSandbox(rootId) {
  const root = document.getElementById(rootId);
  if (!root) return;

  // Show a loading state first
  root.classList.add('sandbox');
  root.innerHTML = '<div class="loading">loading trained model and data</div>';

  let model, data;
  try {
    [model, data] = await Promise.all([
      loadScoreModel('assets/score_model.json'),
      fetch('assets/two_moons.json').then(r => r.json()),
    ]);
  } catch (e) {
    root.innerHTML = '<div class="loading">failed to load model — check console</div>';
    console.error(e);
    return;
  }

  const { canvas, controlsRow, captionEl } = makeSandbox(root,
    'a real trained network — drag noise, watch arrows, sample fresh points');

  const state = {
    t: 0.4,
    showArrows: true,
    showData: true,
    samples: [],
    rng: null,
    gauss: null,
    sampling: false,
    samplingStep: 0,
    samplingTotal: 100,
    arrowGrid: null,    // cached grid of scores
    arrowGridT: -1,
    chamferDist: null,  // last computed comparison metric, null when none
  };
  state.rng = makeRng(31);
  state.gauss = makeGaussian(state.rng);

  // ------ Comparison metric ----------------------------------------------
  // The chamfer distance between samples and training data is a single
  // number that quantifies "how well do samples match the data?".  For each
  // sample, we find the nearest training point and record its distance;
  // for each training point, we find the nearest sample and record that
  // distance; the chamfer is the average of both means.  Low value = good.
  //
  // For a random Gaussian initialization (200 points), the chamfer to two
  // moons is roughly 0.7-1.0.  After successful reverse-SDE sampling, it
  // drops below ~0.15.  This gives the reader a concrete number to watch.
  function computeChamfer() {
    if (state.samples.length === 0) return null;
    let sumS = 0;
    for (const s of state.samples) {
      let best = Infinity;
      for (const d of data) {
        const dx = s.x - d[0], dy = s.y - d[1];
        const r2 = dx * dx + dy * dy;
        if (r2 < best) best = r2;
      }
      sumS += Math.sqrt(best);
    }
    let sumD = 0;
    // Subsample the training data for speed — 100 points is plenty for the metric
    const dataSubsample = data.filter((_, i) => i % Math.ceil(data.length / 100) === 0);
    for (const d of dataSubsample) {
      let best = Infinity;
      for (const s of state.samples) {
        const dx = s.x - d[0], dy = s.y - d[1];
        const r2 = dx * dx + dy * dy;
        if (r2 < best) best = r2;
      }
      sumD += Math.sqrt(best);
    }
    return (sumS / state.samples.length + sumD / dataSubsample.length) / 2;
  }

  function recomputeArrows() {
    const t = state.t;
    if (Math.abs(state.arrowGridT - t) < 1e-6) return;
    const grid = [];
    const n = 18;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const x = -2.2 + (i + 0.5) / n * 4.4;
        const y = -2.2 + (j + 0.5) / n * 4.4;
        const s = model.forward({ x, y }, t);
        grid.push({ x, y, sx: s.sx, sy: s.sy });
      }
    }
    state.arrowGrid = grid;
    state.arrowGridT = t;
  }

  let renderCtx = null, cssW = 0, cssH = 0;
  makeResponsive(canvas, 1.0, (ctx, w, h) => { renderCtx = ctx; cssW = w; cssH = h; });

  // Controls
  const tSlider = slider({
    label: 'noise level t',
    min: 0.01, max: 0.99, step: 0.01, value: 0.4,
    format: v => v.toFixed(2),
    onChange: v => { state.t = v; recomputeArrows(); },
  });
  const arrowsBtn = actionButton('toggle score arrows',
    () => { state.showArrows = !state.showArrows; });
  const dataBtn = actionButton('toggle training data',
    () => { state.showData = !state.showData; });
  const sampleBtn = actionButton('sample 200 fresh points', () => {
    state.samples = [];
    for (let i = 0; i < 200; i++) {
      state.samples.push({ x: state.gauss(), y: state.gauss() });
    }
    state.sampling = true;
    state.samplingStep = 0;
    // Initial chamfer (from pure Gaussian noise) — typically ~0.7-1.0
    state.chamferDist = computeChamfer();
  }, true);
  const clearBtn = actionButton('clear samples', () => {
    state.samples = []; state.sampling = false; state.chamferDist = null;
  });
  addControls(controlsRow, tSlider, arrowsBtn, dataBtn, sampleBtn, clearBtn);

  // Click on canvas to place a sample
  canvas.addEventListener('click', (ev) => {
    if (!renderCtx) return;
    const r = canvas.getBoundingClientRect();
    const px = ev.clientX - r.left;
    const py = ev.clientY - r.top;
    const world = { xMin: -2.5, xMax: 2.5, yMin: -2.5, yMax: 2.5 };
    const T = makeTransform({ x: 0, y: 0, width: cssW, height: cssH }, world);
    const wc = T.fromScreen(px, py);
    state.samples.push({ x: wc.x, y: wc.y });
    if (!state.sampling) {
      state.sampling = true;
      state.samplingStep = 0;
    }
  });

  setCaption(captionEl,
    `This network was trained to predict the <em>score</em> ∇log p<sub>t</sub>(x) ` +
    `for a two-moons distribution at every noise level t ∈ [0, 1].  Drag t and ` +
    `watch the arrows: at high t the field points everywhere toward the origin ` +
    `(everything looks like Gaussian noise); at low t the field resolves into ` +
    `two crescents.  Hit "sample" to start with random Gaussian points and watch ` +
    `them flow back along the field to the moons.  <em>Or click anywhere</em> ` +
    `to drop a point and watch it be pulled home.`);

  recomputeArrows();

  makeVisibilityLoop(root, () => {
    if (!renderCtx) return;

    // Step the reverse SDE for any active samples
    if (state.sampling && state.samples.length > 0 && state.samplingStep < state.samplingTotal) {
      const total = state.samplingTotal;
      // Decrease t from current state.t down to a small value over `total` steps
      const tStart = 0.99;
      const tEnd = 1e-3;
      const tNow = tStart - (state.samplingStep / total) * (tStart - tEnd);
      const dt = -(tStart - tEnd) / total;
      reverseSdeStep(state.samples, tNow, dt, state.gauss, model);
      state.samplingStep++;
      // Update the chamfer distance periodically so the reader can watch it drop
      if (state.samplingStep % 10 === 0 || state.samplingStep >= state.samplingTotal) {
        state.chamferDist = computeChamfer();
      }
      if (state.samplingStep >= state.samplingTotal) state.sampling = false;
    }

    const ctx = renderCtx;
    clearCanvas(ctx, cssW, cssH, '#0a0e1a');
    const world = { xMin: -2.5, xMax: 2.5, yMin: -2.5, yMax: 2.5 };
    const screen = { x: 0, y: 0, width: cssW, height: cssH };
    const T = makeTransform(screen, world);
    drawAxes(ctx, T, world);

    // Overlay-style axis labels
    drawLabel(ctx, 'data dimension x →', cssW - 8, cssH - 6,
      { color: '#9da8c2', font: '11px "JetBrains Mono"', align: 'right', baseline: 'bottom' });
    drawLabel(ctx, '↑ data dimension y', 8, 8,
      { color: '#9da8c2', font: '11px "JetBrains Mono"', align: 'left', baseline: 'top' });
    drawLabel(ctx, 'cyan dots = training data,  gold arrows = learned score ∇log p_t(x)',
      cssW / 2, 8,
      { color: '#9da8c2', font: '11px "JetBrains Mono"', align: 'center', baseline: 'top' });

    // Training data
    if (state.showData) {
      for (const d of data) {
        const p = T.toScreen(d[0], d[1]);
        dot(ctx, p.x, p.y, 1.8, '#5bc5f2', 0.7);
      }
    }

    // Score arrows
    if (state.showArrows && state.arrowGrid) {
      for (const g of state.arrowGrid) {
        const mag = Math.hypot(g.sx, g.sy);
        if (mag < 0.01) continue;
        // Display scale (large scores get clamped)
        const scale = Math.min(0.25, mag * 0.04);
        const len = scale / mag;
        const x2 = g.x + g.sx * len;
        const y2 = g.y + g.sy * len;
        const p1 = T.toScreen(g.x, g.y);
        const p2 = T.toScreen(x2, y2);
        arrow(ctx, p1.x, p1.y, p2.x, p2.y, '#f4d35e',
          { lineWidth: 1.2, headLen: 5, alpha: Math.min(0.85, mag * 0.18) });
      }
    }

    // Active samples
    for (const s of state.samples) {
      const p = T.toScreen(s.x, s.y);
      dot(ctx, p.x, p.y, 3, '#3ddc97', 0.95);
    }

    // Status
    drawLabel(ctx, `t = ${state.t.toFixed(2)}`,
      cssW - 10, 16,
      { color: '#f4d35e', font: '13px "JetBrains Mono"', align: 'right', baseline: 'top' });
    if (state.sampling) {
      drawLabel(ctx, `sampling: ${state.samplingStep}/${state.samplingTotal}`,
        cssW - 10, 34,
        { color: '#3ddc97', font: '11px "JetBrains Mono"', align: 'right', baseline: 'top' });
    } else if (state.samples.length > 0) {
      drawLabel(ctx, `${state.samples.length} sampled points`,
        cssW - 10, 34,
        { color: '#3ddc97', font: '11px "JetBrains Mono"', align: 'right', baseline: 'top' });
    }
    // Comparison metric: chamfer distance between samples and training data
    if (state.chamferDist !== null) {
      drawLabel(ctx,
        `chamfer(samples, data) = ${state.chamferDist.toFixed(3)}`,
        cssW - 10, 52,
        { color: '#3ddc97', font: '11px "JetBrains Mono"', align: 'right', baseline: 'top' });
    }
  });
}
