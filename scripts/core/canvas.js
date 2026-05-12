// ============================================================================
// Canvas rendering helpers.  Every sandbox uses these.
// ============================================================================

/**
 * Set up a canvas for crisp DPI-aware rendering. Returns the 2D context.
 * Call this whenever the canvas's container size might have changed.
 * @param {HTMLCanvasElement} canvas
 * @param {{width: number, height: number}} cssSize size in CSS pixels
 * @returns {CanvasRenderingContext2D}
 */
export function setupCanvas(canvas, cssSize) {
  const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  canvas.style.width = cssSize.width + 'px';
  canvas.style.height = cssSize.height + 'px';
  canvas.width = Math.round(cssSize.width * dpr);
  canvas.height = Math.round(cssSize.height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

/**
 * Build a coordinate transformer between scene (math) coordinates and screen
 * (pixel) coordinates, given a bounding box and target screen rect.
 *
 * @param {{x: number, y: number, width: number, height: number}} screen
 *   pixel rect to render into
 * @param {{xMin: number, xMax: number, yMin: number, yMax: number}} world
 *   the math-coordinate bounding box (yMin is the bottom, yMax the top in math sense)
 * @returns {{toScreen: (x:number,y:number)=>{x:number,y:number}, fromScreen: ...,
 *           xScale: number, yScale: number}}
 */
export function makeTransform(screen, world) {
  const xScale = screen.width / (world.xMax - world.xMin);
  const yScale = screen.height / (world.yMax - world.yMin);
  return {
    toScreen(x, y) {
      return {
        x: screen.x + (x - world.xMin) * xScale,
        // Y is flipped: math up is screen up
        y: screen.y + (world.yMax - y) * yScale,
      };
    },
    fromScreen(px, py) {
      return {
        x: world.xMin + (px - screen.x) / xScale,
        y: world.yMax - (py - screen.y) / yScale,
      };
    },
    xScale, yScale,
  };
}

/**
 * Draw axes (only the ones in view).  Subtle, doesn't dominate.
 */
export function drawAxes(ctx, T, world, opts = {}) {
  const color = opts.color || '#1f2735';
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;

  // y = 0 line if visible
  if (world.yMin <= 0 && world.yMax >= 0) {
    const p1 = T.toScreen(world.xMin, 0);
    const p2 = T.toScreen(world.xMax, 0);
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  }
  // x = 0 line if visible
  if (world.xMin <= 0 && world.xMax >= 0) {
    const p1 = T.toScreen(0, world.yMin);
    const p2 = T.toScreen(0, world.yMax);
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  }
}

/**
 * Draw a path (polyline through points).
 */
export function drawPath(ctx, points, color, lineWidth = 2, alpha = 1) {
  if (points.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
  ctx.stroke();
  ctx.globalAlpha = 1;
}

/**
 * Draw a smooth curve given (x, y) arrays in world coordinates.
 */
export function drawCurve(ctx, T, xs, ys, color, lineWidth = 2, alpha = 1) {
  if (xs.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  const p0 = T.toScreen(xs[0], ys[0]);
  ctx.moveTo(p0.x, p0.y);
  for (let i = 1; i < xs.length; i++) {
    const p = T.toScreen(xs[i], ys[i]);
    ctx.lineTo(p.x, p.y);
  }
  ctx.stroke();
  ctx.globalAlpha = 1;
}

/**
 * Draw a filled circle in screen coordinates.
 */
export function dot(ctx, px, py, r, color, alpha = 1) {
  ctx.fillStyle = color;
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.arc(px, py, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = 1;
}

/**
 * Draw an arrow from (x1, y1) to (x2, y2) in screen coordinates.
 */
export function arrow(ctx, x1, y1, x2, y2, color, opts = {}) {
  const headLen = opts.headLen || 6;
  const lineWidth = opts.lineWidth || 1.5;
  const alpha = opts.alpha != null ? opts.alpha : 1;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  // Head
  const ang = Math.atan2(y2 - y1, x2 - x1);
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - headLen * Math.cos(ang - Math.PI / 6),
             y2 - headLen * Math.sin(ang - Math.PI / 6));
  ctx.lineTo(x2 - headLen * Math.cos(ang + Math.PI / 6),
             y2 - headLen * Math.sin(ang + Math.PI / 6));
  ctx.closePath();
  ctx.fill();
  ctx.globalAlpha = 1;
}

/**
 * Draw histogram bars as filled rectangles.
 */
export function drawHistogram(ctx, T, edges, density, color, alpha = 0.6, yScale = 1) {
  ctx.fillStyle = color;
  ctx.globalAlpha = alpha;
  for (let i = 0; i < density.length; i++) {
    const h = density[i] * yScale;
    if (h <= 0) continue;
    const p1 = T.toScreen(edges[i], h);
    const p2 = T.toScreen(edges[i + 1], 0);
    ctx.fillRect(p1.x + 1, p1.y, p2.x - p1.x - 2, p2.y - p1.y);
  }
  ctx.globalAlpha = 1;
}

/**
 * Draw axis labels and optional tick marks at salient values.
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {{toScreen, fromScreen}} T  the coordinate transform
 * @param {{xMin, xMax, yMin, yMax}} world  the visible world bounds
 * @param {{x, y, width, height}} screen  the screen rectangle being plotted in
 * @param {Object} opts
 * @param {string} [opts.xLabel] short label for x-axis (e.g. "x", "time t")
 * @param {string} [opts.yLabel] short label for y-axis (e.g. "f(x)", "density")
 * @param {Array<{at:number, label:string}>} [opts.xTicks] tick marks on x-axis
 * @param {Array<{at:number, label:string}>} [opts.yTicks] tick marks on y-axis
 *
 * Labels are placed just inside the screen rect's corners so they don't
 * collide with axis lines: x at bottom-right, y at top-left.  Ticks are
 * drawn as small dashed verticals/horizontals at the specified world values.
 */
export function drawAxisLabels(ctx, T, world, screen, opts = {}) {
  const labelColor = '#9da8c2';
  const tickColor = '#6b7280';
  const labelFont = '11px "JetBrains Mono", monospace';
  const tickFont = '10px "JetBrains Mono", monospace';

  // X-axis label: anchor bottom-right of the screen rect, italic-ish look.
  if (opts.xLabel) {
    ctx.fillStyle = labelColor;
    ctx.font = labelFont;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillText(opts.xLabel + ' →', screen.x + screen.width - 6, screen.y + screen.height - 4);
  }

  // Y-axis label: anchor top-left.  Drawn horizontally (a rotated label is
  // marginally cleaner but harder to read at small sizes; the convention in
  // explainer-style explorables is a horizontal label up top.)
  if (opts.yLabel) {
    ctx.fillStyle = labelColor;
    ctx.font = labelFont;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('↑ ' + opts.yLabel, screen.x + 6, screen.y + 4);
  }

  // X-ticks: drawn just below the x-axis (we assume y=0 if visible, otherwise
  // at the bottom of the screen rect).
  if (opts.xTicks && opts.xTicks.length) {
    ctx.strokeStyle = tickColor;
    ctx.fillStyle = tickColor;
    ctx.font = tickFont;
    ctx.lineWidth = 1;
    const yLine = world.yMin <= 0 && world.yMax >= 0 ? 0 : world.yMin;
    for (const t of opts.xTicks) {
      const p = T.toScreen(t.at, yLine);
      ctx.beginPath();
      ctx.moveTo(p.x, p.y - 3);
      ctx.lineTo(p.x, p.y + 3);
      ctx.stroke();
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(t.label, p.x, p.y + 5);
    }
  }

  // Y-ticks: drawn at the screen-LEFT edge of the plot (not at the world's
  // x=0 line, because that crosses through the middle of histograms and gets
  // hidden by the data).
  if (opts.yTicks && opts.yTicks.length) {
    ctx.strokeStyle = tickColor;
    ctx.fillStyle = tickColor;
    ctx.font = tickFont;
    ctx.lineWidth = 1;
    for (const t of opts.yTicks) {
      const p = T.toScreen(world.xMin, t.at);
      ctx.beginPath();
      ctx.moveTo(p.x - 3, p.y);
      ctx.lineTo(p.x + 3, p.y);
      ctx.stroke();
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(t.label, p.x - 5, p.y);
    }
  }
}

/**
 * Soft-clear the canvas (lays down the background tint).
 */
export function clearCanvas(ctx, w, h, bg) {
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);
}

/**
 * Draw a label at a screen position.
 */
export function drawLabel(ctx, text, px, py, opts = {}) {
  const font = opts.font || '12px "JetBrains Mono", monospace';
  const color = opts.color || '#9da8c2';
  const align = opts.align || 'left';
  const baseline = opts.baseline || 'middle';
  ctx.font = font;
  ctx.fillStyle = color;
  ctx.textAlign = align;
  ctx.textBaseline = baseline;
  ctx.fillText(text, px, py);
}

/**
 * Make a canvas responsive to its container, calling onResize whenever the
 * available width changes.  Returns a cleanup function.
 */
export function makeResponsive(canvas, aspectRatio, onResize) {
  const parent = canvas.parentElement;
  let lastWidth = 0;

  function doResize() {
    const cssWidth = parent.clientWidth;
    if (cssWidth === lastWidth) return;
    lastWidth = cssWidth;
    const cssHeight = Math.round(cssWidth / aspectRatio);
    const ctx = setupCanvas(canvas, { width: cssWidth, height: cssHeight });
    onResize(ctx, cssWidth, cssHeight);
  }

  doResize();
  const ro = new ResizeObserver(doResize);
  ro.observe(parent);
  return () => ro.disconnect();
}
