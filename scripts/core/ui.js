// ============================================================================
// Declarative UI helpers for sandboxes.
// All functions create DOM elements and return them.  The state lives in a
// caller-supplied object that gets mutated; the caller hooks in with onChange.
// ============================================================================

/**
 * Create a slider with a label and live value readout.
 * @param {object} opts
 * @param {string} opts.label
 * @param {number} opts.min
 * @param {number} opts.max
 * @param {number} opts.step
 * @param {number} opts.value initial value
 * @param {string} [opts.format] sprintf-ish e.g. "{}" or "n={}"; default "{:.2f}"
 * @param {(v:number)=>void} opts.onChange called on every input event
 * @returns {{element: HTMLElement, set: (v:number)=>void, get: ()=>number}}
 */
export function slider(opts) {
  const ctrl = document.createElement('div');
  ctrl.className = 'control';
  const labelRow = document.createElement('div');
  labelRow.className = 'control-label';
  const labelSpan = document.createElement('span');
  labelSpan.textContent = opts.label;
  const valueSpan = document.createElement('span');
  valueSpan.className = 'control-value';
  labelRow.appendChild(labelSpan);
  labelRow.appendChild(valueSpan);
  ctrl.appendChild(labelRow);

  const range = document.createElement('input');
  range.type = 'range';
  range.min = opts.min;
  range.max = opts.max;
  range.step = opts.step;
  range.value = opts.value;
  ctrl.appendChild(range);

  function formatValue(v) {
    if (opts.format) return opts.format(v);
    if (opts.step >= 1) return v.toFixed(0);
    if (opts.step >= 0.01) return v.toFixed(2);
    return v.toFixed(3);
  }

  function updateDisplay(v) {
    valueSpan.textContent = formatValue(v);
  }

  updateDisplay(opts.value);
  range.addEventListener('input', () => {
    const v = parseFloat(range.value);
    updateDisplay(v);
    opts.onChange(v);
  });

  return {
    element: ctrl,
    set: (v) => { range.value = v; updateDisplay(v); },
    get: () => parseFloat(range.value),
  };
}

/**
 * Row of preset buttons (mutually exclusive choice).
 * @param {object} opts
 * @param {string[]} opts.labels button labels
 * @param {number} opts.active index of initially-active button
 * @param {(i:number)=>void} opts.onChange called with the new active index
 */
export function presetRow(opts) {
  const wrap = document.createElement('div');
  wrap.className = 'preset-row';
  const buttons = [];
  opts.labels.forEach((label, i) => {
    const btn = document.createElement('button');
    btn.className = 'preset-btn';
    btn.textContent = label;
    if (i === opts.active) btn.classList.add('active');
    btn.addEventListener('click', () => {
      buttons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      opts.onChange(i);
    });
    buttons.push(btn);
    wrap.appendChild(btn);
  });
  return {
    element: wrap,
    select: (i) => {
      buttons.forEach((b, j) => b.classList.toggle('active', i === j));
    },
  };
}

/**
 * A single action button.
 */
export function actionButton(label, onClick, primary = false) {
  const btn = document.createElement('button');
  btn.textContent = label;
  if (primary) btn.classList.add('primary');
  btn.addEventListener('click', onClick);
  return btn;
}

/**
 * Toggle button (boolean state).
 */
export function toggleButton(opts) {
  const btn = document.createElement('button');
  let state = opts.value;
  function render() {
    btn.textContent = state ? opts.labelOn : opts.labelOff;
    btn.classList.toggle('primary', state);
  }
  render();
  btn.addEventListener('click', () => {
    state = !state;
    render();
    opts.onChange(state);
  });
  return {
    element: btn,
    set: (v) => { state = v; render(); },
    get: () => state,
  };
}

/**
 * A live readout (key-value display).
 * @param {{label:string, format?:(v:any)=>string}[]} fields
 * @returns {{element:HTMLElement, update: (values:object)=>void}}
 */
export function readout(fields) {
  const el = document.createElement('div');
  el.className = 'sandbox-readout';
  const spans = {};
  fields.forEach((f, i) => {
    if (i > 0) el.appendChild(document.createTextNode('  '));
    const label = document.createElement('span');
    label.className = 'label';
    label.textContent = f.label + ' ';
    el.appendChild(label);
    const value = document.createElement('span');
    value.className = 'value';
    el.appendChild(value);
    spans[f.key] = { value, format: f.format || ((v) => String(v)) };
  });
  return {
    element: el,
    update(values) {
      for (const k in values) {
        if (spans[k]) spans[k].value.textContent = spans[k].format(values[k]);
      }
    },
  };
}

/**
 * Set up a sandbox container with a title and a canvas, returning references.
 * Standard skeleton used by all sandboxes.
 * @param {HTMLElement} root  the placeholder element to fill
 * @param {string} title
 * @param {object} [opts] {wide:boolean, aspectRatio:number}
 * @returns {{canvas, controlsRow, captionEl}}
 */
export function makeSandbox(root, title, opts = {}) {
  if (opts.wide) root.classList.add('sandbox-wide');
  root.classList.add('sandbox');
  root.innerHTML = '';

  const titleEl = document.createElement('div');
  titleEl.className = 'sandbox-title';
  titleEl.textContent = title;
  root.appendChild(titleEl);

  const canvas = document.createElement('canvas');
  canvas.className = 'sandbox-canvas';
  root.appendChild(canvas);

  const controlsRow = document.createElement('div');
  controlsRow.className = 'sandbox-controls';
  root.appendChild(controlsRow);

  const captionEl = document.createElement('div');
  captionEl.className = 'sandbox-caption';
  root.appendChild(captionEl);

  return { canvas, controlsRow, captionEl };
}

/**
 * Add controls to the controls row in order.
 */
export function addControls(row, ...elements) {
  elements.forEach(e => row.appendChild(e.element || e));
}

/**
 * Set a sandbox's caption (italic prose below the controls).
 */
export function setCaption(captionEl, html) {
  captionEl.innerHTML = html;
}

/**
 * Only run a sandbox's render loop when it's actually in view.
 * Returns a controller {start, stop, isRunning}.
 *
 * @param {HTMLElement} target the element to observe
 * @param {() => void} renderFn called once per frame while visible
 */
export function makeVisibilityLoop(target, renderFn) {
  let running = false;
  let rafId = null;
  function tick() {
    if (!running) return;
    renderFn();
    rafId = requestAnimationFrame(tick);
  }
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting && !running) {
        running = true;
        tick();
      } else if (!e.isIntersecting && running) {
        running = false;
        if (rafId) cancelAnimationFrame(rafId);
      }
    }
  }, { rootMargin: '50px' });
  io.observe(target);
  return {
    start: () => { if (!running) { running = true; tick(); } },
    stop: () => { running = false; if (rafId) cancelAnimationFrame(rafId); },
    isRunning: () => running,
    disconnect: () => { running = false; io.disconnect(); },
  };
}
