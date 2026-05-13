// ============================================================================
// Story-so-far widget.
// Lives as a fixed sidebar on the right; scrolls along, lighting up entries
// as the reader passes through their chapters.
// ============================================================================

/**
 * Build the widget DOM and attach scroll spy.
 *
 * @param {{key:string, label:string, sectionId:string}[]} items
 *   In article order.  Each gets lit up when its section enters the viewport.
 */
export function initStoryWidget(items) {
  // Build DOM
  const widget = document.createElement('aside');
  widget.id = 'story-widget';
  widget.innerHTML = `<div class="widget-title">Story so far</div>`;
  const itemEls = {};
  items.forEach((item) => {
    const el = document.createElement('div');
    el.className = 'story-item';
    el.innerHTML = `<span class="dot"></span><span>${item.label}</span>`;
    el.addEventListener('click', () => {
      const target = document.getElementById(item.sectionId);
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    widget.appendChild(el);
    itemEls[item.key] = { el, sectionId: item.sectionId };
  });
  document.body.appendChild(widget);

  // Track which sections have been seen + which is current.
  const seen = new Set();
  let currentKey = null;

  function update() {
    const fold = window.innerHeight * 0.4;  // "the reader is at this height"
    let bestSection = null;
    let bestDistance = Infinity;

    items.forEach((item) => {
      const section = document.getElementById(item.sectionId);
      if (!section) return;
      const r = section.getBoundingClientRect();
      // If any part is above the fold, mark as seen.
      if (r.top < window.innerHeight * 0.5) seen.add(item.key);
      // Find the section whose top is closest to the fold (above).
      const d = fold - r.top;
      if (d >= 0 && d < bestDistance) {
        bestDistance = d;
        bestSection = item.key;
      }
    });

    if (bestSection !== currentKey) {
      currentKey = bestSection;
      items.forEach((item) => {
        const entry = itemEls[item.key];
        entry.el.classList.toggle('unlocked', seen.has(item.key));
        entry.el.classList.toggle('active', item.key === currentKey);
      });
    } else {
      // Update unlocked state in case new sections came in view
      items.forEach((item) => {
        if (seen.has(item.key)) {
          itemEls[item.key].el.classList.add('unlocked');
        }
      });
    }

    // Show the widget after the reader has scrolled past the article header
    const headerBottom = document.querySelector('.article-header')?.getBoundingClientRect().bottom || 0;
    if (headerBottom < 0) {
      widget.classList.add('visible');
    } else {
      widget.classList.remove('visible');
    }
  }

  window.addEventListener('scroll', update, { passive: true });
  window.addEventListener('resize', update);
  update();
}

/**
 * Initialise the sticky table of contents at the top of the page.
 */
export function initToc(items) {
  const toc = document.createElement('nav');
  toc.id = 'toc';
  const inner = document.createElement('div');
  inner.id = 'toc-inner';
  toc.appendChild(inner);
  const titleSpan = document.createElement('span');
  titleSpan.id = 'toc-title';
  titleSpan.textContent = 'Noise';
  inner.appendChild(titleSpan);

  const links = {};
  items.forEach((item) => {
    const a = document.createElement('a');
    a.href = '#' + item.sectionId;
    a.textContent = item.shortLabel || item.label;
    a.dataset.key = item.key;
    inner.appendChild(a);
    links[item.key] = a;
  });

  document.body.insertBefore(toc, document.body.firstChild);

  function update() {
    const fold = window.innerHeight * 0.4;
    let currentKey = null;
    let bestDistance = Infinity;
    items.forEach((item) => {
      const section = document.getElementById(item.sectionId);
      if (!section) return;
      const r = section.getBoundingClientRect();
      const d = fold - r.top;
      if (d >= 0 && d < bestDistance) {
        bestDistance = d;
        currentKey = item.key;
      }
    });
    items.forEach((item) => {
      links[item.key].classList.toggle('toc-current', item.key === currentKey);
    });
  }

  window.addEventListener('scroll', update, { passive: true });
  update();
}
