'use strict';
/* Tab 1: interfața pentru rețeaua neurală. Calculele se fac pe server (/api/nn/...). */
const NNApp = (function () {
  const POS = [90, 170, 255], NEG = [255, 120, 80], OUT = [61, 220, 132];
  const HOVER_HINT = 'Treci cu mouse-ul peste un neuron ca să-i vezi conexiunile.';

  const state = { snap: null, running: false, hover: null, layout: null, view: 'w' };

  const grid = new PixelGrid($('nn-draw'), {
    onChange: () => { $('nn-draw-msg').textContent = ''; refreshActivations(); },
  });
  for (let d = 0; d < 10; d++) $('nn-label').add(new Option(String(d), String(d)));

  const input = () => Array.from(grid.values);
  const learningRate = () => +$('nn-lr').value;

  // ---------- comunicarea cu serverul ----------
  function apply(snap) {
    state.snap = snap;
    if (state.hover && !(snap.sizes.length > state.hover.l && snap.sizes[state.hover.l] > state.hover.i)) state.hover = null;
    renderPred();
    updateStats();
    drawCharts();
    requestRender();
  }

  /** Recalculează activările pentru desenul curent (doar când nu antrenăm — altfel vin cu antrenarea). */
  const refreshActivations = latestOnly(async () => {
    if (!state.snap || state.running) return;
    const r = await Api.post('/api/nn/predict', { input: input() });
    state.snap.activations = r.activations;
    renderPred();
    requestRender();
  });

  async function trainLoop() {
    while (state.running) {
      try {
        const snap = await Api.post('/api/nn/train', {
          batches: +$('nn-speed').value, learningRate: learningRate(), input: input(),
        });
        if (state.running) apply(snap);
      } catch (e) {
        toast(e.message, 'error');
        setRunning(false);
      }
      await nextFrame();
    }
  }

  function setRunning(on) {
    if (on === state.running) return;
    state.running = on;
    const btn = $('nn-train');
    btn.textContent = on ? '⏸ Pauză' : '▶ Antrenează';
    btn.classList.toggle('running', on);
    if (on) trainLoop();
    else refreshActivations();
  }

  async function withButton(btn, fn) {
    btn.disabled = true;
    try { await guard(fn); } finally { btn.disabled = false; }
  }

  // ---------- desen ----------
  let renderQueued = false;
  function requestRender() {
    if (renderQueued) return;
    renderQueued = true;
    requestAnimationFrame(() => {
      renderQueued = false;
      if (Tabs.active !== 'nn' || !state.snap) return;
      renderNet();
      renderFields();
    });
  }

  function computeLayout(sizes, w, h) {
    const L = sizes.length, top = 38, bottom = 14;
    const avail = h - top - bottom;
    const gridSize = Math.min(avail, w * 0.33);
    const sp = gridSize / 8;
    const gx = 10, gy = top + (avail - gridSize) / 2;
    const layers = [[]];
    for (let k = 0; k < 64; k++) {
      layers[0].push({ x: gx + (k % 8 + 0.5) * sp, y: gy + (Math.floor(k / 8) + 0.5) * sp, r: Math.max(2.5, sp * 0.3) });
    }
    const left = gx + gridSize + Math.max(50, w * 0.13), right = w - 80;
    for (let l = 1; l < L; l++) {
      const n = sizes[l];
      const x = L === 2 ? right : left + (l - 1) * (right - left) / (L - 2);
      const spacing = Math.min(38, avail / n);
      const r = Math.max(3, Math.min(12, spacing * 0.36));
      const arr = [];
      for (let i = 0; i < n; i++) arr.push({ x, y: top + avail / 2 + (i - (n - 1) / 2) * spacing, r });
      layers.push(arr);
    }
    return layers;
  }

  const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  const mix = (c, m) => `rgb(${Math.round(24 + (c[0] - 24) * m)},${Math.round(32 + (c[1] - 32) * m)},${Math.round(48 + (c[2] - 48) * m)})`;

  function argmax(a) {
    let best = 0;
    for (let j = 1; j < a.length; j++) if (a[j] > a[best]) best = j;
    return best;
  }

  function renderNet() {
    const { ctx, w, h } = fitCanvas($('nn-net'));
    const { sizes, weights, activations: acts } = state.snap;
    const lay = computeLayout(sizes, w, h);
    state.layout = lay;
    const signal = state.view === 's';
    const thr = +$('nn-thr').value;
    const hv = state.hover;
    const L = sizes.length;

    // titluri straturi
    ctx.font = '600 12px system-ui, sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = '#8d9ab3';
    const g0 = lay[0];
    ctx.fillText('INTRARE 8×8 · 64', (g0[0].x + g0[7].x) / 2, 14);
    for (let l = 1; l < L; l++) {
      ctx.fillText(l === L - 1 ? 'IEȘIRE · 10' : `ASCUNS ${l} · ${sizes[l]}`, lay[l][0].x, 14);
    }

    // conexiuni
    for (let l = 0; l < L - 1; l++) {
      const W = weights[l], nin = sizes[l], nout = sizes[l + 1], a = acts[l];
      const src = lay[l], dst = lay[l + 1];
      let mx = 1e-9;
      for (let k = 0; k < W.length; k++) {
        const v = Math.abs(signal ? W[k] * a[k % nin] : W[k]);
        if (v > mx) mx = v;
      }
      for (let j = 0; j < nout; j++) {
        for (let i = 0; i < nin; i++) {
          const wv = W[j * nin + i];
          const v = signal ? wv * a[i] : wv;
          let s = Math.abs(v) / mx;
          const focus = hv && ((hv.l === l && hv.i === i) || (hv.l === l + 1 && hv.i === j));
          if (hv && !focus) {
            if (s < thr) continue;
            s *= 0.12;
          } else if (!focus && s < thr) continue;
          const alpha = focus ? Math.max(0.12, Math.min(1, s * 1.1)) : Math.min(0.85, 0.03 + s * s * 0.8);
          ctx.strokeStyle = rgba(v >= 0 ? POS : NEG, alpha.toFixed(3));
          ctx.lineWidth = focus ? 0.6 + s * 2.4 : 0.5 + s * 1.2;
          ctx.beginPath();
          ctx.moveTo(src[i].x, src[i].y);
          ctx.lineTo(dst[j].x, dst[j].y);
          ctx.stroke();
        }
      }
    }

    // noduri
    const out = acts[L - 1];
    const best = argmax(out);
    const blank = grid.isBlank();
    for (let l = 0; l < L; l++) {
      for (let i = 0; i < lay[l].length; i++) {
        const p = lay[l][i], a = acts[l][i];
        let fill, glow = null;
        if (l === 0) {
          const c = Math.round(30 + 215 * a);
          fill = `rgb(${c},${c},${Math.min(255, c + 10)})`;
        } else if (l === L - 1) {
          fill = mix(OUT, a);
          if (a > 0.3) glow = rgba(OUT, 0.8);
        } else {
          fill = mix(a >= 0 ? POS : NEG, Math.abs(a));
          if (Math.abs(a) > 0.7) glow = rgba(a >= 0 ? POS : NEG, 0.7);
        }
        if (glow) { ctx.shadowColor = glow; ctx.shadowBlur = p.r * 1.4; }
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = fill; ctx.fill();
        ctx.shadowBlur = 0;
        const isHover = hv && hv.l === l && hv.i === i;
        const isBest = l === L - 1 && i === best && !blank;
        ctx.lineWidth = isHover || isBest ? 2.5 : 1;
        ctx.strokeStyle = isHover ? '#ffd84d' : isBest ? '#3ddc84' : 'rgba(255,255,255,0.22)';
        ctx.stroke();

        if (l === L - 1) {
          ctx.textAlign = 'left';
          ctx.font = (isBest ? '800 ' : '600 ') + '13px system-ui, sans-serif';
          ctx.fillStyle = isBest ? '#3ddc84' : '#c2cbdb';
          ctx.fillText(String(i), p.x + p.r + 8, p.y);
          ctx.font = '11px system-ui, sans-serif';
          ctx.fillStyle = '#8d9ab3';
          ctx.fillText((a * 100).toFixed(0) + '%', p.x + p.r + 24, p.y);
        }
      }
    }
    ctx.lineWidth = 1;
  }

  function renderFields() {
    const canvas = $('nn-fields');
    const { sizes, weights } = state.snap;
    const n = sizes[1];
    const cols = Math.min(8, n), gap = 6;
    const cw = canvas.clientWidth;
    if (!cw) return;
    const fs = (cw - (cols - 1) * gap) / cols;
    const rows = Math.ceil(n / cols);
    const { ctx } = fitCanvas(canvas, rows * (fs + gap) - gap);
    const W = weights[0], cell = fs / 8;
    for (let k = 0; k < n; k++) {
      const ox = (k % cols) * (fs + gap), oy = Math.floor(k / cols) * (fs + gap);
      let mx = 1e-9;
      for (let p = 0; p < 64; p++) mx = Math.max(mx, Math.abs(W[k * 64 + p]));
      ctx.fillStyle = '#0c111c';
      roundRect(ctx, ox, oy, fs, fs, 5); ctx.fill();
      ctx.save();
      roundRect(ctx, ox, oy, fs, fs, 5); ctx.clip();
      for (let p = 0; p < 64; p++) {
        const v = W[k * 64 + p] / mx;
        ctx.fillStyle = rgba(v >= 0 ? POS : NEG, Math.abs(v).toFixed(3));
        ctx.fillRect(ox + (p % 8) * cell, oy + Math.floor(p / 8) * cell, cell + 0.3, cell + 0.3);
      }
      ctx.restore();
      const hl = state.hover && state.hover.l === 1 && state.hover.i === k;
      ctx.strokeStyle = hl ? '#ffd84d' : 'rgba(255,255,255,0.12)';
      ctx.lineWidth = hl ? 2 : 1;
      roundRect(ctx, ox + 0.5, oy + 0.5, fs - 1, fs - 1, 5); ctx.stroke();
    }
  }

  // predicție (HTML)
  $('nn-pred').innerHTML =
    '<div class="big-pred"><span class="digit" id="nn-big">–</span><span class="conf" id="nn-conf"></span></div>' +
    Array.from({ length: 10 }, (_, d) =>
      `<div class="pred-row" id="nn-pr-${d}"><span>${d}</span><div class="pred-bar"><div></div></div><span>0%</span></div>`).join('');

  function renderPred() {
    const acts = state.snap.activations;
    const out = acts[acts.length - 1];
    const best = argmax(out);
    const blank = grid.isBlank();
    $('nn-big').textContent = blank ? '–' : best;
    $('nn-conf').innerHTML = blank ? 'Desenează o cifră<br>în grila de sus' :
      `<b>${(out[best] * 100).toFixed(1)}%</b>încredere`;
    for (let d = 0; d < 10; d++) {
      const row = $('nn-pr-' + d);
      row.classList.toggle('top', d === best && !blank);
      row.querySelector('.pred-bar > div').style.width = (out[d] * 100).toFixed(1) + '%';
      row.lastElementChild.textContent = (out[d] * 100).toFixed(0) + '%';
    }
  }

  function updateStats() {
    const s = state.snap, h = s.history;
    const last = a => (a.length ? a[a.length - 1] : null);
    $('nn-s-epoch').textContent = s.epoch;
    $('nn-s-loss').textContent = s.lastLoss == null ? '–' : s.lastLoss.toFixed(3);
    $('nn-s-train').textContent = last(h.trainAccuracy) == null ? '–' : last(h.trainAccuracy).toFixed(1) + '%';
    $('nn-s-test').textContent = last(h.testAccuracy) == null ? '–' : last(h.testAccuracy).toFixed(1) + '%';
    $('nn-s-samples').textContent = s.trainSamples;
    $('nn-s-params').textContent = s.parameters.toLocaleString('ro-RO');
  }

  function drawCharts() {
    if (Tabs.active !== 'nn' || !state.snap) return;
    const h = state.snap.history;
    drawChart($('nn-loss'), [
      { label: 'antrenare', color: '#4da3ff', data: h.trainLoss },
      { label: 'test', color: '#ff9a4d', data: h.testLoss },
    ], { min: 0, xLabel: 'epoci' });
    drawChart($('nn-acc'), [
      { label: 'antrenare', color: '#4da3ff', data: h.trainAccuracy },
      { label: 'test', color: '#3ddc84', data: h.testAccuracy },
    ], { min: 0, max: 100, format: v => v.toFixed(0) + '%', xLabel: 'epoci' });
  }

  // ---------- hover pe neuroni ----------
  function describe(found) {
    const a = state.snap.activations[found.l][found.i];
    const L = state.snap.sizes.length;
    if (found.l === 0) return `Pixelul (${found.i % 8}, ${Math.floor(found.i / 8)}) = ${a.toFixed(2)}`;
    if (found.l === L - 1) return `Ieșirea „${found.i}” — probabilitate ${(a * 100).toFixed(1)}%`;
    return `Neuronul #${found.i + 1} din stratul ascuns ${found.l} — activare ${a.toFixed(2)}`;
  }

  $('nn-net').addEventListener('pointermove', e => {
    if (!state.layout) return;
    const r = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let found = null;
    state.layout.forEach((layer, l) => layer.forEach((p, i) => {
      if ((p.x - mx) ** 2 + (p.y - my) ** 2 <= (p.r + 4) ** 2) found = { l, i };
    }));
    const h = state.hover;
    if ((found && h && found.l === h.l && found.i === h.i) || (!found && !h)) return;
    state.hover = found;
    $('nn-hover-info').textContent = found ? describe(found) : HOVER_HINT;
    requestRender();
  });
  $('nn-net').addEventListener('pointerleave', () => {
    state.hover = null;
    $('nn-hover-info').textContent = HOVER_HINT;
    requestRender();
  });

  // ---------- controale ----------
  bindRange('nn-lr', v => v.toFixed(3));
  bindRange('nn-speed', v => v);
  const showNoise = bindRange('nn-noise', v => Math.round(v * 100) + '%');

  $('nn-view').addEventListener('click', e => {
    const b = e.target.closest('button');
    if (!b) return;
    state.view = b.dataset.value;
    $('nn-view').querySelectorAll('button').forEach(x => x.classList.toggle('active', x === b));
    requestRender();
  });
  $('nn-thr').addEventListener('input', requestRender);

  $('nn-train').addEventListener('click', () => setRunning(!state.running));

  $('nn-epoch').addEventListener('click', e => withButton(e.currentTarget, async () => {
    apply(await Api.post('/api/nn/epoch', { learningRate: learningRate(), input: input() }));
  }));

  const resetNet = btn => withButton(btn, async () => {
    const hidden = $('nn-arch').value.split(',').map(Number);
    apply(await Api.post('/api/nn/reset', { hidden, input: input() }));
    toast(`Rețea nouă: 64 → ${hidden.join(' → ')} → 10, ponderi aleatoare`);
  });
  $('nn-reset').addEventListener('click', e => resetNet(e.currentTarget));
  $('nn-arch').addEventListener('change', e => resetNet(e.currentTarget));

  const regenerate = btn => withButton(btn, async () => {
    apply(await Api.post('/api/nn/data', { noise: +$('nn-noise').value, input: input() }));
    toast('Am generat exemple noi de antrenare și test');
  });
  $('nn-regen').addEventListener('click', e => regenerate(e.currentTarget));
  $('nn-noise').addEventListener('change', e => regenerate(e.currentTarget));

  $('nn-clear').addEventListener('click', () => grid.clear());

  $('nn-random').addEventListener('click', () => guard(async () => {
    const s = await Api.get('/api/nn/random-sample');
    grid.set(s.pixels);
    $('nn-label').value = String(s.label);
    $('nn-draw-msg').textContent = `Exemplu din setul de test — răspunsul corect: ${s.label}`;
  }));

  $('nn-add').addEventListener('click', e => {
    if (grid.isBlank()) { toast('Desenează mai întâi ceva.', 'error'); return; }
    const label = +$('nn-label').value;
    withButton(e.currentTarget, async () => {
      apply(await Api.post('/api/nn/samples', { pixels: input(), label }));
      toast(`Adăugat ca „${label}” (+20 variații). Antrenează ca rețeaua să-l învețe.`, 'success');
    });
  });

  Tabs.onShow('nn', () => { grid.render(); drawCharts(); requestRender(); });

  // ---------- pornire ----------
  async function init() {
    const sample = await Api.get('/api/nn/random-sample');
    grid.setSilently(sample.pixels);
    $('nn-label').value = String(sample.label);
    const snap = await Api.post('/api/nn/state', { input: input() });
    $('nn-noise').value = snap.noise;
    showNoise();
    const arch = snap.sizes.slice(1, -1).join(',');
    if ([...$('nn-arch').options].some(o => o.value === arch)) $('nn-arch').value = arch;
    apply(snap);
  }

  return { init };
})();
