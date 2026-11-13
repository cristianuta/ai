'use strict';
/* Tab 2: interfața pentru algoritmul genetic. Evoluția rulează pe server (/api/ga/...). */
const GAApp = (function () {
  const C = {
    a: '#4da3ff', b: '#ff9a4d', green: '#3ddc84', red: '#ff5d6c', yellow: '#ffd84d',
    off: '#182131', text: '#c2cbdb', muted: '#8d9ab3',
  };
  const bits = s => Uint8Array.from(s, c => (c === '1' ? 1 : 0));

  const state = { snap: null, running: false };
  const targetGrid = new PixelGrid($('ga-target'), { onChange: () => sendTarget(), onColor: [255, 216, 77] });

  const params = () => ({
    mutationRate: +$('ga-mut').value,
    crossover: $('ga-cross-type').value,
    elitism: $('ga-elite').checked,
  });

  // ---------- comunicarea cu serverul ----------
  function apply(snap) {
    state.snap = snap;
    renderAll();
  }

  const sendTarget = latestOnly(async () => {
    const target = Array.from(targetGrid.values, v => (v > 0.5 ? '1' : '0')).join('');
    apply(await Api.post('/api/ga/target', { target }));
    $('ga-status').textContent = '';
  });

  async function loadPreset(preset) {
    await guard(async () => {
      const snap = await Api.post('/api/ga/target', { preset });
      targetGrid.setSilently(bits(snap.target));
      $('ga-status').textContent = '';
      apply(snap);
    });
  }

  async function reset() {
    setRunning(false);
    await guard(async () => {
      apply(await Api.post('/api/ga/reset', { populationSize: +$('ga-pop').value, ...params() }));
      $('ga-status').textContent = '';
    });
  }

  async function step(generations) {
    const snap = await Api.post('/api/ga/step', { generations, ...params() });
    apply(snap);
    if (snap.solved) {
      const msg = `🎉 Ținta a fost atinsă în generația ${snap.generation}!`;
      $('ga-status').textContent = msg;
      if (state.running) toast(msg, 'success');
      setRunning(false);
    } else {
      $('ga-status').textContent = '';
    }
  }

  async function loop() {
    let last = performance.now(), acc = 1;
    while (state.running) {
      const now = performance.now();
      acc += Math.min(0.5, (now - last) / 1000) * +$('ga-speed').value;
      last = now;
      const n = Math.min(200, Math.floor(acc));
      if (n >= 1) {
        acc -= n;
        try { await step(n); } catch (e) { toast(e.message, 'error'); setRunning(false); }
      }
      await nextFrame();
    }
  }

  function setRunning(on) {
    if (on === state.running) return;
    state.running = on;
    $('ga-start').textContent = on ? '⏸ Pauză' : '▶ Start evoluție';
    $('ga-start').classList.toggle('running', on);
    if (on) loop();
  }

  // ---------- desen ----------
  function renderAll() {
    const s = state.snap;
    if (Tabs.active !== 'ga' || !s) return;
    $('ga-s-gen').textContent = s.generation;
    $('ga-s-best').textContent = `${s.best}/${s.length}`;
    $('ga-s-avg').textContent = `${s.average.toFixed(1)}/${s.length}`;
    $('ga-s-cross').textContent = s.crossCount.toLocaleString('ro-RO');
    renderCross();
    renderChart();
    renderPool();
    renderPopulation();
  }

  /** Desenează o grilă 8×8 de gene. cellFn(i) -> {fill, alpha, stroke} */
  function miniGrid(ctx, x, y, size, n, cellFn) {
    const s = size / 8;
    ctx.fillStyle = '#0a0f18';
    roundRect(ctx, x - 3, y - 3, size + 6, size + 6, 8); ctx.fill();
    for (let i = 0; i < n; i++) {
      const cx = x + (i % 8) * s, cy = y + Math.floor(i / 8) * s;
      const { fill, alpha = 1, stroke } = cellFn(i);
      ctx.globalAlpha = alpha;
      ctx.fillStyle = fill;
      roundRect(ctx, cx + 1, cy + 1, s - 2, s - 2, Math.min(3, s * 0.15)); ctx.fill();
      ctx.globalAlpha = 1;
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.strokeRect(cx + 1.5, cy + 1.5, s - 3, s - 3); }
    }
  }

  const geneColor = (on, color) => (on ? color : C.off);

  function renderCross() {
    const { ctx, w, h } = fitCanvas($('ga-cross'));
    ctx.font = '12px system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    const s = state.snap, lc = s.lastCross;
    const T = bits(s.target), n = s.length;
    if (!lc) {
      ctx.fillStyle = C.muted; ctx.textAlign = 'center';
      ctx.font = '14px system-ui, sans-serif';
      ctx.fillText('Apasă „+1 generație” sau „Start evoluție” ca să vezi prima încrucișare.', w / 2, h / 2);
      return;
    }
    const a = bits(lc.parentA), b = bits(lc.parentB), child = bits(lc.child), origin = bits(lc.origin);
    const mut = new Set(lc.mutated);

    // --- rândul 1: grile 8×8 ---
    const gs = Math.min(124, (w - 120) / 4), gap = (w - 4 * gs) / 5, gy = 28;
    const xs = [0, 1, 2, 3].map(k => gap + k * (gs + gap));
    const titles = [
      [`Părinte A · ${lc.fitnessA}/${n}`, C.a], [`Părinte B · ${lc.fitnessB}/${n}`, C.b],
      [`Copil · ${lc.fitnessChild}/${n}`, C.green], ['Ținta', C.yellow],
    ];
    ctx.textAlign = 'center';
    ctx.font = '600 12.5px system-ui, sans-serif';
    titles.forEach(([t, c], k) => { ctx.fillStyle = c; ctx.fillText(t, xs[k] + gs / 2, 10); });
    miniGrid(ctx, xs[0], gy, gs, n, i => ({ fill: geneColor(a[i], C.a), alpha: origin[i] === 0 ? 1 : 0.22 }));
    miniGrid(ctx, xs[1], gy, gs, n, i => ({ fill: geneColor(b[i], C.b), alpha: origin[i] === 1 ? 1 : 0.22 }));
    miniGrid(ctx, xs[2], gy, gs, n, i => ({ fill: geneColor(child[i], origin[i] ? C.b : C.a), stroke: mut.has(i) ? C.yellow : null }));
    miniGrid(ctx, xs[3], gy, gs, n, i => ({ fill: T[i] ? C.yellow : C.off }));
    ctx.fillStyle = C.muted; ctx.font = 'bold 22px system-ui, sans-serif';
    ['×', '→', '≈'].forEach((sym, k) => ctx.fillText(sym, xs[k] + gs + gap / 2, gy + gs / 2));

    // --- rândul 2: vectorii ca benzi de 64 de gene ---
    const labelW = 74, sx = labelW, cw = (w - labelW - 6) / n;
    const top = gy + gs + 30, rowH = Math.max(14, Math.min(20, cw * 1.6)), rowGap = 12;
    const rows = [
      { label: 'Vector A', genes: a, color: () => C.a, alpha: i => (origin[i] === 0 ? 1 : 0.22) },
      { label: 'Vector B', genes: b, color: () => C.b, alpha: i => (origin[i] === 1 ? 1 : 0.22) },
      { label: 'Copil', genes: child, color: i => (origin[i] ? C.b : C.a), alpha: () => 1, mut: true },
      { label: 'Ținta', genes: T, color: () => C.yellow, alpha: () => 1, noMatch: true },
    ];
    ctx.font = '600 12px system-ui, sans-serif';
    rows.forEach((r, k) => {
      const y = top + k * (rowH + rowGap);
      ctx.textAlign = 'right'; ctx.fillStyle = C.text;
      ctx.fillText(r.label, labelW - 10, y + rowH / 2);
      for (let i = 0; i < n; i++) {
        const x = sx + i * cw;
        ctx.globalAlpha = r.alpha(i);
        ctx.fillStyle = geneColor(r.genes[i], r.color(i));
        ctx.fillRect(x + 0.5, y, cw - 1, rowH);
        if (!r.noMatch) {
          ctx.fillStyle = r.genes[i] === T[i] ? C.green : C.red;
          ctx.fillRect(x + 0.5, y + rowH + 2, cw - 1, 3);
        }
        ctx.globalAlpha = 1;
        if (r.mut && mut.has(i)) { ctx.strokeStyle = C.yellow; ctx.lineWidth = 2; ctx.strokeRect(x + 1, y + 1, cw - 2, rowH - 2); }
      }
    });
    // liniile de tăiere
    ctx.setLineDash([4, 3]); ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.5;
    const bottom = top + 3 * (rowH + rowGap) - rowGap + 6;
    for (const c of lc.cuts) {
      const x = sx + c * cw;
      ctx.beginPath(); ctx.moveTo(x, top - 6); ctx.lineTo(x, bottom); ctx.stroke();
    }
    ctx.setLineDash([]);
    // indici + descriere
    ctx.fillStyle = C.muted; ctx.font = '10px system-ui, sans-serif'; ctx.textAlign = 'center';
    const yIdx = top + 4 * (rowH + rowGap) - 2;
    for (const i of [0, 16, 32, 48, 63]) ctx.fillText(String(i), sx + (i + 0.5) * cw, yIdx);
    const cutText = lc.cuts.length ? `✂ tăiere la poziția ${lc.cuts.join(' și ')}` : '✂ încrucișare uniformă: fiecare genă aleasă aleator';
    const m = lc.mutated.length;
    ctx.textAlign = 'left'; ctx.font = '11.5px system-ui, sans-serif';
    ctx.fillText(`${cutText} · ${m} ${m === 1 ? 'mutație' : 'mutații'}`, sx, yIdx + 16);
  }

  function renderChart() {
    const s = state.snap, pct = v => (v / s.length) * 100;
    drawChart($('ga-chart'), [
      { label: 'cel mai bun', color: C.green, data: s.history.best.map(pct) },
      { label: 'media', color: C.a, data: s.history.average.map(pct) },
      { label: 'cel mai slab', color: C.red, data: s.history.worst.map(pct) },
    ], { min: 0, max: 100, format: v => v.toFixed(0) + '%', xLabel: 'generații', xCount: s.history.generations - 1 });
  }

  function renderPool() {
    const { ctx, w } = fitCanvas($('ga-pool'));
    const pool = state.snap.genePool, s = w / 8;
    ctx.font = `600 ${Math.max(9, Math.min(13, s * 0.32))}px system-ui, sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let i = 0; i < 64; i++) {
      const f = pool[i];
      const r = Math.round(255 + (61 - 255) * f), g = Math.round(93 + (220 - 93) * f), b = Math.round(108 + (132 - 108) * f);
      const x = (i % 8) * s, y = Math.floor(i / 8) * s;
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.globalAlpha = 0.3 + 0.7 * Math.abs(f - 0.5) * 2;
      roundRect(ctx, x + 1.5, y + 1.5, s - 3, s - 3, Math.min(6, s * 0.15)); ctx.fill();
      ctx.globalAlpha = 1;
      if (s >= 24) { ctx.fillStyle = '#070a12'; ctx.fillText(Math.round(f * 100), x + s / 2, y + s / 2); }
    }
  }

  function renderPopulation() {
    const canvas = $('ga-population');
    const cw = canvas.clientWidth;
    if (!cw) return;
    const s = state.snap, T = bits(s.target);
    const thumb = 42, gap = 8, labelH = 15;
    const cols = Math.max(1, Math.floor((cw + gap) / (thumb + gap)));
    const rows = Math.ceil(s.population.length / cols);
    const { ctx } = fitCanvas(canvas, rows * (thumb + labelH + gap));
    ctx.font = '10.5px system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    s.population.forEach((ind, k) => {
      const genes = bits(ind.genes);
      const x = (k % cols) * (thumb + gap), y = Math.floor(k / cols) * (thumb + labelH + gap);
      const f = ind.fitness / s.length;
      ctx.fillStyle = '#0a0f18';
      roundRect(ctx, x, y, thumb, thumb, 6); ctx.fill();
      const cs = (thumb - 4) / 8;
      for (let i = 0; i < 64; i++) {
        if (!genes[i]) continue;
        ctx.fillStyle = genes[i] === T[i] ? '#e8f1ff' : C.red;
        ctx.fillRect(x + 2 + (i % 8) * cs + 0.4, y + 2 + Math.floor(i / 8) * cs + 0.4, cs - 0.8, cs - 0.8);
      }
      const isBestChild = k === s.bestChildIndex;
      ctx.strokeStyle = ind.elite ? C.yellow : isBestChild ? C.green : 'rgba(255,255,255,0.12)';
      ctx.lineWidth = ind.elite || isBestChild ? 2 : 1;
      roundRect(ctx, x + 0.5, y + 0.5, thumb - 1, thumb - 1, 6); ctx.stroke();
      ctx.fillStyle = f === 1 ? C.green : C.muted;
      ctx.fillText(Math.round(f * 100) + '%', x + thumb / 2, y + thumb + labelH / 2 + 1);
    });
  }

  // ---------- controale ----------
  bindRange('ga-pop', v => v);
  bindRange('ga-mut', v => (v * 100).toFixed(1) + '%');
  bindRange('ga-speed', v => v);
  $('ga-pop').addEventListener('change', reset);

  $('ga-start').addEventListener('click', () => {
    if (!state.running) $('ga-status').textContent = '';
    setRunning(!state.running);
  });
  $('ga-step').addEventListener('click', async e => {
    const btn = e.currentTarget;
    btn.disabled = true;
    await guard(() => step(1));
    btn.disabled = false;
  });
  $('ga-reset').addEventListener('click', reset);
  $('ga-preset').addEventListener('change', () => loadPreset($('ga-preset').value));
  $('ga-clear-target').addEventListener('click', () => targetGrid.clear());

  Tabs.onShow('ga', () => { targetGrid.render(); renderAll(); });

  // ---------- pornire ----------
  async function init() {
    const snap = await Api.get('/api/ga/state');
    targetGrid.setSilently(bits(snap.target));
    $('ga-pop').value = snap.populationSize;
    $('ga-mut').value = snap.mutationRate;
    $('ga-cross-type').value = snap.crossover;
    $('ga-elite').checked = snap.elitism;
    ['ga-pop', 'ga-mut'].forEach(id => $(id).dispatchEvent(new Event('input')));
    apply(snap);
  }

  return { init };
})();
