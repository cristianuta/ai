'use strict';
/* Utilitare comune: API, notificări, canvas HiDPI, grilă de pixeli, grafice, tab-uri. */

const $ = id => document.getElementById(id);
const nextFrame = () => new Promise(r => requestAnimationFrame(r));

/* ---------------- API către backend-ul Spring Boot ---------------- */
function setServerStatus(ok) {
  const el = $('server-status');
  el.classList.toggle('ok', ok);
  el.classList.toggle('err', !ok);
  el.querySelector('span').textContent = ok ? 'server Java conectat' : 'server indisponibil';
}

const Api = {
  async request(method, url, body) {
    let res;
    try {
      res = await fetch(url, {
        method,
        headers: body !== undefined ? { 'Content-Type': 'application/json' } : {},
        body: body !== undefined ? JSON.stringify(body) : undefined,
      });
    } catch (e) {
      setServerStatus(false);
      throw new Error('Serverul nu răspunde. Pornește aplicația Spring Boot (mvnw spring-boot:run).');
    }
    setServerStatus(true);
    const data = await res.json().catch(() => null);
    if (!res.ok) throw new Error((data && data.error) || `Eroare server (${res.status})`);
    return data;
  },
  get(url) { return this.request('GET', url); },
  post(url, body = {}) { return this.request('POST', url, body); },
};

/** Execută o funcție async și afișează eventualele erori ca notificare. */
async function guard(fn) {
  try { return await fn(); } catch (e) { toast(e.message, 'error'); return null; }
}

/**
 * Împachetează o funcție async astfel încât să ruleze cel mult o instanță odată;
 * apelurile venite între timp se comasează într-un singur apel ulterior (cu ultimele argumente).
 */
function latestOnly(fn) {
  let busy = false, pending = null;
  return async function run(...args) {
    if (busy) { pending = args; return; }
    busy = true;
    try { await fn(...args); } catch (e) { toast(e.message, 'error'); }
    finally {
      busy = false;
      if (pending) { const a = pending; pending = null; run(...a); }
    }
  };
}

/* ---------------- notificări ---------------- */
function toast(message, kind = 'info', ms = 3200) {
  const el = document.createElement('div');
  el.className = 'toast ' + kind;
  el.textContent = message;
  $('toasts').appendChild(el);
  setTimeout(() => { el.classList.add('out'); setTimeout(() => el.remove(), 300); }, ms);
}

/* ---------------- canvas ---------------- */
/** Potrivește rezoluția canvas-ului cu mărimea lui CSS (HiDPI) și îl golește. */
function fitCanvas(canvas, cssHeight) {
  if (cssHeight != null) canvas.style.height = cssHeight + 'px';
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const W = Math.max(1, Math.round(w * dpr)), H = Math.max(1, Math.round(h * dpr));
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/** Grilă de N×N pixeli pe care se poate desena cu mouse-ul. */
class PixelGrid {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.size = opts.size || 8;
    this.values = new Float32Array(this.size * this.size);
    this.onChange = opts.onChange || (() => {});
    this.onColor = opts.onColor || [232, 241, 255];
    this.painting = null;

    canvas.addEventListener('contextmenu', e => e.preventDefault());
    canvas.addEventListener('pointerdown', e => {
      this.painting = (e.button === 2 || e.shiftKey) ? 0 : 1;
      canvas.setPointerCapture(e.pointerId);
      this.paintAt(e);
    });
    canvas.addEventListener('pointermove', e => { if (this.painting !== null) this.paintAt(e); });
    const stop = () => { this.painting = null; };
    canvas.addEventListener('pointerup', stop);
    canvas.addEventListener('pointercancel', stop);
    this.render();
  }

  cellAt(e) {
    const r = this.canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - r.left) / r.width * this.size);
    const y = Math.floor((e.clientY - r.top) / r.height * this.size);
    if (x < 0 || y < 0 || x >= this.size || y >= this.size) return -1;
    return y * this.size + x;
  }

  paintAt(e) {
    const i = this.cellAt(e);
    if (i < 0 || this.values[i] === this.painting) return;
    this.values[i] = this.painting;
    this.render();
    this.onChange(this.values);
  }

  /** Setează valorile și anunță schimbarea. */
  set(values) { this.setSilently(values); this.onChange(this.values); }

  /** Setează valorile fără a declanșa onChange. */
  setSilently(values) { this.values.set(values); this.render(); }

  clear() { this.set(new Float32Array(this.values.length)); }

  isBlank() { return !this.values.some(v => v > 0); }

  render() {
    const { ctx, w } = fitCanvas(this.canvas);
    const n = this.size, s = w / n;
    const grad = ctx.createLinearGradient(0, 0, w, w);
    grad.addColorStop(0, '#0d1422'); grad.addColorStop(1, '#090d16');
    ctx.fillStyle = grad;
    roundRect(ctx, 0, 0, w, w, 12); ctx.fill();
    const [r, g, b] = this.onColor;
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const v = this.values[y * n + x];
        const k = 0.1 + 0.9 * v;
        if (v > 0.5) { ctx.shadowColor = `rgba(${r},${g},${b},0.55)`; ctx.shadowBlur = s * 0.35; }
        ctx.fillStyle = `rgb(${Math.round(24 + (r - 24) * v * k)},${Math.round(32 + (g - 32) * v * k)},${Math.round(50 + (b - 50) * v * k)})`;
        roundRect(ctx, x * s + 2, y * s + 2, s - 4, s - 4, Math.min(6, s * 0.18));
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }
  }
}

/** Grafic cu linii. series: [{label, color, data:[...]}]; opts: min, max, format, xLabel, xCount */
function drawChart(canvas, series, opts = {}) {
  const { ctx, w, h } = fitCanvas(canvas);
  const pad = { l: 40, r: 12, t: 24, b: 20 };
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
  let n = 0, lo = Infinity, hi = -Infinity;
  for (const s of series) {
    n = Math.max(n, s.data.length);
    for (const v of s.data) { if (v < lo) lo = v; if (v > hi) hi = v; }
  }
  if (opts.min != null) lo = opts.min;
  if (opts.max != null) hi = opts.max;
  if (!isFinite(lo) || !isFinite(hi)) { lo = 0; hi = 1; }
  if (hi - lo < 1e-9) hi = lo + 1;
  const fmt = opts.format || (v => v.toFixed(2));
  const X = i => pad.l + (n <= 1 ? 0 : i / (n - 1)) * pw;
  const Y = v => pad.t + (1 - (v - lo) / (hi - lo)) * ph;

  ctx.font = '10.5px system-ui, sans-serif';
  ctx.textBaseline = 'middle';
  for (let k = 0; k <= 4; k++) {
    const v = lo + (hi - lo) * k / 4, y = Y(v);
    ctx.strokeStyle = 'rgba(148,170,214,0.08)';
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillStyle = '#6f7d96'; ctx.textAlign = 'right';
    ctx.fillText(fmt(v), pad.l - 6, y);
  }
  ctx.textAlign = 'right';
  const xCount = opts.xCount != null ? opts.xCount : Math.max(0, n - 1);
  ctx.fillText(`${xCount} ${opts.xLabel || ''}`, w - pad.r, h - 7);

  if (n === 0) {
    ctx.textAlign = 'center'; ctx.fillStyle = '#6f7d96';
    ctx.fillText('Nicio dată încă', pad.l + pw / 2, pad.t + ph / 2);
    return;
  }
  for (const s of series) {
    if (!s.data.length) continue;
    // umplere discretă sub linie
    const fill = ctx.createLinearGradient(0, pad.t, 0, pad.t + ph);
    fill.addColorStop(0, s.color + '33'); fill.addColorStop(1, s.color + '00');
    ctx.beginPath();
    s.data.forEach((v, i) => (i === 0 ? ctx.moveTo(X(i), Y(v)) : ctx.lineTo(X(i), Y(v))));
    ctx.lineTo(X(s.data.length - 1), pad.t + ph);
    ctx.lineTo(X(0), pad.t + ph);
    ctx.closePath();
    ctx.fillStyle = fill; ctx.fill();

    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.lineJoin = 'round';
    ctx.beginPath();
    s.data.forEach((v, i) => (i === 0 ? ctx.moveTo(X(i), Y(v)) : ctx.lineTo(X(i), Y(v))));
    ctx.stroke();
    const last = s.data.length - 1;
    ctx.fillStyle = s.color;
    ctx.beginPath(); ctx.arc(X(last), Y(s.data[last]), 3, 0, Math.PI * 2); ctx.fill();
  }
  // legendă
  ctx.textAlign = 'left';
  let lx = pad.l;
  for (const s of series) {
    const last = s.data.length ? fmt(s.data[s.data.length - 1]) : '';
    const text = `${s.label} ${last}`;
    ctx.fillStyle = s.color;
    roundRect(ctx, lx, 4, 9, 9, 2); ctx.fill();
    ctx.fillStyle = '#c2cbdb'; ctx.fillText(text, lx + 14, 9);
    lx += ctx.measureText(text).width + 30;
  }
  ctx.lineWidth = 1;
}

/* ---------------- slidere: umplerea colorată + valoarea afișată ---------------- */
function paintRange(el) {
  const p = ((+el.value - +el.min) / (+el.max - +el.min)) * 100;
  el.style.setProperty('--p', p + '%');
}
document.querySelectorAll('input[type=range]').forEach(el => {
  el.addEventListener('input', () => paintRange(el));
  paintRange(el);
});

/** Leagă un slider de eticheta lui "<id>-val". */
function bindRange(id, fmt, onInput) {
  const el = $(id);
  const show = () => { $(id + '-val').textContent = fmt(+el.value); paintRange(el); };
  el.addEventListener('input', () => { show(); if (onInput) onInput(); });
  show();
  return show;
}

/* ---------------- tab-uri ---------------- */
const Tabs = {
  listeners: {},
  active: 'nn',
  onShow(name, fn) { (this.listeners[name] ||= []).push(fn); },
  show(name) {
    this.active = name;
    document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
    document.querySelectorAll('.panel').forEach(p => p.classList.toggle('active', p.id === 'tab-' + name));
    (this.listeners[name] || []).forEach(fn => fn());
  },
};
document.querySelectorAll('.tab').forEach(b => b.addEventListener('click', () => {
  history.replaceState(null, '', '#' + b.dataset.tab);
  Tabs.show(b.dataset.tab);
}));
let resizeTimer = null;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => Tabs.show(Tabs.active), 80);
});
