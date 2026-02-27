/**
 * app.js — Main entry point
 */

import { WsManager }     from './ws.js';
import { CanvasManager } from './canvas_manager.js';
import { RegionsManager } from './regions.js';

// ── Managers ──────────────────────────────────────────────────────────────────
const ws      = new WsManager(`ws://${location.host}/ws`);
const canvas  = new CanvasManager('main-image', 'overlay-canvas');
const regions = new RegionsManager('regions-list', canvas);

// ── Client state ──────────────────────────────────────────────────────────────
const state = {
  isGenerating: false,
  isPaused:     false,
  modelLoaded:  false,
  lastSeed:     null,
  config:       {},
};

let _errorTimer = null;
let _statsTimer = null;

// ── DOM helpers ───────────────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const on = (id, ev, fn) => $(id).addEventListener(ev, fn);

// ── WS handlers ───────────────────────────────────────────────────────────────

ws.on('status', ({ status }) => {
  if (status === 'generating' || status === 'loading_model') return; // handled elsewhere
  setStatus(status);
});

ws.on('step', ({ step, total, image, seed }) => {
  canvas.setImage(image);
  setStatus('generating');
  updateProgress(step, total);
  addThumbnail(image, step);
  state.lastSeed = seed;
  state.isPaused = false;
});

ws.on('complete', ({ image, seed }) => {
  canvas.setImage(image);
  setStatus('idle');
  clearProgress();
  state.lastSeed = seed;
  $('seed-display').textContent   = `seed ${seed}`;
  $('last-seed-info').textContent = `Last seed: ${seed}`;
  $('input-seed').value = seed;   // auto-fill seed field
  markLastThumb();
});

ws.on('paused', ({ step, total, region_name }) => {
  state.isPaused = true;
  setStatus('paused');
  updateProgress(step, total);
  $('btn-resume').classList.remove('hidden');
  $('btn-stop').disabled = false;
  showError(`Paused after step ${step}/${total} — region "${region_name}". Press Resume to continue.`, 'warn');
});

ws.on('model_loaded', ({ path }) => {
  state.modelLoaded = true;
  const name = path.split(/[\\/]/).pop();
  $('model-loaded-badge').classList.remove('hidden');
  $('model-status-text').textContent = `✓ ${name}`;
  $('btn-generate').disabled = false;
  $('btn-generate').title = '';
  dismissError();
});

ws.on('model_unloaded', () => {
  state.modelLoaded = false;
  $('model-loaded-badge').classList.add('hidden');
  $('model-status-text').textContent = '';
  $('btn-generate').disabled = true;
  $('btn-generate').title = 'Load a model first';
});

ws.on('error', ({ message }) => {
  setStatus('error');
  clearProgress();
  showError(message);
});

// ── Buttons ───────────────────────────────────────────────────────────────────

on('btn-generate', 'click', async () => {
  if (state.isGenerating) return;

  syncConfigFromUI();
  await fetch('/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(state.config),
  });
  const activeRegions = regions.getRegions();
  console.log(`[generate] sending ${activeRegions.length} active region(s) of ${regions.totalCount()} total`);
  await fetch('/regions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(activeRegions),
  });

  $('step-strip').innerHTML = '';
  dismissError();

  const res  = await fetch('/generate/start', { method: 'POST' });
  const data = await res.json();
  if (!data.ok) {
    showError(data.error);
    return;
  }
  setStatus('loading_model');
});

on('btn-stop', 'click', async () => {
  await fetch('/generate/stop', { method: 'POST' });
  $('btn-resume').classList.add('hidden');
  state.isPaused = false;
});

on('btn-resume', 'click', async () => {
  const res  = await fetch('/generate/resume', { method: 'POST' });
  const data = await res.json();
  if (data.ok) {
    $('btn-resume').classList.add('hidden');
    state.isPaused = false;
    dismissError();
  } else {
    showError(data.error);
  }
});

on('btn-panic', 'click', async () => {
  if (!confirm('Panic: interrupt generation and free VRAM immediately?')) return;
  await fetch('/generate/panic', { method: 'POST' });
  setStatus('idle');
  clearProgress();
  $('btn-resume').classList.add('hidden');
  state.isPaused = false;
});

on('btn-load-model', 'click', async () => {
  // Sync model path from UI before loading
  app_state_config_model_path_sync();
  $('model-status-text').textContent = 'loading…';
  $('btn-load-model').disabled = true;

  const cfgRes = await fetch('/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...state.config, model_path: $('input-model').value.trim() }),
  });

  const res  = await fetch('/model/load', { method: 'POST' });
  const data = await res.json();
  $('btn-load-model').disabled = false;
  if (!data.ok) {
    $('model-status-text').textContent = '';
    showError(data.error);
  }
  // Success state is handled by model_loaded WS message
});

// Model picker dropdown → populate text field
on('model-picker', 'change', () => {
  const val = $('model-picker').value;
  if (val) $('input-model').value = val;
});

on('btn-rand-seed', 'click', () => {
  $('input-seed').value = Math.floor(Math.random() * 2 ** 32);
});

on('btn-dismiss-error', 'click', dismissError);

// ── Range live display ────────────────────────────────────────────────────────

function bindRange(inputId, valId, parse = parseFloat) {
  $(inputId).addEventListener('input', e => {
    $(valId).textContent = parse(e.target.value);
  });
}
bindRange('input-steps', 'val-steps', parseInt);
bindRange('input-cfg',   'val-cfg',   parseFloat);

// ── Status ────────────────────────────────────────────────────────────────────

const STATUS_LABELS = {
  idle:          'idle',
  loading_model: 'loading model…',
  generating:    'generating',
  paused:        'paused',
  error:         'error',
};

function setStatus(status) {
  const active = ['loading_model', 'generating'].includes(status);
  const paused = status === 'paused';
  state.isGenerating = active || paused;

  const dot = $('status-dot');
  if (active)       dot.className = 'active';
  else if (paused)  dot.className = 'warn';
  else if (status === 'error') dot.className = 'error';
  else              dot.className = 'idle';

  $('status-label').textContent = STATUS_LABELS[status] ?? status;

  $('btn-generate').disabled = state.isGenerating || !state.modelLoaded;
  $('btn-stop').disabled     = !state.isGenerating || paused;
}

function updateProgress(step, total) {
  $('progress-bar').style.width = `${(step / total) * 100}%`;
  $('progress-text').textContent = `${step} / ${total}`;
}

function clearProgress() {
  $('progress-bar').style.width = '0%';
  $('progress-text').textContent = '';
}

// ── Error banner ──────────────────────────────────────────────────────────────

function showError(msg, level = 'error') {
  clearTimeout(_errorTimer);
  const banner = $('error-banner');
  $('error-text').textContent = msg;
  banner.classList.remove('hidden');
  banner.dataset.level = level;
  const timeout = level === 'success' ? 3_000 : level === 'warn' ? 0 : 10_000;
  if (timeout > 0) _errorTimer = setTimeout(dismissError, timeout);
}

const showSuccess = msg => showError(msg, 'success');

function dismissError() {
  clearTimeout(_errorTimer);
  $('error-banner').classList.add('hidden');
}

// ── Thumbnails ────────────────────────────────────────────────────────────────

function addThumbnail(b64, step) {
  const strip = $('step-strip');
  strip.querySelectorAll('.latest').forEach(t => t.classList.remove('latest'));
  const img = document.createElement('img');
  img.className = 'step-thumb latest';
  img.src       = 'data:image/jpeg;base64,' + b64;
  img.title     = `Step ${step}`;
  img.addEventListener('click', () => canvas.setImage(b64));
  strip.appendChild(img);
  strip.scrollLeft = strip.scrollWidth;
}

function markLastThumb() {
  const thumbs = $('step-strip').querySelectorAll('.step-thumb');
  thumbs.forEach(t => t.classList.remove('latest'));
  thumbs[thumbs.length - 1]?.classList.add('latest');
}

// ── GPU / RAM stats polling ───────────────────────────────────────────────────

async function fetchStats() {
  try {
    const s = await fetch('/stats').then(r => r.json());
    console.debug(
      `[stats] reserved=${s.vram_used} allocated=${s.vram_allocated} ` +
      `peak=${s.vram_peak} total=${s.vram_total} GB`
    );
    if (s.vram_total > 0) {
      // Show reserved/total — reserved is what the CUDA allocator holds (model + runtime)
      // allocated is shown in the title tooltip for detail
      $('stat-vram').textContent = `${s.vram_used.toFixed(1)} / ${s.vram_total} GB`;
      $('stat-vram').title = `reserved: ${s.vram_used} GB  |  allocated: ${s.vram_allocated} GB  |  peak: ${s.vram_peak} GB`;
    } else {
      $('stat-vram').textContent = '— / — GB';
      $('stat-vram').title = '';
    }
    $('stat-ram').textContent = s.ram_total > 0
      ? `${s.ram_used} / ${s.ram_total} GB`
      : '0 GB';
  } catch { /* server not ready */ }
}

function startStatsPolling() {
  fetchStats();
  _statsTimer = setInterval(fetchStats, 2000);
}

// ── Model library ─────────────────────────────────────────────────────────────

async function loadModelList() {
  try {
    const data = await fetch('/models/list').then(r => r.json());
    const picker = $('model-picker');
    picker.innerHTML = '<option value="">— browse library —</option>';
    for (const p of data.models) {
      const name = p.split(/[\\/]/).pop();
      const opt  = document.createElement('option');
      opt.value       = p;
      opt.textContent = name;
      picker.appendChild(opt);
    }
    if (!data.exists) {
      const opt = document.createElement('option');
      opt.disabled    = true;
      opt.textContent = `(folder not found: ${data.base_path})`;
      picker.appendChild(opt);
    }
  } catch { /* ignore */ }
}

// ── Config sync ───────────────────────────────────────────────────────────────

function syncConfigFromUI() {
  state.config = {
    prompt:          $('input-prompt').value,
    negative_prompt: $('input-negative').value,
    steps:           parseInt($('input-steps').value),
    cfg_scale:       parseFloat($('input-cfg').value),
    seed:            parseInt($('input-seed').value),
    width:           parseInt($('input-width').value),
    height:          parseInt($('input-height').value),
    model_path:      $('input-model').value.trim(),
    scheduler:       $('input-scheduler').value,
  };
}

function app_state_config_model_path_sync() {
  state.config.model_path = $('input-model').value.trim();
}

async function loadConfigFromServer() {
  try {
    const cfg = await fetch('/config').then(r => r.json());
    $('input-prompt').value    = cfg.prompt           ?? '';
    $('input-negative').value  = cfg.negative_prompt  ?? '';
    $('input-steps').value     = cfg.steps;
    $('val-steps').textContent = cfg.steps;
    $('input-cfg').value       = cfg.cfg_scale;
    $('val-cfg').textContent   = cfg.cfg_scale;
    $('input-seed').value      = cfg.seed;
    $('input-width').value     = cfg.width;
    $('input-height').value    = cfg.height;
    $('input-model').value     = cfg.model_path;
    $('input-scheduler').value = cfg.scheduler;
    Object.assign(state.config, cfg);
  } catch { /* ignore */ }
}

// ── Canvas overlay opacity ────────────────────────────────────────────────────

on('overlay-opacity', 'input', e => {
  const pct = parseInt(e.target.value);
  $('overlay-opacity-val').textContent = pct + '%';
  $('overlay-canvas').style.opacity = pct / 100;
});

// ── Gallery ───────────────────────────────────────────────────────────────────

function openGallery() {
  $('gallery-modal').classList.remove('hidden');
  fetchGallery();
}

function closeGallery() {
  $('gallery-modal').classList.add('hidden');
  $('gallery-preview-img').src     = '';
  $('gallery-preview-img').style.display = 'none';
  $('gallery-preview-empty').style.display = 'block';
  $('gallery-preview-info').textContent    = '';
}

async function fetchGallery() {
  const grid = $('gallery-grid');
  grid.innerHTML = '<span style="color:var(--text-xdim);font-size:11px;grid-column:1/-1">Loading…</span>';
  try {
    const res  = await fetch('/outputs/list');
    const data = await res.json();
    console.log('[gallery] /outputs/list →', res.status, data);
    grid.innerHTML = '';
    if (!data.outputs.length) {
      grid.innerHTML = '<span style="color:var(--text-xdim);font-size:11px;grid-column:1/-1">No outputs yet</span>';
      return;
    }
    for (const item of data.outputs) {
      const img = document.createElement('img');
      img.className = 'gallery-thumb';
      img.src       = `/outputs/${item.filename}`;
      img.title     = item.prompt ? item.prompt.slice(0, 80) : item.filename;
      img.addEventListener('click', () => selectGalleryItem(img, item));
      grid.appendChild(img);
    }
  } catch (err) {
    console.error('[gallery] fetch failed:', err);
    grid.innerHTML = '<span style="color:var(--danger);font-size:11px;grid-column:1/-1">Failed to load</span>';
  }
}

function selectGalleryItem(thumbEl, item) {
  $('gallery-grid').querySelectorAll('.gallery-thumb').forEach(t => t.classList.remove('selected'));
  thumbEl.classList.add('selected');

  const previewImg = $('gallery-preview-img');
  previewImg.src          = `/outputs/${item.filename}`;
  previewImg.style.display = 'block';
  $('gallery-preview-empty').style.display = 'none';

  const info = $('gallery-preview-info');
  const truncate = (s, n) => s && s.length > n ? s.slice(0, n) + '…' : (s || '');
  info.innerHTML =
    (item.prompt ? `<b>Prompt:</b> ${truncate(item.prompt, 120)}<br>` : '') +
    (item.seed   != null ? `<b>Seed:</b> ${item.seed}  ` : '') +
    (item.steps  != null ? `<b>Steps:</b> ${item.steps}  ` : '') +
    (item.cfg_scale != null ? `<b>CFG:</b> ${item.cfg_scale}  ` : '') +
    (item.model  ? `<b>Model:</b> ${item.model}` : '');
}

on('btn-gallery',         'click', openGallery);
on('btn-close-gallery',   'click', closeGallery);
on('btn-refresh-gallery', 'click', fetchGallery);
on('gallery-backdrop',    'click', closeGallery);

// ── Mask preview modal ────────────────────────────────────────────────────────

function showMaskPreview(b64, regionName, stats) {
  $('modal-title').textContent   = `Mask — ${regionName}`;
  $('modal-mask-img').src        = 'data:image/png;base64,' + b64;

  const statsEl = $('modal-mask-stats');
  if (stats.exists) {
    statsEl.innerHTML =
      `<span>${stats.size} px</span>` +
      `<span>coverage <b>${stats.coverage}</b></span>` +
      `<span>min <b>${stats.min}</b></span>` +
      `<span>max <b>${stats.max}</b></span>` +
      `<span>mean <b>${stats.mean}</b></span>`;
  } else {
    statsEl.innerHTML = '<span>No mask drawn — will use full white (100% coverage)</span>';
  }

  $('mask-preview-modal').classList.remove('hidden');
}

function closeMaskPreview() {
  $('mask-preview-modal').classList.add('hidden');
  $('modal-mask-img').src = '';
}

on('btn-close-modal', 'click', closeMaskPreview);
on('modal-backdrop',  'click', closeMaskPreview);
document.addEventListener('keydown', e => {
  if (e.key !== 'Escape') return;
  if (!$('gallery-modal').classList.contains('hidden'))      closeGallery();
  else if (!$('mask-preview-modal').classList.contains('hidden')) closeMaskPreview();
});

// Make showMaskPreview accessible to regions.js (same module scope via window)
window.showMaskPreview = showMaskPreview;

// ── Presets ───────────────────────────────────────────────────────────────────

async function loadPresetList() {
  const listEl = $('preset-list');
  try {
    const data = await fetch('/presets/list').then(r => r.json());
    listEl.innerHTML = '';
    if (!data.presets.length) {
      listEl.innerHTML = '<div class="preset-empty">No presets saved yet</div>';
      return;
    }
    for (const p of data.presets) {
      const item = document.createElement('div');
      item.className = 'preset-item';
      item.innerHTML =
        `<span class="preset-item-name" title="${p.name}">${p.name}</span>` +
        `<span class="preset-item-date">${p.modified_at.slice(5, 16)}</span>` +
        `<button class="btn-preset-load">Load</button>` +
        `<button class="btn-preset-del" title="Delete preset">✕</button>`;
      item.querySelector('.btn-preset-load').addEventListener('click', () => applyPreset(p.name));
      item.querySelector('.btn-preset-del').addEventListener('click', () => confirmDeletePreset(p.name, item));
      listEl.appendChild(item);
    }
  } catch (err) {
    console.error('[presets] loadPresetList failed:', err);
    listEl.innerHTML = '<div class="preset-empty" style="color:var(--danger)">Error loading presets</div>';
  }
}

async function savePreset() {
  const name = $('preset-name').value.trim();
  if (!name) { showError('Enter a preset name before saving.'); return; }

  syncConfigFromUI();
  const activeRegions = regions.getRegions();  // already filtered active, mask_b64 included

  try {
    const res  = await fetch('/presets/save', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ name, config: state.config, regions: activeRegions }),
    });
    const data = await res.json();
    if (!data.ok) { showError(data.detail ?? 'Save failed'); return; }
    showSuccess(`Preset guardado: "${name}"`);
    loadPresetList();
  } catch (err) {
    showError('Error saving preset: ' + err.message);
  }
}

async function applyPreset(name) {
  try {
    const data = await fetch(`/presets/load/${encodeURIComponent(name)}`).then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    });

    // 1. Restore global config fields
    const cfg = data.config ?? {};
    if (cfg.prompt          != null) $('input-prompt').value    = cfg.prompt;
    if (cfg.negative_prompt != null) $('input-negative').value  = cfg.negative_prompt;
    if (cfg.steps           != null) { $('input-steps').value   = cfg.steps;     $('val-steps').textContent = cfg.steps; }
    if (cfg.cfg_scale       != null) { $('input-cfg').value     = cfg.cfg_scale; $('val-cfg').textContent   = cfg.cfg_scale; }
    if (cfg.seed            != null) $('input-seed').value      = cfg.seed;
    if (cfg.width           != null) $('input-width').value     = cfg.width;
    if (cfg.height          != null) $('input-height').value    = cfg.height;
    if (cfg.scheduler       != null) $('input-scheduler').value = cfg.scheduler;
    if (cfg.model_path      != null) $('input-model').value     = cfg.model_path;
    Object.assign(state.config, cfg);

    // 2. Clear existing regions, restore from preset
    regions.clearAll();
    for (const r of (data.regions ?? [])) {
      const newRegion = regions.addRegion(r);
      // 3. Restore mask if present
      if (r.mask_b64) {
        await canvas.setMaskFromB64(newRegion.id, r.mask_b64);
      }
    }

    showSuccess(`Preset cargado: "${name}"`);
  } catch (err) {
    showError('Error loading preset: ' + err.message);
  }
}

async function confirmDeletePreset(name, itemEl) {
  if (!confirm(`Delete preset "${name}"?`)) return;
  try {
    const res  = await fetch(`/presets/delete/${encodeURIComponent(name)}`, { method: 'DELETE' });
    const data = await res.json();
    if (data.ok) { itemEl.remove(); }
    else         { showError(data.detail ?? 'Delete failed'); }
  } catch (err) {
    showError('Error deleting preset: ' + err.message);
  }
}

on('btn-save-preset', 'click', savePreset);

// ── Init ──────────────────────────────────────────────────────────────────────
loadConfigFromServer();
loadModelList();
loadPresetList();
startStatsPolling();
