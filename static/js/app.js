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
  scheduleTokenize();
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
    token_masks:     _getActiveTokenMasks(),
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

    // 4. Restore token masks
    _clearAllTokenMasks();
    for (const tm of (cfg.token_masks ?? [])) {
      const id = 'tok_' + tm.token_index;
      _tokenEntries[id] = {
        token_index: tm.token_index,
        text:        `token_${tm.token_index}`,
        intensity:   tm.intensity ?? 0.5,
        active:      true,
      };
      if (tm.mask_b64) {
        await canvas.setMaskFromB64(id, tm.mask_b64);
      }
    }
    if ((cfg.token_masks ?? []).length > 0) {
      _renderTokenMaskCards();
      scheduleTokenize();  // re-tokenize para obtener textos reales
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

// ── Attention Masks — Token UI ────────────────────────────────────────────────

// token_id ('tok_N') → { token_index, text, intensity, active }
const _tokenEntries = {};
let _currentTokenId  = null;
let _tokenizeTimer   = null;

function scheduleTokenize() {
  clearTimeout(_tokenizeTimer);
  _tokenizeTimer = setTimeout(_doTokenize, 500);
}

async function _doTokenize() {
  const prompt    = $('input-prompt').value.trim();
  const chipsWrap = $('token-chips-wrap');
  if (!prompt || !state.modelLoaded) {
    chipsWrap.innerHTML = '<div class="attn-empty">Load a model and type a prompt to see tokens.</div>';
    return;
  }
  try {
    const data = await fetch('/tokenize', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ prompt }),
    }).then(r => r.json());

    if (data.error || !data.tokens?.length) {
      chipsWrap.innerHTML = '<div class="attn-empty">Could not tokenize prompt.</div>';
      return;
    }
    _renderTokenChips(data.tokens);
  } catch { /* server not ready */ }
}

function _renderTokenChips(tokens) {
  const visible = tokens.filter(t => !t.special);
  const wrap    = $('token-chips-wrap');
  wrap.innerHTML = '';
  if (!visible.length) {
    wrap.innerHTML = '<div class="attn-empty">No tokens found.</div>';
    return;
  }
  for (const tok of visible) {
    const chip = document.createElement('button');
    chip.className           = 'token-chip';
    chip.dataset.tokenIndex  = tok.index;
    chip.textContent         = tok.text;
    const id = 'tok_' + tok.index;
    if (_tokenEntries[id]) chip.classList.add('has-mask');
    chip.addEventListener('click', () => _activateTokenCanvas(tok));
    wrap.appendChild(chip);
  }
  // Actualizar texto de cards existentes con texto real del token
  for (const [id, entry] of Object.entries(_tokenEntries)) {
    const chip = document.querySelector(`.token-chip[data-token-index="${entry.token_index}"]`);
    if (chip && entry.text.startsWith('token_')) {
      entry.text = chip.textContent;
      const nameEl = document.querySelector(`#tmc-${id} .tmc-token-text`);
      if (nameEl) nameEl.textContent = `"${entry.text}"`;
    }
  }
}

function _activateTokenCanvas(tok) {
  const id = 'tok_' + tok.index;
  _currentTokenId = id;

  // Desactivar cualquier región en edición
  if (canvas.activeRegionId && !canvas.activeRegionId.startsWith('tok_')) {
    canvas.deactivate();
    document.dispatchEvent(new CustomEvent('canvas:done'));
  }

  canvas.activateForRegion(id);
  const lbl = $('canvas-edit-label');
  if (lbl) lbl.textContent = `Token: "${tok.text}"`;

  // Resaltar chip seleccionado
  document.querySelectorAll('.token-chip').forEach(c => {
    c.classList.toggle('selected', parseInt(c.dataset.tokenIndex) === tok.index);
  });
}

// Cuando el canvas termina (botón Done) — guardar máscara del token activo
document.addEventListener('canvas:done', () => {
  if (!_currentTokenId) return;
  const tokenIndex = parseInt(_currentTokenId.replace('tok_', ''));
  const chip = document.querySelector(`.token-chip[data-token-index="${tokenIndex}"]`);
  const text = chip ? chip.textContent : `token_${tokenIndex}`;

  if (!_tokenEntries[_currentTokenId]) {
    _tokenEntries[_currentTokenId] = { token_index: tokenIndex, text, intensity: 0.5, active: true };
  }
  _currentTokenId = null;

  // Desmarcar chips
  document.querySelectorAll('.token-chip').forEach(c => c.classList.remove('selected'));

  _renderTokenMaskCards();
});

function _renderTokenMaskCards() {
  const listEl = $('token-masks-list');
  listEl.innerHTML = '';
  for (const [id, entry] of Object.entries(_tokenEntries)) {
    _appendTokenMaskCard(listEl, id, entry);
  }
  // Actualizar estado has-mask en chips
  document.querySelectorAll('.token-chip').forEach(chip => {
    const id = 'tok_' + chip.dataset.tokenIndex;
    chip.classList.toggle('has-mask', !!_tokenEntries[id]);
  });
}

function _appendTokenMaskCard(listEl, id, entry) {
  const maskB64 = canvas.getMaskB64(id);
  const card    = document.createElement('div');
  card.id        = 'tmc-' + id;
  card.className = 'token-mask-card' + (entry.active ? '' : ' inactive');
  card.innerHTML = `
    <div class="tmc-header">
      <label class="region-toggle" title="${entry.active ? 'Active' : 'Inactive'}">
        <input type="checkbox" class="tmc-active" ${entry.active ? 'checked' : ''}>
        <span class="region-toggle-track"></span>
      </label>
      <span class="tmc-token-text">"${_escHtml(entry.text)}"</span>
      <span class="tmc-index">#${entry.token_index}</span>
      <button class="btn-icon tmc-del" title="Remove mask">✕</button>
    </div>
    <div class="tmc-body">
      <img class="tmc-thumb" src="data:image/png;base64,${maskB64}" alt="mask" title="Click to re-edit">
      <div class="tmc-controls">
        <label>Intensity</label>
        <div class="region-intensity-row">
          <input class="tmc-intensity" type="range" min="0" max="1" step="0.05" value="${entry.intensity}">
          <span class="tmc-intensity-val">${entry.intensity.toFixed(2)}</span>
        </div>
      </div>
    </div>
  `;

  const q = sel => card.querySelector(sel);

  q('.tmc-active').addEventListener('change', e => {
    entry.active = e.target.checked;
    card.classList.toggle('inactive', !entry.active);
  });

  const slider = q('.tmc-intensity');
  const valEl  = q('.tmc-intensity-val');
  slider.addEventListener('input', e => {
    entry.intensity = parseFloat(e.target.value);
    valEl.textContent = entry.intensity.toFixed(2);
  });

  q('.tmc-del').addEventListener('click', () => {
    delete _tokenEntries[id];
    canvas.removeMask(id);
    card.remove();
    const chip = document.querySelector(`.token-chip[data-token-index="${entry.token_index}"]`);
    if (chip) chip.classList.remove('has-mask');
  });

  q('.tmc-thumb').addEventListener('click', () => {
    _activateTokenCanvas({ index: entry.token_index, text: entry.text });
  });

  listEl.appendChild(card);
}

function _getActiveTokenMasks() {
  return Object.entries(_tokenEntries)
    .filter(([, e]) => e.active)
    .map(([id, e]) => ({
      token_index: e.token_index,
      mask_b64:    canvas.getMaskB64(id),
      intensity:   e.intensity,
    }));
}

function _clearAllTokenMasks() {
  Object.keys(_tokenEntries).forEach(id => canvas.removeMask(id));
  Object.keys(_tokenEntries).forEach(k => delete _tokenEntries[k]);
  $('token-masks-list').innerHTML = '';
  document.querySelectorAll('.token-chip').forEach(c => c.classList.remove('has-mask', 'selected'));
}

function _escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// Conectar prompt input → tokenize con debounce
$('input-prompt').addEventListener('input', scheduleTokenize);

// ── Init ──────────────────────────────────────────────────────────────────────
loadConfigFromServer();
loadModelList();
loadPresetList();
startStatsPolling();
