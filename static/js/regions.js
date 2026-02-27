/**
 * RegionsManager — manages the regions panel and their lifecycle.
 * Each region has: id, name, prompt, negative_prompt, intensity,
 *                  step_start, step_end, mask_b64
 */
export class RegionsManager {
  constructor(listId, canvasManager) {
    this._listEl  = document.getElementById(listId);
    this._canvas  = canvasManager;
    this._regions = [];
    this._seq     = 0;

    document.getElementById('btn-add-region')
      .addEventListener('click', () => this.addRegion());

    // Canvas "done" button deactivates the editing state on all cards
    document.addEventListener('canvas:done', () => this._clearMaskEditing());
  }

  // ── Public API ────────────────────────────────────────────────────────────

  addRegion(data = {}) {
    const id = 'r' + (++this._seq);
    const region = {
      id,
      name:            data.name             ?? `Region ${this._seq}`,
      prompt:          data.prompt           ?? '',
      negative_prompt: data.negative_prompt  ?? '',
      intensity:       data.intensity        ?? 1.0,
      step_start:      data.step_start       ?? 0,
      step_end:        data.step_end         ?? -1,
      mask_b64:        data.mask_b64         ?? null,
      cfg_override:    data.cfg_override     ?? null,
      noise:           data.noise            ?? 0.0,
      active:          data.active           ?? true,   // frontend-only: excludes from payload when false
    };
    this._regions.push(region);
    this._renderCard(region);
    this._hideEmpty();
    return region;
  }

  /** Total number of regions regardless of active state. */
  totalCount() { return this._regions.length; }

  /** Remove all regions and reset state (used when loading a preset). */
  clearAll() {
    if (this._canvas.activeRegionId) this._canvas.deactivate();
    this._regions.forEach(r => document.getElementById('rc-' + r.id)?.remove());
    this._regions = [];
    this._seq     = 0;
    this._showEmpty();
  }

  removeRegion(id) {
    this._regions = this._regions.filter(r => r.id !== id);
    document.getElementById('rc-' + id)?.remove();
    if (this._canvas.activeRegionId === id) this._canvas.deactivate();
    if (this._regions.length === 0) this._showEmpty();
  }

  /**
   * Returns only ACTIVE regions with up-to-date mask_b64.
   * Inactive regions are excluded from the backend payload entirely.
   * The `active` field itself is stripped — it's frontend-only state.
   */
  getRegions() {
    return this._regions
      .filter(r => r.active)
      .map(({ active, ...r }) => ({
        ...r,
        mask_b64: this._canvas.getMaskB64(r.id),
      }));
  }

  // ── Card rendering ────────────────────────────────────────────────────────

  _renderCard(region) {
    const card = document.createElement('div');
    card.id        = 'rc-' + region.id;
    card.className = 'region-card';
    card.innerHTML = `
      <div class="region-card-header">
        <label class="region-toggle" title="${region.active ? 'Active — click to disable' : 'Inactive — click to enable'}">
          <input type="checkbox" class="r-active" ${region.active ? 'checked' : ''}>
          <span class="region-toggle-track"></span>
        </label>
        <span class="region-dot"></span>
        <input class="region-name-input" type="text" value="${this._esc(region.name)}">
        <button class="btn-icon region-del" title="Delete region">✕</button>
      </div>
      <div class="region-card-body">
        <div>
          <label>Prompt</label>
          <textarea class="r-prompt" rows="2">${this._esc(region.prompt)}</textarea>
        </div>
        <div>
          <label>Negative</label>
          <input class="r-negative" type="text" value="${this._esc(region.negative_prompt)}">
        </div>
        <div>
          <label>Intensity</label>
          <div class="region-intensity-row">
            <input class="r-intensity" type="range" min="0" max="1" step="0.05"
                   value="${region.intensity}">
            <span class="region-intensity-val">${region.intensity.toFixed(2)}</span>
          </div>
        </div>
        <div>
          <label>Step range</label>
          <div class="region-steps-row">
            <input class="r-step-start" type="number" min="0" value="${region.step_start}">
            <span>→</span>
            <input class="r-step-end" type="number" min="-1" value="${region.step_end}">
            <span style="font-size:10px;color:var(--text-xdim)">(-1=end)</span>
          </div>
        </div>
        <div>
          <label>Behavior at step_end</label>
          <div class="mode-toggle">
            <input type="radio" name="mode-${region.id}" id="mode-auto-${region.id}" value="auto"
                   ${region.mode === 'auto' ? 'checked' : ''}>
            <label for="mode-auto-${region.id}">▶ Auto</label>
            <input type="radio" name="mode-${region.id}" id="mode-pause-${region.id}" value="pause"
                   ${region.mode === 'pause' ? 'checked' : ''}>
            <label for="mode-pause-${region.id}">⏸ Pause</label>
          </div>
        </div>
        <div>
          <label>CFG override <span style="color:var(--text-xdim);text-transform:none;font-weight:400">(empty = global)</span></label>
          <div class="region-cfg-row">
            <input class="r-cfg-override" type="number" min="1" max="30" step="0.5"
                   value="${region.cfg_override ?? ''}" placeholder="—">
            <span>scale</span>
          </div>
        </div>
        <div>
          <label>Noise injection</label>
          <div class="region-noise-row">
            <input class="r-noise" type="range" min="0" max="0.5" step="0.05"
                   value="${region.noise}">
            <span class="region-noise-val">${region.noise.toFixed(2)}</span>
          </div>
        </div>
        <div class="region-mask-actions">
          <button class="region-mask-btn">Edit Mask</button>
          <button class="region-preview-btn" title="Preview mask">👁 Preview</button>
        </div>
      </div>
    `;

    // ── Bind inputs ──────────────────────────────────────────────────────────
    const q  = sel => card.querySelector(sel);

    q('.region-name-input').addEventListener('input', e => {
      region.name = e.target.value;
    });

    q('.r-prompt').addEventListener('input', e => {
      region.prompt = e.target.value;
    });

    q('.r-negative').addEventListener('input', e => {
      region.negative_prompt = e.target.value;
    });

    const intensitySlider = q('.r-intensity');
    const intensityVal    = q('.region-intensity-val');
    intensitySlider.addEventListener('input', e => {
      region.intensity = parseFloat(e.target.value);
      intensityVal.textContent = region.intensity.toFixed(2);
    });

    q('.r-step-start').addEventListener('change', e => {
      region.step_start = parseInt(e.target.value) || 0;
    });
    q('.r-step-end').addEventListener('change', e => {
      region.step_end = parseInt(e.target.value) ?? -1;
    });

    q('.r-cfg-override').addEventListener('change', e => {
      const v = parseFloat(e.target.value);
      region.cfg_override = isNaN(v) ? null : v;
    });

    const noiseSlider = q('.r-noise');
    const noiseVal    = q('.region-noise-val');
    noiseSlider.addEventListener('input', e => {
      region.noise = parseFloat(e.target.value);
      noiseVal.textContent = region.noise.toFixed(2);
    });

    // Active toggle
    if (!region.active) card.classList.add('inactive');
    q('.r-active').addEventListener('change', e => {
      region.active = e.target.checked;
      card.classList.toggle('inactive', !region.active);
      q('.region-toggle').title = region.active
        ? 'Active — click to disable'
        : 'Inactive — click to enable';
    });

    // Mode toggle — Auto / Pause
    card.querySelectorAll(`input[name="mode-${region.id}"]`).forEach(radio => {
      radio.addEventListener('change', e => {
        region.mode = e.target.value;
        // Visual: highlight the card dot when pause is selected
        card.querySelector('.region-dot').style.background =
          region.mode === 'pause' ? 'var(--warn)' : '';
      });
    });
    if (region.mode === 'pause') {
      card.querySelector('.region-dot').style.background = 'var(--warn)';
    }

    // Delete
    q('.region-del').addEventListener('click', () => this.removeRegion(region.id));

    // Preview mask
    q('.region-preview-btn').addEventListener('click', () => {
      const stats = this._canvas.getMaskStats(region.id);
      const b64   = this._canvas.getMaskB64(region.id);
      window.showMaskPreview?.(b64, region.name, stats);
    });

    // Mask toggle
    const maskBtn = q('.region-mask-btn');
    maskBtn.addEventListener('click', () => {
      if (this._canvas.activeRegionId === region.id) {
        this._canvas.deactivate();
        this._clearMaskEditing();
      } else {
        // Deactivate any previously active card
        this._clearMaskEditing();
        // Activate this one
        this._canvas.activateForRegion(region.id);
        maskBtn.classList.add('editing');
        maskBtn.textContent = '✓ Done Editing';
        card.classList.add('mask-active');
      }
    });

    this._listEl.appendChild(card);
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  _clearMaskEditing() {
    document.querySelectorAll('.region-mask-btn').forEach(btn => {
      btn.classList.remove('editing');
      btn.textContent = 'Edit Mask';
    });
    document.querySelectorAll('.region-card').forEach(c => {
      c.classList.remove('mask-active');
    });
  }

  _hideEmpty() {
    this._listEl.querySelector('.regions-empty')?.remove();
  }

  _showEmpty() {
    if (!this._listEl.querySelector('.regions-empty')) {
      const div = document.createElement('div');
      div.className = 'regions-empty';
      div.innerHTML = 'No regions.<br>Add one to use MultiDiffusion.';
      this._listEl.appendChild(div);
    }
  }

  _esc(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
}
