/**
 * CanvasManager
 * Manages the overlay canvas on top of the generation viewer.
 *
 * - One canvas per session; stores per-region mask data in memory.
 * - Mask is stored as ImageData (canvas resolution = generation resolution).
 * - Visual overlay draws the mask as a semi-transparent orange fill.
 * - getMaskB64(regionId) returns a grayscale PNG for the backend.
 */
export class CanvasManager {
  constructor(imgId, canvasId) {
    this._img    = document.getElementById(imgId);
    this._canvas = document.getElementById(canvasId);
    this._ctx    = this._canvas.getContext('2d');

    // State
    this.activeRegionId = null;
    this._isDrawing     = false;
    this._drawMode      = 'draw';   // 'draw' | 'erase'
    this._brushSize     = 20;

    // Per-region mask storage: regionId → Float32Array (w*h, values 0-1)
    // We store opacity values rather than ImageData to be resolution-independent.
    this._masks = {};

    // Track current generation resolution
    this._genW = 512;
    this._genH = 512;

    this._bindToolbar();
    this._bindPointer();
    this._observeImage();
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /** Update the viewer image. Also syncs canvas size to match. */
  setImage(b64, genW, genH) {
    if (genW) this._genW = genW;
    if (genH) this._genH = genH;

    this._img.style.display = 'block';
    this._img.src = 'data:image/jpeg;base64,' + b64;
    document.getElementById('viewer-placeholder').classList.add('hidden');
  }

  /** Activate canvas for editing a specific region. */
  activateForRegion(regionId) {
    this._saveCurrentMask();
    this.activeRegionId = regionId;
    this._canvas.classList.add('active');
    document.getElementById('canvas-toolbar').classList.remove('hidden');
    this._syncCanvasToImage();
    this._renderMask();
  }

  /** Deactivate canvas (done editing). */
  deactivate() {
    this._saveCurrentMask();
    this.activeRegionId = null;
    this._canvas.classList.remove('active');
    document.getElementById('canvas-toolbar').classList.add('hidden');
    const lbl = document.getElementById('canvas-edit-label');
    if (lbl) lbl.textContent = '';
    this._ctx.clearRect(0, 0, this._canvas.width, this._canvas.height);
  }

  /** Remove the stored mask for a given id (region or token). */
  removeMask(id) {
    delete this._masks[id];
  }

  /** Clear the mask of the active region. */
  clearMask() {
    if (!this.activeRegionId) return;
    this._masks[this.activeRegionId] = null;
    this._ctx.clearRect(0, 0, this._canvas.width, this._canvas.height);
  }

  /** Fill the entire canvas with the mask. */
  fillMask() {
    if (!this.activeRegionId) return;
    const { width: w, height: h } = this._canvas;
    const data = new Float32Array(w * h).fill(1);
    this._masks[this.activeRegionId] = { data, w, h };
    this._renderMask();
  }

  /**
   * Returns a base64-encoded grayscale PNG of the region mask,
   * always scaled to the current generation resolution read from the UI.
   * The canvas display size is irrelevant — export is always at gen resolution.
   */
  getMaskB64(regionId) {
    const mask = this._masks[regionId];
    // Always read from generation config inputs — never use stale canvas size
    const w = parseInt(document.getElementById('input-width')?.value)  || 512;
    const h = parseInt(document.getElementById('input-height')?.value) || 512;

    const offscreen = document.createElement('canvas');
    offscreen.width  = w;
    offscreen.height = h;
    const ctx = offscreen.getContext('2d');

    if (mask) {
      // Blit mask data scaled to generation resolution
      const src = document.createElement('canvas');
      src.width  = mask.w;
      src.height = mask.h;
      const sctx = src.getContext('2d');
      const id = sctx.createImageData(mask.w, mask.h);
      for (let i = 0; i < mask.data.length; i++) {
        const v = Math.round(mask.data[i] * 255);
        id.data[i * 4]     = v;
        id.data[i * 4 + 1] = v;
        id.data[i * 4 + 2] = v;
        id.data[i * 4 + 3] = 255;
      }
      sctx.putImageData(id, 0, 0);
      ctx.drawImage(src, 0, 0, w, h);
    } else {
      // No mask → full white (full effect everywhere)
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, w, h);
    }

    return offscreen.toDataURL('image/png').split(',')[1];
  }

  /**
   * Restores a mask from a base64 PNG (e.g. when loading a preset).
   * Decodes the PNG into a Float32Array using the red channel.
   * Returns a Promise that resolves when the mask is stored.
   */
  setMaskFromB64(regionId, b64) {
    return new Promise(resolve => {
      const img = new Image();
      img.onload = () => {
        const c   = document.createElement('canvas');
        c.width   = img.width;
        c.height  = img.height;
        const ctx = c.getContext('2d');
        ctx.drawImage(img, 0, 0);
        const id   = ctx.getImageData(0, 0, img.width, img.height);
        const data = new Float32Array(img.width * img.height);
        for (let i = 0; i < data.length; i++) {
          data[i] = id.data[i * 4] / 255;   // red channel of grayscale PNG
        }
        this._masks[regionId] = { data, w: img.width, h: img.height };
        resolve();
      };
      img.onerror = () => resolve();  // don't block if image fails
      img.src = 'data:image/png;base64,' + b64;
    });
  }

  setBrushSize(sz) { this._brushSize = sz; }
  setDrawMode(mode) { this._drawMode = mode; }

  /**
   * Returns stats about the mask Float32Array for a region.
   * Logs to console and returns the object.
   */
  getMaskStats(regionId) {
    const mask = this._masks[regionId];
    if (!mask) {
      const s = { exists: false, regionId };
      console.debug('[mask stats]', regionId, '→ no mask drawn (will be full white)');
      return s;
    }
    const { data, w, h } = mask;
    let min = Infinity, max = -Infinity, sum = 0, nonZero = 0;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
      if (v > 0.01) nonZero++;
    }
    const genW = parseInt(document.getElementById('input-width')?.value)  || 512;
    const genH = parseInt(document.getElementById('input-height')?.value) || 512;
    const stats = {
      exists:      true,
      regionId,
      canvas_size: `${w} × ${h}`,
      export_size: `${genW} × ${genH}`,
      pixels:      data.length,
      min:      +min.toFixed(4),
      max:      +max.toFixed(4),
      mean:     +(sum / data.length).toFixed(4),
      nonZero,
      coverage: ((nonZero / data.length) * 100).toFixed(1) + '%',
    };
    console.debug('[mask stats]', regionId, stats);
    return stats;
  }

  // ── Canvas sync ───────────────────────────────────────────────────────────

  _observeImage() {
    const ro = new ResizeObserver(() => {
      if (this.activeRegionId) this._syncCanvasToImage();
    });
    ro.observe(this._img);
  }

  _syncCanvasToImage() {
    const rect = this._img.getBoundingClientRect();
    if (rect.width === 0) return;

    // Canvas buffer = current display size (for smooth drawing)
    if (this._canvas.width !== Math.round(rect.width) ||
        this._canvas.height !== Math.round(rect.height)) {
      this._canvas.width  = Math.round(rect.width);
      this._canvas.height = Math.round(rect.height);
      this._renderMask();
    }

    // Position canvas exactly over the image
    const wrap = this._img.parentElement.getBoundingClientRect();
    this._canvas.style.left   = (rect.left - wrap.left) + 'px';
    this._canvas.style.top    = (rect.top  - wrap.top)  + 'px';
    this._canvas.style.width  = rect.width  + 'px';
    this._canvas.style.height = rect.height + 'px';
  }

  // ── Mask save / render ────────────────────────────────────────────────────

  _saveCurrentMask() {
    if (!this.activeRegionId) return;
    const { width: w, height: h } = this._canvas;
    const id = this._ctx.getImageData(0, 0, w, h);
    const data = new Float32Array(w * h);
    for (let i = 0; i < data.length; i++) {
      data[i] = id.data[i * 4 + 3] / 255;   // use alpha channel as mask
    }
    this._masks[this.activeRegionId] = { data, w, h };
  }

  _renderMask() {
    const { width: w, height: h } = this._canvas;
    this._ctx.clearRect(0, 0, w, h);
    if (!this.activeRegionId) return;
    const mask = this._masks[this.activeRegionId];
    if (!mask) return;

    // Scale stored mask (mask.w × mask.h) → current canvas size
    const src = document.createElement('canvas');
    src.width  = mask.w;
    src.height = mask.h;
    const sctx = src.getContext('2d');
    const id = sctx.createImageData(mask.w, mask.h);
    for (let i = 0; i < mask.data.length; i++) {
      const a = Math.round(mask.data[i] * 180);  // semi-transparent orange
      id.data[i * 4]     = 255;
      id.data[i * 4 + 1] = 100;
      id.data[i * 4 + 2] = 30;
      id.data[i * 4 + 3] = a;
    }
    sctx.putImageData(id, 0, 0);
    this._ctx.drawImage(src, 0, 0, w, h);
  }

  // ── Drawing ───────────────────────────────────────────────────────────────

  _draw(x, y) {
    const r = this._brushSize / 2;
    this._ctx.save();
    if (this._drawMode === 'erase') {
      this._ctx.globalCompositeOperation = 'destination-out';
      this._ctx.fillStyle = 'rgba(0,0,0,1)';
    } else {
      this._ctx.globalCompositeOperation = 'source-over';
      this._ctx.fillStyle = 'rgba(255,100,30,0.65)';
    }
    this._ctx.beginPath();
    this._ctx.arc(x, y, r, 0, Math.PI * 2);
    this._ctx.fill();
    this._ctx.restore();
  }

  _canvasXY(e) {
    const rect = this._canvas.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  }

  // ── Event binding ─────────────────────────────────────────────────────────

  _bindPointer() {
    const c = this._canvas;
    let rightDown = false;
    let savedMode = null;

    c.addEventListener('contextmenu', e => e.preventDefault());

    c.addEventListener('mousedown', e => {
      if (!this.activeRegionId) return;
      if (e.button === 2) {
        savedMode = this._drawMode;
        this._drawMode = 'erase';
        rightDown = true;
      }
      this._isDrawing = true;
      const [x, y] = this._canvasXY(e);
      this._draw(x, y);
    });

    c.addEventListener('mousemove', e => {
      if (!this._isDrawing || !this.activeRegionId) return;
      const [x, y] = this._canvasXY(e);
      this._draw(x, y);
    });

    const endDraw = (e) => {
      if (this._isDrawing) this._saveCurrentMask();
      this._isDrawing = false;
      if (rightDown) {
        this._drawMode = savedMode;
        rightDown = false;
        savedMode = null;
      }
    };

    c.addEventListener('mouseup', endDraw);
    c.addEventListener('mouseleave', endDraw);
  }

  _bindToolbar() {
    const btn = id => document.getElementById(id);

    btn('btn-tool-draw').addEventListener('click', () => {
      this._drawMode = 'draw';
      btn('btn-tool-draw').classList.add('active');
      btn('btn-tool-erase').classList.remove('active');
    });

    btn('btn-tool-erase').addEventListener('click', () => {
      this._drawMode = 'erase';
      btn('btn-tool-erase').classList.add('active');
      btn('btn-tool-draw').classList.remove('active');
    });

    btn('brush-size').addEventListener('input', e => {
      this._brushSize = +e.target.value;
      btn('brush-size-val').textContent = e.target.value;
    });

    btn('btn-mask-clear').addEventListener('click', () => this.clearMask());
    btn('btn-mask-fill').addEventListener('click',  () => this.fillMask());

    btn('btn-canvas-done').addEventListener('click', () => {
      this.deactivate();
      // Notify RegionsManager that editing stopped
      document.dispatchEvent(new CustomEvent('canvas:done'));
    });
  }
}
