import asyncio
import base64
import json
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.state import AppState, GenerationConfig, Region
from core.pipeline import PipelineWorker
from core.model_loader import load_pipeline

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_MODELS_DIR = Path(r"C:\Users\aguse\Documents\ComfyUI\models\checkpoints")
DEBUG_MASKS_DIR    = Path("debug/masks")
OUTPUTS_DIR        = Path("outputs")
PRESETS_DIR        = Path("presets")


def _save_debug_mask(region_id: str, mask_b64: str) -> Path:
    """Guarda la máscara como PNG en debug/masks/ y retorna la ruta."""
    DEBUG_MASKS_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = DEBUG_MASKS_DIR / f"region_{region_id}_{ts}.png"
    path.write_bytes(base64.b64decode(mask_b64))
    return path

# ── Shared state ───────────────────────────────────────────────────────────────
app_state = AppState()
active_ws: list[WebSocket] = []
worker: Optional[PipelineWorker] = None
_model_loading = False   # guard against concurrent /model/load calls

_loop: asyncio.AbstractEventLoop = None
_broadcast_queue: asyncio.Queue = None


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loop, _broadcast_queue
    _loop = asyncio.get_running_loop()
    _broadcast_queue = asyncio.Queue()
    OUTPUTS_DIR.mkdir(exist_ok=True)
    DEBUG_MASKS_DIR.mkdir(parents=True, exist_ok=True)
    PRESETS_DIR.mkdir(exist_ok=True)
    asyncio.create_task(_broadcaster())
    yield
    if worker and worker.is_running():
        worker.panic()


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="MultiDiff Controller", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Broadcaster ────────────────────────────────────────────────────────────────
async def _broadcaster():
    while True:
        msg = await _broadcast_queue.get()
        dead = []
        for ws in list(active_ws):
            try:
                await ws.send_text(json.dumps(msg))
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in active_ws:
                active_ws.remove(ws)


def enqueue_step(data: dict) -> None:
    """Thread-safe broadcast to all WebSocket clients."""
    _loop.call_soon_threadsafe(_broadcast_queue.put_nowait, data)


# ── HTTP Routes ────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/config")
async def get_config():
    return app_state.config.model_dump()


@app.post("/config")
async def set_config(config: GenerationConfig):
    with app_state.lock:
        app_state.config = config
    return {"ok": True}


@app.get("/regions")
async def get_regions():
    return [r.model_dump() for r in app_state.regions]


@app.post("/regions")
async def set_regions(regions: List[Region]):
    with app_state.lock:
        app_state.regions = regions
    return {"ok": True}


# ── Model management ───────────────────────────────────────────────────────────
@app.get("/models/list")
async def list_models():
    """List .safetensors and .ckpt files in the default checkpoints folder."""
    if not DEFAULT_MODELS_DIR.exists():
        return {"models": [], "base_path": str(DEFAULT_MODELS_DIR), "exists": False}

    models = sorted(
        p for ext in ("*.safetensors", "*.ckpt")
        for p in DEFAULT_MODELS_DIR.glob(ext)
    )
    return {
        "models": [str(p) for p in models],
        "base_path": str(DEFAULT_MODELS_DIR),
        "exists": True,
    }


@app.post("/model/load")
async def load_model():
    global _model_loading
    if _model_loading:
        return {"ok": False, "error": "already loading a model"}
    if worker and worker.is_running():
        return {"ok": False, "error": "generation in progress — stop it first"}

    model_path = app_state.config.model_path.strip()
    if not model_path:
        return {"ok": False, "error": "model path is empty"}

    scheduler = app_state.config.scheduler

    def _load():
        global _model_loading
        _model_loading = True
        enqueue_step({"type": "status", "status": "loading_model"})
        try:
            with app_state.model_cache.lock:
                app_state.model_cache.unload()
                pipe, dtype, device, is_xl = load_pipeline(model_path, scheduler)
                app_state.model_cache.pipe       = pipe
                app_state.model_cache.dtype      = dtype
                app_state.model_cache.device     = device
                app_state.model_cache.model_path = model_path
                app_state.model_cache.is_xl      = is_xl
            enqueue_step({"type": "model_loaded", "path": model_path})
        except Exception as exc:
            enqueue_step({"type": "error", "message": f"Model load failed: {exc}"})
        finally:
            _model_loading = False
            enqueue_step({"type": "status", "status": "idle"})

    threading.Thread(target=_load, daemon=True).start()
    return {"ok": True}


@app.post("/model/unload")
async def unload_model():
    if worker and worker.is_running():
        return {"ok": False, "error": "generation in progress"}
    with app_state.model_cache.lock:
        app_state.model_cache.unload()
    enqueue_step({"type": "model_unloaded"})
    return {"ok": True}


# ── Generation control ─────────────────────────────────────────────────────────
@app.post("/generate/start")
async def start_generation():
    global worker

    if worker and worker.is_running():
        return {"ok": False, "error": "already running"}

    with app_state.lock:
        config  = app_state.config.model_copy()
        regions = list(app_state.regions)

    # Require model to be pre-loaded
    if not app_state.model_cache.is_loaded_for(config.model_path):
        loaded_path = app_state.model_cache.model_path
        if loaded_path:
            return {
                "ok": False,
                "error": f'Model path changed. Loaded: "{Path(loaded_path).name}". '
                         f'Click "Load Model" to reload.',
            }
        return {"ok": False, "error": 'No model loaded. Click "Load Model" first.'}

    # Save masks for debug inspection
    for region in regions:
        if region.mask_b64:
            p = _save_debug_mask(region.id, region.mask_b64)
            print(f"[debug] mask saved → {p}")

    worker = PipelineWorker(
        config, regions, on_step=enqueue_step, model_cache=app_state.model_cache
    )
    worker.start()
    return {"ok": True}


@app.post("/generate/stop")
async def stop_generation():
    if worker:
        worker.stop()
    return {"ok": True}


@app.post("/generate/resume")
async def resume_generation():
    if worker and worker.is_running():
        worker.resume()
        return {"ok": True}
    return {"ok": False, "error": "not running or not paused"}


@app.post("/generate/panic")
async def panic():
    global worker
    if worker:
        worker.panic()
        worker = None
    # Also nuke the model cache
    with app_state.model_cache.lock:
        app_state.model_cache.unload()
    enqueue_step({"type": "model_unloaded"})
    enqueue_step({"type": "status", "status": "idle"})
    return {"ok": True}


# ── System stats ───────────────────────────────────────────────────────────────
@app.get("/stats")
async def get_stats():
    try:
        import psutil
        mem = psutil.virtual_memory()
        ram_used  = round(mem.used  / 1024**3, 1)
        ram_total = round(mem.total / 1024**3, 1)
    except ImportError:
        ram_used = ram_total = 0.0

    vram_allocated = vram_reserved = vram_peak = vram_total = 0.0
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        raw_allocated = torch.cuda.memory_allocated(dev)
        raw_reserved  = torch.cuda.memory_reserved(dev)
        raw_peak      = torch.cuda.max_memory_allocated(dev)
        raw_total     = torch.cuda.get_device_properties(dev).total_memory
        print(
            f"[stats] device={dev}  "
            f"allocated={raw_allocated}B  reserved={raw_reserved}B  "
            f"peak={raw_peak}B  total={raw_total}B",
            flush=True,
        )
        vram_allocated = round(raw_allocated / 1024**3, 2)
        vram_reserved  = round(raw_reserved  / 1024**3, 2)
        vram_peak      = round(raw_peak      / 1024**3, 2)
        vram_total     = round(raw_total     / 1024**3, 1)

    return {
        # vram_used = reserved (includes model params cached by the allocator)
        # more representative than allocated alone when model is loaded at rest
        "vram_used":      vram_reserved,
        "vram_allocated": vram_allocated,
        "vram_peak":      vram_peak,
        "vram_total":     vram_total,
        "ram_used":       ram_used,
        "ram_total":      ram_total,
    }


# ── Presets ────────────────────────────────────────────────────────────────────

class PresetSave(BaseModel):
    name:    str
    config:  dict
    regions: list


@app.get("/presets/list")
async def list_presets():
    presets = []
    for f in sorted(PRESETS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        presets.append({
            "name":        f.stem,
            "filename":    f.name,
            "modified_at": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        })
    return {"presets": presets}


@app.post("/presets/save")
async def save_preset(body: PresetSave):
    from fastapi import HTTPException
    safe = "".join(c if c.isalnum() or c in "-_ ()" else "_" for c in body.name).strip()
    if not safe:
        raise HTTPException(400, "Nombre de preset inválido")
    path = PRESETS_DIR / f"{safe}.json"
    path.write_text(json.dumps({"name": body.name, "config": body.config, "regions": body.regions}, indent=2))
    return {"ok": True, "filename": path.name}


@app.get("/presets/load/{name}")
async def load_preset(name: str):
    from fastapi import HTTPException
    path = PRESETS_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, "Preset no encontrado")
    return json.loads(path.read_text())


@app.delete("/presets/delete/{name}")
async def delete_preset(name: str):
    from fastapi import HTTPException
    path = PRESETS_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, "Preset no encontrado")
    path.unlink()
    return {"ok": True}


# ── Outputs ────────────────────────────────────────────────────────────────────
@app.get("/outputs/list")
async def list_outputs():
    if not OUTPUTS_DIR.exists():
        return {"outputs": []}
    items = []
    for png in sorted(OUTPUTS_DIR.glob("*.png"), reverse=True):
        stem      = png.stem
        meta: dict = {}
        json_path = OUTPUTS_DIR / f"{stem}.json"
        if json_path.exists():
            meta = json.loads(json_path.read_text())
        items.append({"filename": png.name, **meta})
    return {"outputs": items}


@app.get("/outputs/{filename}")
async def get_output(filename: str):
    from fastapi import HTTPException
    path = OUTPUTS_DIR / filename
    if not path.exists() or path.suffix != ".png":
        raise HTTPException(404, "Not found")
    return FileResponse(str(path), media_type="image/png")


# ── Debug ─────────────────────────────────────────────────────────────────────
@app.get("/debug/mask/{region_id}")
async def get_debug_mask(region_id: str):
    """Retorna la última máscara guardada de una región como imagen PNG."""
    from fastapi import HTTPException
    from fastapi.responses import FileResponse as FR
    if not DEBUG_MASKS_DIR.exists():
        raise HTTPException(404, "No debug masks saved yet")
    masks = sorted(DEBUG_MASKS_DIR.glob(f"region_{region_id}_*.png"))
    if not masks:
        raise HTTPException(404, f"No mask found for region '{region_id}'")
    return FR(str(masks[-1]), media_type="image/png")


@app.get("/debug/masks")
async def list_debug_masks():
    """Lista todos los archivos de máscaras guardados."""
    if not DEBUG_MASKS_DIR.exists():
        return {"masks": []}
    files = sorted(DEBUG_MASKS_DIR.glob("*.png"))
    return {"masks": [f.name for f in files]}


# ── WebSocket ──────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_ws.append(websocket)
    # Push current model state immediately on connect
    if app_state.model_cache.is_loaded:
        await websocket.send_text(json.dumps({
            "type": "model_loaded",
            "path": app_state.model_cache.model_path,
        }))
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "regions_update":
                with app_state.lock:
                    app_state.regions = [Region(**r) for r in msg["regions"]]
            # ping → no-op
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in active_ws:
            active_ws.remove(websocket)
