from pydantic import BaseModel, Field
from typing import Optional, List
import threading
import torch


class GenerationConfig(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    steps: int = 20
    cfg_scale: float = 7.5
    seed: int = -1
    width: int = 512
    height: int = 512
    model_path: str = "runwayml/stable-diffusion-v1-5"
    scheduler: str = "DDIM"


class Region(BaseModel):
    id: str
    name: str
    prompt: str = ""
    negative_prompt: str = ""
    intensity: float = Field(default=1.0, ge=0.0, le=1.0)
    step_start: int = 0
    step_end: int = -1          # -1 = until end
    mask_b64: Optional[str] = None
    mode: str = "auto"          # "auto" | "pause"
    cfg_override: Optional[float] = None  # None = use global CFG
    noise: float = Field(default=0.0, ge=0.0, le=0.5)  # additive noise before region denoising


class ModelCache:
    """Holds the loaded diffusion pipeline between generations."""

    def __init__(self):
        self.pipe = None
        self.dtype: Optional[torch.dtype] = None
        self.device: Optional[str] = None
        self.model_path: Optional[str] = None
        self.is_xl: bool = False
        self.lock = threading.Lock()

    def is_loaded_for(self, path: str) -> bool:
        return self.pipe is not None and self.model_path == path

    @property
    def is_loaded(self) -> bool:
        return self.pipe is not None

    def unload(self) -> None:
        """Release model from memory. Must be called while holding self.lock."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self.dtype = None
        self.device = None
        self.model_path = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class AppState:
    def __init__(self):
        self.config = GenerationConfig()
        self.regions: List[Region] = []
        self.model_cache = ModelCache()
        self.lock = threading.Lock()
