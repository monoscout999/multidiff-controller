import torch
from pathlib import Path
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPM++": DPMSolverMultistepScheduler,
    "Euler": EulerDiscreteScheduler,
    "EulerA": EulerAncestralDiscreteScheduler,
}


def detect_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    if sm >= 120:       # RTX 5000 Blackwell
        return torch.bfloat16
    return torch.float16  # RTX 4000 Ada


def _is_local_file(path: str) -> bool:
    p = Path(path)
    return p.suffix.lower() in (".safetensors", ".ckpt") and p.exists()


def _detect_sdxl(path: str) -> bool:
    """
    Detecta si un checkpoint local es SDXL leyendo las keys del header
    del safetensors (operación rápida, no carga pesos).
    Fallback a False para .ckpt (necesitaría torch.load, omitimos por seguridad).
    """
    p = Path(path)
    if p.suffix.lower() != ".safetensors":
        return False
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = f.keys()
            for k in keys:
                if "conditioner.embedders" in k or "add_embedding" in k:
                    return True
        return False
    except Exception:
        return False


def load_pipeline(model_path: str, scheduler_name: str = "DDIM"):
    dtype  = detect_dtype()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    is_local = _is_local_file(model_path)
    is_xl    = _detect_sdxl(model_path) if is_local else ("xl" in model_path.lower())

    if is_local:
        cls = StableDiffusionXLPipeline if is_xl else StableDiffusionPipeline
        pipe = cls.from_single_file(
            model_path,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(device)
    else:
        # Hub ID or local diffusers directory
        cls = StableDiffusionXLPipeline if is_xl else StableDiffusionPipeline
        pipe = cls.from_pretrained(
            model_path,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)

    sched_cls = SCHEDULERS.get(scheduler_name, DDIMScheduler)
    pipe.scheduler = sched_cls.from_config(pipe.scheduler.config)

    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.eval()

    return pipe, dtype, device, is_xl
