"""
Fase 3 — MultiDiffusion
========================
Denoising regional con prompts independientes por zona, combinados via
promedio ponderado por máscaras suavizadas con Gaussian blur.

Flujo:
  1. Una llamada UNet con embeddings globales produce noise_global (base fallback)
  2. Por cada región activa: una llamada UNet con sus embeddings → noise_i
  3. Acumulación: noise_accum  += blur(mask_i) * intensity_i * noise_i
                  weight_accum += blur(mask_i) * intensity_i
  4. Resultado: noise_final = (noise_global + noise_accum) / (1 + weight_accum)
     → zonas sin ninguna región → noise_global puro
     → zonas con región coverage → blend ponderado
  5. scheduler.step(noise_final, t, latents) → nuevos latents
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class RegionBundle:
    region_id:         str
    mask:              torch.Tensor          # [1, 1, latent_H, latent_W]  float32 0-1
    text_embeddings:   torch.Tensor          # [2, seq_len, dim]  (uncond ‖ cond)
    pooled_embeddings: Optional[torch.Tensor]  # [2, dim] para SDXL, None para SD1.5
    intensity:         float                 # 0-1, escala el peso de la máscara
    step_start:        int
    step_end:          int                   # ya resuelto (sin -1)
    cfg_scale:         float  = 7.5         # per-region CFG (may differ from global)
    noise:             float  = 0.0         # additive latent noise before region forward pass


# ── Public API ─────────────────────────────────────────────────────────────────

def multidiffusion_step(
    latents:         torch.Tensor,
    timestep:        torch.Tensor,
    step_index:      int,
    total_steps:     int,
    region_bundles:  List[RegionBundle],
    global_text_emb: torch.Tensor,
    global_pooled_emb: Optional[torch.Tensor],
    unet,
    scheduler,
    cfg_scale:       float,
    add_cond_kwargs: Optional[Dict] = None,
) -> Optional[torch.Tensor]:
    """
    Punto de entrada del hook en pipeline.py.
    Filtra regiones activas en este step y corre MultiDiffusion si hay alguna.
    Retorna nuevos latents, o None para delegarlo al denoising estándar.
    """
    print(
        f"[MULTIDIFF] step={step_index}/{total_steps}  |  "
        f"bundles recibidos: {len(region_bundles)}  |  "
        f"ranges: {[(b.region_id, b.step_start, b.step_end) for b in region_bundles]}",
        flush=True,
    )

    active = [
        b for b in region_bundles
        if b.step_start <= step_index < b.step_end
    ]

    print(
        f"[MULTIDIFF] step={step_index}  |  "
        f"activas tras filtro: {len(active)}  |  "
        f"noise_strengths: {[round(b.noise, 3) for b in active]}",
        flush=True,
    )

    if not active:
        return None

    return _apply_multidiffusion(
        latents          = latents,
        t                = timestep,
        step_index       = step_index,
        active_bundles   = active,
        global_text_emb  = global_text_emb,
        global_pooled_emb= global_pooled_emb,
        unet             = unet,
        scheduler        = scheduler,
        cfg_scale        = cfg_scale,
        add_cond_kwargs  = add_cond_kwargs,
    )


# ── Core algorithm ─────────────────────────────────────────────────────────────

def _apply_multidiffusion(
    latents:           torch.Tensor,
    t:                 torch.Tensor,
    step_index:        int,
    active_bundles:    List[RegionBundle],
    global_text_emb:   torch.Tensor,
    global_pooled_emb: Optional[torch.Tensor],
    unet,
    scheduler,
    cfg_scale:         float,
    add_cond_kwargs:   Optional[Dict],
) -> torch.Tensor:
    device = latents.device
    dtype  = latents.dtype
    _, C, H, W = latents.shape

    print(
        f"[MULTIDIFF APPLY] step={step_index}  |  "
        f"regiones activas: {len(active_bundles)}  |  "
        f"noise_strengths: {[round(b.noise, 3) for b in active_bundles]}",
        flush=True,
    )

    # Scale latent input once (shared across all UNet calls this step)
    latent_input = scheduler.scale_model_input(torch.cat([latents] * 2), t)

    # ── Global noise prediction ────────────────────────────────────────
    global_add_cond = add_cond_kwargs  # uses global pooled embeddings already
    noise_global = _unet_cfg(
        latent_input, t, global_text_emb, unet, cfg_scale, global_add_cond
    )

    # Accumulators
    noise_accum  = torch.zeros(1, C, H, W, device=device, dtype=dtype)
    weight_accum = torch.zeros(1, 1, H, W, device=device, dtype=dtype)

    # Sigma del timestep actual — usado para escalar el noise injection
    sigma = _get_sigma(scheduler, step_index, t)

    # ── Per-region predictions ─────────────────────────────────────────
    for bundle in active_bundles:
        region_add_cond = _region_add_cond(add_cond_kwargs, bundle.pooled_embeddings)

        # Noise injection escalado por sigma del scheduler
        # Efecto: coherente con el nivel de ruido esperado en este step.
        # En steps tempranos (sigma alto) la perturbación es mayor,
        # en steps tardíos (sigma bajo) la perturbación es proporcional.
        if bundle.noise > 0.0:
            blurred_mask  = _blur_mask(bundle.mask)
            raw_noise     = torch.randn_like(latents)
            noise_mean    = raw_noise.mean().item()
            noise_std     = raw_noise.std().item()

            print(
                f"[NOISE PRE]  Region {bundle.region_id} | step {step_index} | "
                f"noise_strength {bundle.noise:.2f} | sigma {sigma:.4f} | "
                f"ruido mean {noise_mean:.4f} std {noise_std:.4f}",
                flush=True,
            )

            perturbation  = raw_noise * (bundle.noise * sigma) * blurred_mask
            noisy_latents = latents + perturbation

            print(
                f"[NOISE POST] Region {bundle.region_id} | step {step_index} | "
                f"effective_scale {bundle.noise * sigma:.4f} | "
                f"perturb mean {perturbation.mean().item():.4f} std {perturbation.std().item():.4f} | "
                f"latente antes mean {latents.mean().item():.4f} → después mean {noisy_latents.mean().item():.4f}",
                flush=True,
            )

            region_latent_input = scheduler.scale_model_input(
                torch.cat([noisy_latents] * 2), t
            )
        else:
            region_latent_input = latent_input

        noise_region = _unet_cfg(
            region_latent_input, t, bundle.text_embeddings, unet,
            bundle.cfg_scale, region_add_cond,
        )
        weight = _blur_mask(bundle.mask) * bundle.intensity   # [1, 1, H, W]
        noise_accum  += weight * noise_region
        weight_accum += weight

        # Per-step blend diagnostic
        diff_l2 = (noise_region - noise_global).norm().item()
        print(
            f"[MD blend] region={bundle.region_id}  cfg={bundle.cfg_scale:.1f}  "
            f"w=[{weight.min().item():.3f},{weight.max().item():.3f}]  "
            f"L2_diff={diff_l2:.3f}"
        )

    # ── Weighted blend ─────────────────────────────────────────────────
    # Denominator = 1 (global) + accumulated region weights per pixel
    denom       = (1.0 + weight_accum).clamp(min=1e-6)
    noise_final = (noise_global + noise_accum) / denom

    return scheduler.step(noise_final, t, latents).prev_sample


# ── Helpers ────────────────────────────────────────────────────────────────────

def _unet_cfg(
    latent_input:   torch.Tensor,
    t:              torch.Tensor,
    text_emb:       torch.Tensor,
    unet,
    cfg_scale:      float,
    add_cond_kwargs: Optional[Dict],
) -> torch.Tensor:
    """Single UNet call + CFG. Returns predicted noise [1, C, H, W]."""
    kwargs = dict(sample=latent_input, timestep=t, encoder_hidden_states=text_emb)
    if add_cond_kwargs:
        kwargs["added_cond_kwargs"] = add_cond_kwargs
    with torch.no_grad():
        noise_pred = unet(**kwargs).sample
    uncond, cond = noise_pred.chunk(2)
    return uncond + cfg_scale * (cond - uncond)


def _blur_mask(mask: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """
    Gaussian blur sobre máscara [1, 1, H, W].
    Suaviza los bordes para evitar artefactos lineales en el blend.
    """
    radius    = max(1, int(sigma * 2))
    size      = 2 * radius + 1
    x         = torch.arange(size, dtype=mask.dtype, device=mask.device) - radius
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :])[None, None]  # [1,1,k,k]
    return F.conv2d(mask, kernel_2d, padding=radius).clamp(0.0, 1.0)


def _get_sigma(scheduler, step_index: int, t: torch.Tensor) -> float:
    """
    Deriva el sigma (nivel de ruido) para el timestep actual.
    Compatible con todos los schedulers de diffusers:
      - Euler, DPM++, EulerA → usan scheduler.sigmas (ODE flow)
      - DDIM, PNDM           → derivado de alphas_cumprod
    """
    # Schedulers ODE (Euler, DPM++, EulerA): exponen .sigmas tras set_timesteps
    if hasattr(scheduler, 'sigmas') and scheduler.sigmas is not None:
        sigmas = scheduler.sigmas
        if step_index < len(sigmas):
            return float(sigmas[step_index])

    # DDIM, PNDM: derivar sigma de alpha_cumprod usando el timestep discreto
    if hasattr(scheduler, 'alphas_cumprod'):
        t_idx = int(t.long().item())
        alpha = float(scheduler.alphas_cumprod[t_idx])
        return ((1.0 - alpha) / max(alpha, 1e-8)) ** 0.5

    # Fallback sin scaling (equivalente al comportamiento anterior)
    return 1.0


def _region_add_cond(
    base_add_cond: Optional[Dict],
    pooled_emb:    Optional[torch.Tensor],
) -> Optional[Dict]:
    """
    Para SDXL: reemplaza text_embeds con los pooled embeddings de la región,
    conservando time_ids del contexto global.
    """
    if base_add_cond is None or pooled_emb is None:
        return base_add_cond
    return {"text_embeds": pooled_emb, "time_ids": base_add_cond["time_ids"]}
