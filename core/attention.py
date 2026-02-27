"""
Fase 4 — Inyección de Mapas de Atención (Estrategia B)
=======================================================
Control espacial por token via máscaras dibujadas a mano.

AttentionInjector instala _CrossAttnProcessor en las capas de cross-attention
(attn2) del UNet para interceptar y modificar los attention weights en runtime.

Modos:
  record : captura attention maps promediados sobre heads, por step y layer.
  inject : mezcla attention weights del modelo con máscaras del usuario por token.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ── Custom attention processor ───────────────────────────────────────────────

class _CrossAttnProcessor:
    """
    Reemplaza el processor standard en capas cross-attention (attn2).
    Usa la misma lógica que diffusers.AttnProcessor para poder interceptar
    `attention_probs` entre get_attention_scores y bmm(probs, value).
    """

    def __init__(self, injector: "AttentionInjector", layer_name: str) -> None:
        self._inj  = injector
        self._name = layer_name

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        is_cross = encoder_hidden_states is not None
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = (
                hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            )

        batch_size, sequence_length, _ = (
            hidden_states.shape if not is_cross else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        ctx = hidden_states if not is_cross else encoder_hidden_states
        if is_cross and attn.norm_cross:
            ctx = attn.norm_encoder_hidden_states(ctx)

        key   = attn.to_k(ctx)
        value = attn.to_v(ctx)

        query = attn.head_to_batch_dim(query)
        key   = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # attention_probs: (batch*heads, n_spatial, n_tokens)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # ── Intercepción ──────────────────────────────────────────────────────
        if is_cross and self._inj._mode is not None:
            attention_probs = self._inj._process(attention_probs, self._name, attn.heads)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Proyección de salida + dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = (
                hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ── Injector principal ────────────────────────────────────────────────────────

class AttentionInjector:
    """
    Controla la inyección de atención en el UNet.

    Instala _CrossAttnProcessor en cada capa cross-attention (nombre contiene
    "attn2") y restaura los processors originales en stop().

    API pública
    -----------
    set_token_mask(token_index, mask_np, intensity)
    clear_masks()
    start(mode, gen_h, gen_w)   — 'record' | 'inject'
    stop()
    advance_step()              — llamar al final de cada denoising step
    get_recorded_maps()
    """

    def __init__(self, unet) -> None:
        self.unet  = unet
        self._mode: Optional[str] = None
        self._step: int  = 0
        self._gen_h: int = 512
        self._gen_w: int = 512

        # token_index → {"mask": np.ndarray float32, "intensity": float}
        self._token_masks: Dict[int, dict] = {}

        # step → layer_name → Tensor (n_tokens, H, W)  [CPU]
        self._recorded_maps: Dict[int, Dict[str, torch.Tensor]] = {}

        # processors originales para restaurar
        self._original_processors: Dict = {}

    # ── API pública ───────────────────────────────────────────────────────────

    def set_token_mask(self, token_index: int, mask: np.ndarray, intensity: float) -> None:
        self._token_masks[token_index] = {
            "mask":      mask.astype(np.float32),
            "intensity": float(intensity),
        }

    def clear_masks(self) -> None:
        self._token_masks.clear()

    def start(self, mode: str, gen_h: int = 512, gen_w: int = 512) -> None:
        """Activa hooks. mode = 'record' | 'inject'."""
        self._mode  = mode
        self._step  = 0
        self._gen_h = gen_h
        self._gen_w = gen_w
        if mode == "record":
            self._recorded_maps = {}
        self._install_processors()

    def stop(self) -> None:
        """Restaura processors originales y limpia estado activo."""
        self._restore_processors()
        self._mode = None

    def advance_step(self) -> None:
        """Incrementar step counter. Llamar al final de cada denoising step."""
        self._step += 1

    def get_recorded_maps(self) -> Dict:
        return dict(self._recorded_maps)

    # ── Legacy shims (compatibilidad con stub original) ───────────────────────

    def start_recording(self) -> None:
        self.start("record")

    def start_injection(self, maps: Dict, strength: float = 1.0) -> None:
        self.start("inject")

    def start_mask_injection(
        self, token_index: int, mask: np.ndarray, strength: float = 0.8
    ) -> None:
        self.set_token_mask(token_index, mask, strength)
        self.start("inject")

    def get_maps(self) -> Dict:
        return self.get_recorded_maps()

    # ── Gestión de processors ─────────────────────────────────────────────────

    def _install_processors(self) -> None:
        self._original_processors = dict(self.unet.attn_processors)
        new_procs = dict(self._original_processors)
        for name in self._original_processors:
            if "attn2" in name:
                new_procs[name] = _CrossAttnProcessor(self, name)
        self.unet.set_attn_processor(new_procs)

    def _restore_processors(self) -> None:
        if self._original_processors:
            self.unet.set_attn_processor(self._original_processors)
            self._original_processors = {}

    # ── Procesamiento interno ─────────────────────────────────────────────────

    def _infer_hw(self, n_spatial: int) -> Tuple[int, int]:
        """Infiere H y W de la resolución de atención dado n_spatial y el aspect ratio."""
        ratio = self._gen_h / max(self._gen_w, 1)
        h = round(math.sqrt(n_spatial * ratio))
        w = round(math.sqrt(n_spatial / ratio))
        if h * w != n_spatial:
            h = max(1, int(math.isqrt(n_spatial)))
            w = n_spatial // h
        return max(h, 1), max(w, 1)

    def _resize_mask(
        self,
        mask_np: np.ndarray,
        h: int,
        w: int,
        device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Redimensiona máscara numpy (gen_h, gen_w) → tensor (h, w) con bilinear."""
        t = torch.from_numpy(mask_np).to(device=device, dtype=dtype)
        out = F.interpolate(t[None, None], size=(h, w), mode="bilinear", align_corners=False)
        return out[0, 0]

    def _process(
        self,
        attention_probs: torch.Tensor,
        layer_name: str,
        num_heads: int,
    ) -> torch.Tensor:
        """
        Punto de entrada desde _CrossAttnProcessor.
        attention_probs: (batch*heads, n_spatial, n_tokens)
        """
        total_bh, n_spatial, n_tokens = attention_probs.shape
        batch_size = total_bh // max(num_heads, 1)
        h, w = self._infer_hw(n_spatial)
        device = attention_probs.device
        dtype  = attention_probs.dtype

        if self._mode == "record":
            self._do_record(attention_probs, layer_name, num_heads, batch_size, h, w, n_tokens)

        elif self._mode == "inject" and self._token_masks:
            attention_probs = self._do_inject(
                attention_probs, layer_name, num_heads, batch_size, h, w, n_tokens, device, dtype
            )

        return attention_probs

    def _do_record(
        self,
        attention_probs: torch.Tensor,
        layer_name: str,
        num_heads: int,
        batch_size: int,
        h: int,
        w: int,
        n_tokens: int,
    ) -> None:
        # Tomar la parte condicionada (último elemento del batch)
        cond_start = (batch_size - 1) * num_heads
        cond_probs = attention_probs[cond_start:]          # (heads, n_spatial, n_tokens)
        avg        = cond_probs.mean(dim=0)                # (n_spatial, n_tokens)
        maps = avg.transpose(0, 1).view(n_tokens, h, w).detach().cpu()  # (n_tokens, H, W)

        if self._step not in self._recorded_maps:
            self._recorded_maps[self._step] = {}
        self._recorded_maps[self._step][layer_name] = maps

    def _do_inject(
        self,
        attention_probs: torch.Tensor,
        layer_name: str,
        num_heads: int,
        batch_size: int,
        h: int,
        w: int,
        n_tokens: int,
        device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        probs      = attention_probs.clone()
        cond_start = (batch_size - 1) * num_heads

        for token_idx, info in self._token_masks.items():
            if token_idx >= n_tokens:
                continue

            intensity    = info["intensity"]
            mask_resized = self._resize_mask(info["mask"], h, w, device, dtype)  # (h, w)
            mask_flat    = mask_resized.reshape(h * w)                            # (n_spatial,)
            mask_flat    = mask_flat / (mask_flat.sum() + 1e-8)

            # pesos actuales para este token en el batch condicionado
            orig          = probs[cond_start:, :, token_idx]            # (num_heads, n_spatial)
            mask_expanded = mask_flat.unsqueeze(0).expand(num_heads, -1) # (num_heads, n_spatial)
            blended       = orig * (1.0 - intensity) + mask_expanded * intensity
            probs[cond_start:, :, token_idx] = blended

            coverage = float((mask_resized > 0.05).float().mean() * 100)
            delta    = float((blended - orig).pow(2).mean().sqrt())
            print(
                f"[ATTN] layer={layer_name} | step={self._step} | "
                f"token={token_idx} | intensity={intensity:.2f} | "
                f"mask_coverage={coverage:.1f}% | delta_L2={delta:.4f}",
                flush=True,
            )

        # Re-normalizar: cada posición espacial debe sumar 1 sobre tokens
        row_sums = probs[cond_start:].sum(dim=-1, keepdim=True).clamp(min=1e-8)
        probs[cond_start:] = probs[cond_start:] / row_sums

        return probs
