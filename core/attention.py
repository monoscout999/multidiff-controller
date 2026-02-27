"""
Fase 4 — Módulo Inyección de Mapas de Atención
================================================
Interface definida. Implementación pendiente.

Uso target:

    injector = AttentionInjector(pipe.unet)

    # Modo RECORD — capturar mapas durante una generación
    injector.start_recording()
    # ... correr pipeline normalmente ...
    maps = injector.get_maps()   # {layer_name: [tensor_por_step, ...]}
    injector.stop()

    # Modo INJECT — inyectar mapas de una generación previa
    injector.start_injection(maps, strength=1.0)
    # ... correr pipeline normalmente ...
    injector.stop()

    # Modo MASK INJECT — inyectar máscara resizeada como mapa de atención
    injector.start_mask_injection(
        token_index=3,              # token objetivo
        mask=np.ones((64, 64)),     # máscara en resolución display; se resizea por capa
        strength=0.8,
    )
    injector.stop()

Implementación:
    Los hooks se registran sobre los módulos de cross-attention del UNet.
    Se identifica cada capa como pipe.unet.down_blocks[i].attentions[j].transformer_blocks[k].attn2
    El hook intercepts el resultado de softmax(QK^T/sqrt(d)) antes de multiplicar por V.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import torch


class AttentionInjector:
    """
    Placeholder — hooks se registrarán en Fase 4.
    La clase expone la API completa; los métodos son no-ops por ahora.
    """

    def __init__(self, unet):
        self.unet = unet
        self._hooks: List = []
        self._mode: Optional[str] = None   # 'record' | 'inject' | None
        self._recorded_maps: Dict[str, List] = {}
        self._inject_maps: Optional[Dict] = None
        self._inject_strength: float = 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        """Registra hooks para capturar mapas de atención en cada step."""
        self._mode = "record"
        self._recorded_maps = {}
        # TODO (Fase 4): registrar forward_hooks en capas attn2 del UNet

    def start_injection(self, maps: Dict, strength: float = 1.0) -> None:
        """Registra hooks para inyectar mapas de atención externos."""
        self._mode = "inject"
        self._inject_maps = maps
        self._inject_strength = strength
        # TODO (Fase 4): registrar forward_hooks que reemplacen attention weights

    def start_mask_injection(
        self,
        token_index: int,
        mask: np.ndarray,
        strength: float = 0.8,
    ) -> None:
        """
        Inyecta una máscara como mapa de atención para un token específico.
        La máscara se resizea a la resolución interna de cada capa de atención.
        """
        self._mode = "inject"
        self._inject_strength = strength
        # TODO (Fase 4): resize mask per layer, build synthetic attention map,
        #               register hooks

    def get_maps(self) -> Dict:
        """Retorna los mapas capturados. Válido tras start_recording + generación."""
        return dict(self._recorded_maps)

    def stop(self) -> None:
        """Elimina todos los hooks activos."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._mode = None
        self._inject_maps = None
