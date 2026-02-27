import threading
import time
import io
import base64
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import torch
from PIL import Image

from .state import GenerationConfig, ModelCache, Region
from .model_loader import load_pipeline
from .multidiffusion import multidiffusion_step, RegionBundle
from .attention import AttentionInjector


class PipelineWorker(threading.Thread):
    """
    Worker thread para el loop de generación.
    Soporta SD 1.5 y SDXL en el mismo loop (denoising distinto según tipo).
    """

    def __init__(
        self,
        config: GenerationConfig,
        regions: List[Region],
        on_step: Callable[[dict], None],
        model_cache: Optional[ModelCache] = None,
    ):
        super().__init__(daemon=True)
        self.config = config
        self.regions = regions
        self.on_step = on_step
        self._model_cache = model_cache

        self._stop_event  = threading.Event()
        self._panic_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()   # not paused initially

        self._running = False
        self._pipe    = None
        self._is_xl   = False
        self._injector: Optional[AttentionInjector] = None

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.set()

    def panic(self) -> None:
        self._panic_event.set()
        self._stop_event.set()
        self._pause_event.set()

    def resume(self) -> None:
        self._pause_event.set()

    # ------------------------------------------------------------------
    # Thread entry
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._running = True
        try:
            self._run_generation()
        except torch.cuda.OutOfMemoryError:
            self._unload_model(clear_cache=True)
            self.on_step({"type": "error", "message": "CUDA out of memory — VRAM freed"})
        except Exception as exc:
            self.on_step({"type": "error", "message": str(exc)})
        finally:
            self._running = False
            if self._panic_event.is_set():
                self._unload_model(clear_cache=True)
            else:
                self._pipe = None

    # ------------------------------------------------------------------
    # Generation loop
    # ------------------------------------------------------------------

    def _run_generation(self) -> None:
        cfg = self.config

        # ── Load or reuse cached model ────────────────────────────────
        if self._model_cache and self._model_cache.is_loaded_for(cfg.model_path):
            self._pipe  = self._model_cache.pipe
            dtype       = self._model_cache.dtype
            device      = self._model_cache.device
            self._is_xl = getattr(self._model_cache, "is_xl", False)
        else:
            self.on_step({"type": "status", "status": "loading_model"})
            self._pipe, dtype, device, self._is_xl = load_pipeline(
                cfg.model_path, cfg.scheduler
            )
            if self._model_cache:
                with self._model_cache.lock:
                    self._model_cache.pipe       = self._pipe
                    self._model_cache.dtype      = dtype
                    self._model_cache.device     = device
                    self._model_cache.model_path = cfg.model_path
                    self._model_cache.is_xl      = self._is_xl

        if self._stop_event.is_set():
            self.on_step({"type": "status", "status": "idle"})
            return

        self.on_step({"type": "status", "status": "generating"})
        self._injector = AttentionInjector(self._pipe.unet)

        # Seed
        seed = cfg.seed if cfg.seed >= 0 else int(time.time() * 1000) % (2**32)
        generator = torch.Generator(device=device).manual_seed(seed)

        # Prompt embeddings (SD1.5 or SDXL)
        text_emb, pooled_emb = self._encode_prompts(
            cfg.prompt, cfg.negative_prompt, device, dtype
        )

        # Initial latent noise
        latents = torch.randn(
            (1, self._pipe.unet.config.in_channels, cfg.height // 8, cfg.width // 8),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        scheduler = self._pipe.scheduler
        scheduler.set_timesteps(cfg.steps, device=device)
        latents = latents * scheduler.init_noise_sigma

        # SDXL additional conditioning
        add_cond_kwargs = None
        if self._is_xl and pooled_emb is not None:
            add_time_ids = torch.tensor(
                [[cfg.height, cfg.width, 0, 0, cfg.height, cfg.width]],
                dtype=dtype, device=device,
            ).repeat(2, 1)   # 2 = uncond + cond (CFG)
            add_cond_kwargs = {
                "text_embeds": pooled_emb,
                "time_ids":    add_time_ids,
            }

        # ── Pre-compute region bundles (embeddings + mask tensors) ─────
        # Done once before the loop — not recalculated per step.
        latent_h = cfg.height // 8
        latent_w = cfg.width  // 8
        region_bundles = self._prepare_region_bundles(
            self.regions, latent_w, latent_h, device, dtype, cfg.steps
        )
        print(
            f"[PIPELINE] {len(region_bundles)} bundle(s) listos  |  "
            f"ranges: {[(b.region_id, b.step_start, b.step_end) for b in region_bundles]}  |  "
            f"noise: {[round(b.noise, 3) for b in region_bundles]}",
            flush=True,
        )

        # ── Step loop ──────────────────────────────────────────────────
        for i, t in enumerate(scheduler.timesteps):
            if self._stop_event.is_set():
                break

            print(
                f"[PIPELINE] step {i}/{cfg.steps}  |  "
                f"llamando multidiffusion_step con {len(region_bundles)} bundle(s)",
                flush=True,
            )

            # Phase 3 — MultiDiffusion
            md_result = multidiffusion_step(
                latents           = latents,
                timestep          = t,
                step_index        = i,
                total_steps       = cfg.steps,
                region_bundles    = region_bundles,
                global_text_emb   = text_emb,
                global_pooled_emb = pooled_emb,
                unet              = self._pipe.unet,
                scheduler         = scheduler,
                cfg_scale         = cfg.cfg_scale,
                add_cond_kwargs   = add_cond_kwargs,
            )

            if md_result is not None:
                print(f"[PIPELINE] step {i} → MultiDiffusion OK", flush=True)
                latents = md_result
            else:
                print(f"[PIPELINE] step {i} → fallback denoising estándar (md_result=None)", flush=True)
                # Standard CFG denoising
                latent_input = scheduler.scale_model_input(torch.cat([latents] * 2), t)
                with torch.no_grad():
                    unet_kwargs = dict(
                        sample=latent_input,
                        timestep=t,
                        encoder_hidden_states=text_emb,
                    )
                    if add_cond_kwargs:
                        unet_kwargs["added_cond_kwargs"] = add_cond_kwargs
                    noise_pred = self._pipe.unet(**unet_kwargs).sample

                uncond, cond = noise_pred.chunk(2)
                noise_pred = uncond + cfg.cfg_scale * (cond - uncond)
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            # Decode + broadcast
            preview = self._decode_latent(latents)
            step_num = i + 1
            self.on_step({
                "type": "step",
                "step": step_num,
                "total": cfg.steps,
                "image": self._img_to_b64(preview),
                "seed":  seed,
            })

            # ── Pause check ──────────────────────────────────────────
            if not self._stop_event.is_set():
                for region in self.regions:
                    if region.mode != "pause":
                        continue
                    effective_end = region.step_end if region.step_end >= 0 else cfg.steps
                    if step_num == effective_end:
                        self._pause_event.clear()
                        self.on_step({
                            "type":        "paused",
                            "step":        step_num,
                            "total":       cfg.steps,
                            "region_id":   region.id,
                            "region_name": region.name,
                        })
                        self._pause_event.wait()
                        if self._stop_event.is_set():
                            break
                        self.on_step({"type": "status", "status": "generating"})
                        break

        # ── Final image ────────────────────────────────────────────────
        if not self._stop_event.is_set():
            final = self._decode_latent(latents)
            self._save_output(final, seed)
            self.on_step({
                "type":  "complete",
                "image": self._img_to_b64(final),
                "seed":  seed,
            })

        self.on_step({"type": "status", "status": "idle"})

    # ------------------------------------------------------------------
    # Prompt encoding — SD1.5 and SDXL
    # ------------------------------------------------------------------

    def _encode_prompts(self, prompt: str, negative: str, device: str, dtype: torch.dtype):
        """
        Returns (text_embeddings, pooled_embeddings).
        pooled_embeddings is None for SD1.5, tensor for SDXL.
        """
        if self._is_xl:
            # SDXL: pipe.encode_prompt handles both text encoders
            (prompt_emb, neg_emb,
             pooled_prompt, pooled_neg) = self._pipe.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                negative_prompt=negative,
                negative_prompt_2=None,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )
            text_emb   = torch.cat([neg_emb,    prompt_emb])
            pooled_emb = torch.cat([pooled_neg, pooled_prompt])
            return text_emb.to(dtype), pooled_emb.to(dtype)
        else:
            # SD 1.5 / 2.x — manual tokenizer + text_encoder
            def _enc(text: str) -> torch.Tensor:
                tokens = self._pipe.tokenizer(
                    [text],
                    padding="max_length",
                    max_length=self._pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    return self._pipe.text_encoder(
                        tokens.input_ids.to(device)
                    )[0].to(dtype)
            return torch.cat([_enc(negative), _enc(prompt)]), None

    # ------------------------------------------------------------------
    # Decode helpers
    # ------------------------------------------------------------------

    def _decode_latent(self, latents: torch.Tensor) -> Image.Image:
        scaled = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self._pipe.vae.decode(scaled).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return Image.fromarray((image[0] * 255).astype(np.uint8))

    def _img_to_b64(self, img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
        return base64.b64encode(buf.getvalue()).decode()

    def _save_output(self, img: Image.Image, seed: int) -> None:
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]  # trim microseconds
        stem = f"output_{ts}_{seed}"
        img.save(outputs_dir / f"{stem}.png")
        cfg  = self.config
        meta = {
            "seed":            seed,
            "prompt":          cfg.prompt,
            "negative_prompt": cfg.negative_prompt,
            "steps":           cfg.steps,
            "cfg_scale":       cfg.cfg_scale,
            "width":           cfg.width,
            "height":          cfg.height,
            "model":           Path(cfg.model_path).name,
            "timestamp":       ts,
        }
        (outputs_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))

    # ------------------------------------------------------------------
    # Region bundle preparation
    # ------------------------------------------------------------------

    def _prepare_region_bundles(
        self,
        regions:  List[Region],
        latent_w: int,
        latent_h: int,
        device:   str,
        dtype:    torch.dtype,
        total_steps: int,
    ) -> List[RegionBundle]:
        """
        Pre-computes per-region text embeddings and decodes mask_b64 to latent-
        resolution tensors. Called once before the step loop.
        """
        bundles: List[RegionBundle] = []
        for region in regions:
            # ── Mask ─────────────────────────────────────────────────
            if region.mask_b64:
                mask = self._decode_mask_b64(
                    region.mask_b64, latent_w, latent_h, device, dtype
                )
            else:
                mask = torch.ones(1, 1, latent_h, latent_w, device=device, dtype=dtype)

            print(
                f"[mask] '{region.name}' ({region.id}): "
                f"shape={tuple(mask.shape)}  "
                f"mean={mask.mean().item():.3f}  "
                f"steps=[{region.step_start}, "
                f"{'end' if region.step_end < 0 else region.step_end}]"
            )

            # ── Embeddings ───────────────────────────────────────────
            prompt   = region.prompt   or self.config.prompt
            negative = region.negative_prompt or self.config.negative_prompt
            text_emb, pooled_emb = self._encode_prompts(prompt, negative, device, dtype)

            step_end  = region.step_end if region.step_end >= 0 else total_steps
            cfg_scale = region.cfg_override if region.cfg_override is not None else self.config.cfg_scale

            bundles.append(RegionBundle(
                region_id         = region.id,
                mask              = mask,
                text_embeddings   = text_emb,
                pooled_embeddings = pooled_emb,
                intensity         = region.intensity,
                step_start        = region.step_start,
                step_end          = step_end,
                cfg_scale         = cfg_scale,
                noise             = region.noise,
            ))

        return bundles

    @staticmethod
    def _decode_mask_b64(
        mask_b64: str,
        latent_w: int,
        latent_h: int,
        device:   str,
        dtype:    torch.dtype,
    ) -> torch.Tensor:
        """
        Decodes a base64 grayscale PNG (generation resolution W×H) and
        resizes it to latent resolution (latent_w × latent_h).
        Returns tensor [1, 1, latent_h, latent_w] normalised to [0, 1].
        """
        data  = base64.b64decode(mask_b64)
        img   = Image.open(io.BytesIO(data)).convert("L")   # grayscale
        img   = img.resize((latent_w, latent_h), Image.LANCZOS)
        arr   = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr)[None, None]          # [1, 1, H, W]
        return tensor.to(device=device, dtype=dtype)

    def _unload_model(self, clear_cache: bool = False) -> None:
        if clear_cache and self._model_cache:
            with self._model_cache.lock:
                self._model_cache.unload()
        self._pipe = None
