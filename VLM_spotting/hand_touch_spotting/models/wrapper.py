"""Frozen Qwen3-VL wrapper — the LLM boundary.

Loads Qwen3-VL-8B frozen (ViT + LLM) and exposes exactly what the connector/localizer need:
  tokenizer, embed_tokens (frozen nn.Embedding the connector RMS-matches),
  model (.generate + teacher-forced labels), llm / core / visual (FSDP + anchors),
  d_llm, n_layers, and vit_anchor_groups() for the per-second frozen-ViT anchor tokens.
A new LLM backbone plugs in by re-implementing these attributes.
"""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


class QwenWrapper(nn.Module):
    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda",
                 dtype=torch.bfloat16, attn_impl: str = "sdpa"):
        super().__init__()
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        self.device, self.dtype = device, dtype
        self.processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype, attn_implementation=attn_impl, device_map={"": device})
        model.eval().requires_grad_(False)                  # fully frozen
        self.model = model                                  # for .generate / labels forward
        self.core = model.model                             # Qwen3VLModel (get_image_features lives here)
        self.llm = model.model.language_model               # Qwen3VLTextModel (FSDP-sharded at runtime)
        self.visual = model.model.visual
        self.embed_tokens = self.llm.embed_tokens           # frozen token embedding
        self.tokenizer = self.processor.tokenizer
        self.d_llm = model.config.text_config.hidden_size
        self.n_layers = model.config.text_config.num_hidden_layers

    @torch.no_grad()
    def vit_anchor_groups(self, frames_1fps: List, max_side: int = 0, return_grid: bool = False):
        """frames_1fps: list of HxWx3 uint8 RGB (one per anchored second). max_side>0 downsizes each
        frame (shortest side -> max_side) before Qwen's ViT to cut anchor token count. Returns a list
        of (g_s, d) frozen ViT token groups in the LLM space; return_grid also yields the post-merge
        (gh, gw) grid shapes."""
        from PIL import Image
        pil = [Image.fromarray(f) for f in frames_1fps]
        if max_side:
            resized = []
            for im in pil:
                w, h = im.size
                sc = max_side / min(w, h)
                resized.append(im.resize((max(16, round(w * sc)), max(16, round(h * sc)))))
            pil = resized
        proc = self.processor.image_processor(images=pil, return_tensors="pt")
        pixel_values = proc["pixel_values"].to(self.device, self.visual.dtype)
        grid_thw = proc["image_grid_thw"].to(self.device)
        out = self.core.get_image_features(pixel_values, grid_thw)
        # transformers>=4.57 returns (per_image_embeds, deepstack); take per-image
        # (we feed anchors as plain inputs_embeds; the base was trained without the deepstack features).
        if isinstance(out, tuple):
            out = out[0]
        groups = out.pooler_output if hasattr(out, "pooler_output") else out
        groups = [g.to(self.dtype) for g in groups]
        if not return_grid:
            return groups
        merge = getattr(self.model.config.vision_config, "spatial_merge_size", 2)
        grids = []
        for i in range(len(groups)):
            t, h, w = [int(x) for x in grid_thw[i]]
            gh, gw = h // merge, w // merge
            assert gh * gw * t == groups[i].shape[0], (gh, gw, t, groups[i].shape[0])
            grids.append((gh, gw))
        return groups, grids
