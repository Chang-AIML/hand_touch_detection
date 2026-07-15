"""Frozen Qwen3-VL wrapper: token injection + hidden-state extraction.

Builds the §3.3 sequence in EMBEDDING space and runs it through Qwen3-VL's frozen
text transformer, bypassing the multimodal placeholder machinery entirely (we feed
inputs_embeds directly, so position_ids default to a plain arange — M-RoPE spatial
channels collapse to standard RoPE, which is all the similarity readout needs).

Frozen: Qwen ViT (visual) + Qwen LLM (language_model). Trainable pieces (adaptor,
LOC/REJ) live outside this module and are passed in as ready embeddings.

Exposes:
  embed_question(str)            -> (Q, d) question token embeds
  vit_anchor_groups(frames1fps)  -> list[(g_s, d)] frozen ViT anchor groups
  run(seq_embeds)                -> tuple(hidden_states)  [len = n_layers+1]
  build_and_run(...)             -> convenience: assemble + forward, return readout dict
"""
from __future__ import annotations

import sys
from typing import List, Optional

import math

import torch
import torch.nn as nn
from transformers.masking_utils import create_causal_mask


class LoRALinear(nn.Module):
    """Minimal LoRA wrap of an nn.Linear: frozen base + trainable low-rank delta.
    Preserves the module tree (replaces the attribute in place) so the manual
    truncated forward keeps working. B init 0 -> starts as the exact base layer."""

    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        dev, dt = base.weight.device, base.weight.dtype
        self.A = nn.Parameter(torch.zeros(r, base.in_features, device=dev, dtype=dt))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))       # B stays 0 -> delta=0 at init
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * ((x @ self.A.t()) @ self.B.t())

sys.path.insert(0, __file__.rsplit("/models/", 1)[0])
from data.token_layout import assemble_embeds  # noqa: E402

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


class QwenWrapper(nn.Module):
    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda",
                 dtype=torch.bfloat16, attn_impl: str = "sdpa"):
        super().__init__()
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        self.device = device
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype, attn_implementation=attn_impl,
            device_map={"": device})
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.core = model.model                         # Qwen3VLModel
        self.llm = model.model.language_model           # Qwen3VLTextModel
        self.visual = model.model.visual
        self.embed_tokens = self.llm.embed_tokens
        self.tokenizer = self.processor.tokenizer
        self.d_llm = model.config.text_config.hidden_size
        self.n_layers = model.config.text_config.num_hidden_layers

    # ---------------------------------------------------------------- text
    @torch.no_grad()
    def embed_question(self, question: str, add_prefix: bool = True) -> torch.Tensor:
        text = question
        if add_prefix:
            text = "Question: " + question + " Answer: the moment is"
        ids = self.tokenizer(text, return_tensors="pt",
                             add_special_tokens=False).input_ids.to(self.device)
        return self.embed_tokens(ids)[0]                # (Q, d)

    # ---------------------------------------------------------------- vision anchors
    @torch.no_grad()
    def vit_anchor_groups(self, frames_1fps: List, max_side: int = 0,
                          return_grid: bool = False):
        """frames_1fps: list of HxWx3 uint8 RGB (one per second).
        max_side>0 downsizes each frame (shortest side -> max_side) BEFORE Qwen's
        ViT, cutting anchor token count (e.g. ~252 -> ~49 tokens/anchor vs 405).
        Returns list of (g_s, d) frozen ViT token groups, in Qwen's LLM space.
        return_grid -> also return list of (gh, gw) post-merge token-grid shapes
        (for M-RoPE spatial position ids)."""
        from PIL import Image
        pil = [Image.fromarray(f) for f in frames_1fps]
        if max_side:
            out = []
            for im in pil:
                w, h = im.size
                sc = max_side / min(w, h)
                out.append(im.resize((max(16, round(w * sc)), max(16, round(h * sc)))))
            pil = out
        proc = self.processor.image_processor(images=pil, return_tensors="pt")
        pixel_values = proc["pixel_values"].to(self.device, self.visual.dtype)
        grid_thw = proc["image_grid_thw"].to(self.device)
        out = self.core.get_image_features(pixel_values, grid_thw)
        # transformers>=4.57 Qwen3-VL returns (image_embeds_per_image, deepstack_image_embeds);
        # take the per-image embeds (a tuple split by image). We feed anchors as plain
        # inputs_embeds and don't use the deepstack side-features (base was trained that way).
        if isinstance(out, tuple):
            out = out[0]
        groups = out.pooler_output if hasattr(out, "pooler_output") else out
        groups = [g.to(self.dtype) for g in groups]       # list of (g_s, d)
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

    # ---------------------------------------------------------------- LLM forward
    @torch.no_grad()
    def run(self, seq_embeds: torch.Tensor):
        """seq_embeds: (L, d) -> hidden_states tuple, each (1, L, d)."""
        x = seq_embeds.unsqueeze(0).to(self.device, self.dtype)
        attn = torch.ones(x.shape[:2], dtype=torch.long, device=self.device)
        out = self.llm(inputs_embeds=x, attention_mask=attn,
                       output_hidden_states=True, use_cache=False, return_dict=True)
        return out.hidden_states                         # tuple len = n_layers+1

    # ---------------------------------------------------------------- LoRA
    def add_lora(self, rank=16, alpha=32, n_layers=None, target="all"):
        """Inject LoRA into the first n_layers decoder layers (the ones the truncated
        forward runs). target: 'attn' (q,k,v,o) | 'all' (+ mlp). Returns lora params."""
        n = n_layers or self.n_layers
        attn_t = ["q_proj", "k_proj", "v_proj", "o_proj"]
        mlp_t = ["gate_proj", "up_proj", "down_proj"] if target == "all" else []
        params = []
        for i in range(n):
            layer = self.llm.layers[i]
            for mod, names in [(layer.self_attn, attn_t), (layer.mlp, mlp_t)]:
                for nm in names:
                    base = getattr(mod, nm)
                    lora = LoRALinear(base, rank, alpha)
                    setattr(mod, nm, lora)
                    params += [lora.A, lora.B]
        self.lora_params = params
        return params

    # ---------------------------------------------------------------- TRAIN forward
    def forward_to_layer(self, embeds: torch.Tensor, layer: int,
                         attention_mask: Optional[torch.Tensor] = None,
                         use_checkpoint: bool = False) -> torch.Tensor:
        """Grad-enabled forward through the first `layer` decoder layers only.

        embeds: (B, S, d); attention_mask: (B, S) of 1/0 (None => pure causal).
        Returns hidden state after `layer` layers == hidden_states[layer]. Backprop
        flows through the frozen LLM to `embeds` (hence to the adaptor / LOC).
        Truncating at `layer` (=sim_layer) skips ~half the network.
        """
        llm = self.llm
        B, S, _ = embeds.shape
        dev = embeds.device
        pos = torch.arange(S, device=dev).view(1, 1, -1).expand(4, B, -1)
        text_pos = pos[0]
        attn = create_causal_mask(config=llm.config, inputs_embeds=embeds,
                                  attention_mask=attention_mask, past_key_values=None,
                                  position_ids=text_pos)
        pos_emb = llm.rotary_emb(embeds, pos[1:])
        h = embeds
        for i in range(layer):
            lyr = llm.layers[i]
            if use_checkpoint:
                def _call(x, _l=lyr):
                    return _l(x, attention_mask=attn, position_ids=text_pos,
                              position_embeddings=pos_emb)
                h = torch.utils.checkpoint.checkpoint(_call, h, use_reentrant=False)
            else:
                h = lyr(h, attention_mask=attn, position_ids=text_pos,
                        position_embeddings=pos_emb)
        return h

    # ---------------------------------------------------------------- convenience
    @torch.no_grad()
    def build_and_run(self, vjepa_embeds: torch.Tensor, question: str,
                      loc_embeds: torch.Tensor, num_frames: int, fps: int,
                      anchor_groups: Optional[List[torch.Tensor]] = None,
                      rej_embed: Optional[torch.Tensor] = None):
        q = self.embed_question(question)
        lay = assemble_embeds(anchor_groups or [], vjepa_embeds, q, loc_embeds,
                              num_frames, fps, rej_embed=rej_embed)
        hs = self.run(lay["embeds"])
        return {"hidden_states": hs, **lay}
