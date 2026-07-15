"""FrameCompress — the V-JEPA → LLM connector (the only trainable module, ~27.55M params).

Per frame: 576 spatial patches → K=8 learnable query tokens (one cross-attn pool, no 576² cost)
→ concat(K·768) → Linear → d_llm, then RMS-matched to the LLM's token embeddings so the motion
tokens sit in-distribution. This is the pure query-pool connector (language-injection ablated out).

  even grid (N/2,576,768) --linear interp--> (N,576,768)   # interp BEFORE compress
  LN → K queries cross-attend the 576 patches → concat(K·768) → Linear→d_llm → RMS-match
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class XAttn(nn.Module):
    """Cross-attention pool: queries attend keys/values. NOTE attr names .n/.a are in the
    saved state_dict (keys a3.n.*, a3.a.*) — do not rename."""
    def __init__(self, d, h):
        super().__init__()
        self.n = nn.LayerNorm(d)                                  # pre-norm on the queries
        self.a = nn.MultiheadAttention(d, h, batch_first=True)

    def forward(self, q, kv):                                     # q[B,Lq,d] kv[B,Lk,d]
        o, _ = self.a(self.n(q), kv, kv)
        return o


class FrameCompress(nn.Module):
    # state_dict keys are a hard checkpoint contract (existing conn_s*.pt must load strict):
    #   q, in_ln.{w,b}, a3.n.{w,b}, a3.a.{in_proj_weight,in_proj_bias,out_proj.weight,out_proj.bias}, out.0.{w,b}, out.1.{w,b}
    def __init__(self, d=768, n_q=8, d_llm=4096, n_heads=4, target_rms: float | None = None):
        super().__init__()
        self.n_q = n_q
        self.target_rms = target_rms
        self.q = nn.Parameter(torch.randn(n_q, d) * 0.02)         # K learnable query tokens
        self.in_ln = nn.LayerNorm(d)                              # norm the 576 patch tokens
        self.a3 = XAttn(d, n_heads)                               # K queries ← 576 patches (the pool)
        self.out = nn.Sequential(nn.LayerNorm(n_q * d), nn.Linear(n_q * d, d_llm))  # → LLM space

    @torch.no_grad()
    def set_target_rms_from(self, embed_weight):
        """Match output RMS to the LLM token embeddings so motion tokens are in-distribution."""
        self.target_rms = embed_weight.float().pow(2).mean(-1).sqrt().mean().item()
        return self.target_rms

    def _interp(self, even, N):                                  # even (N2,576,768) → (N,576,768)
        n2, P, D = even.shape
        x = even.permute(1, 2, 0).reshape(1, P * D, n2)
        x = F.interpolate(x, size=N, mode="linear", align_corners=False)
        return x.reshape(P, D, N).permute(2, 0, 1).contiguous()

    def qformer_tokens(self, even_grid, N, chunk=64):
        """Per-frame Q-Former output (N, n_q*768) — BEFORE the `out` readout. The InfoNCE tap point
        (keys are separated at the Q-Former extraction, not at the collapsed out.1 readout)."""
        dev = self.q.device
        grid = self._interp(even_grid.to(dev).float(), N)
        outs = []
        for s in range(0, N, chunk):
            patches = self.in_ln(grid[s:s + chunk]); b = patches.shape[0]
            queries = self.q.unsqueeze(0).expand(b, -1, -1)
            outs.append(self.a3(queries, patches).reshape(b, -1))   # (b, n_q*768)
        return torch.cat(outs, 0)

    def forward(self, even_grid, text_emb, N, chunk=64):
        """even_grid (N2,576,768) → (N,d_llm). `text_emb` is accepted for call-site compat but
        unused (the winning connector is a pure query-pool). fp32 compute is a numeric contract."""
        dev = self.q.device
        grid = self._interp(even_grid.to(dev).float(), N)        # (N,576,768) fp32
        outs = []
        for s in range(0, N, chunk):
            patches = self.in_ln(grid[s:s + chunk])              # (b,576,d)
            b = patches.shape[0]
            queries = self.q.unsqueeze(0).expand(b, -1, -1)      # (b,K,d)
            pooled = self.a3(queries, patches)                   # (b,K,d)
            outs.append(self.out(pooled.reshape(b, -1)))         # (b,d_llm)
        y = torch.cat(outs, 0)                                   # (N,d_llm)
        if self.target_rms is not None:                         # RMS-match to LLM embeds
            rms = y.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
            y = y / rms * self.target_rms
        return y
