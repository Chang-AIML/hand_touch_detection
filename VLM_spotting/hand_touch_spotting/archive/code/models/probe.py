"""Pure V-JEPA probe baseline (NO VLM) — the A3 ablation control.

Same Motion-Bridge adaptor, but the query is a learnable per-type vector instead of
the frozen VLM's [LOC] hidden. No LLM, no anchors, no text — just:
    s(t) = cos( q_type , Adaptor(VJEPA_t) ) / temp
Exposes the same forward_batch(batch) -> (s_list, metas) interface as Localizer so
the detection loss / eval / sliding-window code work unchanged. Trains ~100x cheaper
(no 8B forward), so it isolates exactly how much the VLM adds over a V-JEPA probe.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

TYPES = ("touch", "untouch")


class ProbeModel(nn.Module):
    def __init__(self, adaptor, d_llm: int = 4096, temp: float = 0.1,
                 types=TYPES, init_std: float = 0.02):
        super().__init__()
        self.adaptor = adaptor
        self.temp = temp
        self.types = list(types)
        self.query = nn.Parameter(torch.randn(len(self.types), d_llm) * init_std)
        # attributes the train loop probes for (kept None so guards skip them)
        self.sim_head = None
        self.film = None

    def forward_batch(self, batch):
        dev = self.query.device
        dt = next(self.adaptor.parameters()).dtype
        s_list, metas = [], []
        for s in batch:
            feats = s["feats"].to(dev, dt)
            z = self.adaptor(feats).float()                       # (N, d)
            ti = self.types.index(s["type"])
            q = F.normalize(self.query[ti].float(), dim=-1)
            sim = (F.normalize(z, dim=-1) @ q) / self.temp        # (N,)
            s_list.append(sim)
            metas.append({"video_id": s["video_id"], "type": s["type"],
                          "num_frames": feats.shape[0], "gt": s.get("gt", -1),
                          "event_frames": s.get("event_frames")})
        return s_list, metas
