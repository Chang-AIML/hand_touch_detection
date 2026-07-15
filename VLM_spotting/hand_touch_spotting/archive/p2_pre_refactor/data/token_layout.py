"""CORE §3.3 — interleaved ViT-anchor | V-JEPA sequence + time_index.

Layout per second (F = fps frames per second):

    [ViT anchor(sec=0)] [VJEPA f0] .. [VJEPA f_{F-1}]
    [ViT anchor(sec=1)] [VJEPA f_F] .. ...
    ...
    [question tokens] [LOC] (... [LOC]_k [REJ] in Phase 2)

`build_layout` is a PURE function (no torch, no model) that returns the ORDER of
segments and the frame-id of every V-JEPA slot, so the time_index mapping is
unit-testable in isolation. `assemble_embeds` then stitches precomputed embedding
tensors in that order and reports the absolute positions of the V-JEPA slots and
the LOC slot(s) for the similarity readout.

INVARIANT (exp_plan §12.4): len(time_index) == num_frames, and time_index[i] is
the frame id of the i-th V-JEPA token, in strictly increasing frame order.
"""
from __future__ import annotations

import math
from typing import List, Tuple


def build_layout(num_frames: int, fps: int) -> Tuple[List[tuple], List[int]]:
    """Return (plan, time_index).

    plan: ordered list of segment descriptors, one of:
        ("vit",   sec)     -> a 1-fps ViT anchor for second `sec`
        ("vjepa", frame)   -> a per-frame V-JEPA motion token for `frame`
    time_index: frame id of each ("vjepa", frame) entry, in plan order.
    (question / loc segments are appended later by assemble_embeds.)
    """
    num_seconds = math.ceil(num_frames / fps)
    plan: List[tuple] = []
    time_index: List[int] = []
    for sec in range(num_seconds):
        plan.append(("vit", sec))
        start = sec * fps
        end = min(start + fps, num_frames)
        for f in range(start, end):
            plan.append(("vjepa", f))
            time_index.append(f)
    assert len(time_index) == num_frames, (len(time_index), num_frames)
    assert time_index == list(range(num_frames))
    return plan, time_index


def assemble_embeds(vit_anchor_groups, vjepa_embeds, question_embeds, loc_embeds,
                    num_frames: int, fps: int, rej_embed=None):
    """Stitch embeddings into one (L, d) sequence following the §3.3 order.

    Args (all torch tensors, same hidden dim d, same device/dtype):
        vit_anchor_groups: list of length num_seconds; entry s is (g_s, d) — the
            group of frozen Qwen-ViT tokens for the 1-fps anchor of second s.
            Pass [] (empty list) for the §7 "without ViT anchor" ablation, or a
            list of (1,d) tensors for a single pooled anchor per second.
        vjepa_embeds:   (N, d)  N = num_frames, adaptor output (frame-aligned)
        question_embeds:(Q, d)  tokenised+embedded question
        loc_embeds:     (K, d)  K LOC tokens (K=1 in Phase 1)
        rej_embed:      (1, d) or None
    Returns dict with:
        embeds        (L, d)
        vjepa_pos     LongTensor (N,)  absolute positions of V-JEPA slots
        time_index    LongTensor (N,)  frame id per V-JEPA slot (== arange(N))
        loc_pos       LongTensor (K,)  absolute positions of LOC slot(s)
        rej_pos       int or None
    """
    import torch
    import math as _m
    num_seconds = _m.ceil(num_frames / fps)
    use_anchor = len(vit_anchor_groups) > 0
    if use_anchor:
        assert len(vit_anchor_groups) == num_seconds, (len(vit_anchor_groups), num_seconds)
    d = vjepa_embeds.shape[-1]
    segs = []
    vjepa_pos, tidx = [], []
    pos = 0
    for sec in range(num_seconds):
        if use_anchor:
            g = vit_anchor_groups[sec]
            segs.append(g)
            pos += g.shape[0]
        start = sec * fps
        end = min(start + fps, num_frames)
        for f in range(start, end):
            segs.append(vjepa_embeds[f:f + 1])
            vjepa_pos.append(pos)
            tidx.append(f)
            pos += 1
    segs.append(question_embeds)
    pos += question_embeds.shape[0]
    loc_pos = list(range(pos, pos + loc_embeds.shape[0]))
    segs.append(loc_embeds)
    pos += loc_embeds.shape[0]
    rej_pos = None
    if rej_embed is not None:
        rej_pos = pos
        segs.append(rej_embed)
        pos += rej_embed.shape[0]
    embeds = torch.cat(segs, dim=0)                       # (L, d)
    assert embeds.shape[-1] == d
    assert tidx == list(range(num_frames)), "time_index broken"
    return {
        "embeds": embeds,
        "vjepa_pos": torch.tensor(vjepa_pos, dtype=torch.long),
        "time_index": torch.tensor(tidx, dtype=torch.long),
        "loc_pos": torch.tensor(loc_pos, dtype=torch.long),
        "rej_pos": rej_pos,
    }


# ============================================================ unit tests (MUST pass)
def _selftest():
    # 1) time_index correctness across shapes
    for num_frames, fps in [(300, 15), (37, 15), (16, 16), (1, 15), (150, 25)]:
        plan, ti = build_layout(num_frames, fps)
        assert ti == list(range(num_frames)), (num_frames, fps)
        n_vit = sum(1 for k, _ in plan if k == "vit")
        n_vj = sum(1 for k, _ in plan if k == "vjepa")
        assert n_vj == num_frames
        assert n_vit == math.ceil(num_frames / fps)
    # 2) assemble_embeds position bookkeeping (needs torch), variable anchor groups
    import torch
    N, F, d = 300, 15, 8
    S = math.ceil(N / F)
    # anchor group of a random size g_s per second (>=1)
    gsizes = [1 + (s % 4) for s in range(S)]
    vit_groups = [torch.full((gsizes[s], d), -100.0 - s) for s in range(S)]
    vj = torch.arange(N * d, dtype=torch.float32).reshape(N, d) + 1000
    q = torch.zeros(5, d)
    loc = torch.ones(1, d) * -1
    out = assemble_embeds(vit_groups, vj, q, loc, N, F)
    L = sum(gsizes) + N + 5 + 1
    assert out["embeds"].shape == (L, d), (out["embeds"].shape, L)
    assert out["vjepa_pos"].shape[0] == N
    assert torch.equal(out["time_index"], torch.arange(N))
    # the embed sitting at each vjepa_pos must equal that frame's vjepa vector
    got = out["embeds"][out["vjepa_pos"]]
    assert torch.equal(got, vj), "vjepa slot embeds misaligned!"
    assert out["loc_pos"].tolist() == [L - 1]
    assert torch.equal(out["embeds"][out["loc_pos"][0]], loc[0])
    # 3) no-anchor ablation path
    out2 = assemble_embeds([], vj, q, loc, N, F)
    assert out2["embeds"].shape == (N + 5 + 1, d)
    assert torch.equal(out2["embeds"][out2["vjepa_pos"]], vj)
    print("[selftest] token_layout time_index + assembly alignment (groups + no-anchor) OK")


if __name__ == "__main__":
    _selftest()
