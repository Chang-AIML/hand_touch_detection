"""Phase 1 localization loss — exp_plan §5.2.

s(t) = cos(h_loc, h_vjepa(t)) / temp over the N frames of one video. The GT frame
is the positive; its temporal neighbours are hard negatives (naturally the hardest
classes in the softmax), the rest of the video are easy negatives.

  L = CE(s, gt)  [+ label smoothing over a Gaussian window]
      + lambda_mae * |soft_argmax(s) - gt| / N   [localization regressor]

CE over frames == in-video InfoNCE with every other frame as a negative; the
optional Gaussian soft target tolerates the eval frame-tolerance while keeping the
peak at gt. A neighbour-margin term can be enabled for extra hard-neg pressure.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _soft_target(N: int, gt: int, sigma: float, device) -> torch.Tensor:
    if sigma <= 0:
        t = torch.zeros(N, device=device)
        t[gt] = 1.0
        return t
    idx = torch.arange(N, device=device, dtype=torch.float32)
    w = torch.exp(-0.5 * ((idx - gt) / sigma) ** 2)
    return w / w.sum()


def soft_argmax(s: torch.Tensor) -> torch.Tensor:
    """Expected frame index under softmax(s). s already temp-scaled."""
    p = F.softmax(s, dim=-1)
    idx = torch.arange(s.shape[-1], device=s.device, dtype=torch.float32)
    return (p * idx).sum()


def loc_loss(s_scaled: torch.Tensor, gt: int, lambda_mae: float = 0.5,
             label_smooth_frames: float = 1.0, neighbor_k: int = 4,
             neighbor_margin: float = 0.0):
    """s_scaled: (N,) cos-sim already divided by temperature. Returns (loss, stats)."""
    N = s_scaled.shape[-1]
    gt = int(max(0, min(N - 1, gt)))
    logp = F.log_softmax(s_scaled, dim=-1)
    tgt = _soft_target(N, gt, label_smooth_frames, s_scaled.device)
    ce = -(tgt * logp).sum()

    exp_idx = soft_argmax(s_scaled)
    mae = (exp_idx - gt).abs() / N

    loss = ce + lambda_mae * mae

    if neighbor_margin > 0 and neighbor_k > 0:
        lo, hi = max(0, gt - neighbor_k), min(N, gt + neighbor_k + 1)
        neigh = torch.cat([s_scaled[lo:gt], s_scaled[gt + 1:hi]])
        if neigh.numel():
            # push gt above each neighbour by a margin (hinge)
            loss = loss + neighbor_margin * F.relu(
                margin_target(neigh, s_scaled[gt])).mean()

    stats = {"ce": float(ce.detach()), "mae_frac": float(mae.detach()),
             "argmax_frame": float(exp_idx.detach()), "hard_frame": int(s_scaled.argmax())}
    return loss, stats


def margin_target(neigh, pos, margin=1.0):
    return margin - (pos - neigh)


def dist_contrastive_loss(s_scaled: torch.Tensor, gt: int, sigma: float = 4.0,
                          far_margin: float = 5.0, far_weight: float = 1.0,
                          lambda_mae: float = 0.5):
    """Distance-weighted contrastive loss (per user's idea): tokens NEAR the action
    should be similar (low penalty), tokens FAR should be dissimilar (penalty grows
    with distance) -> directly suppresses the spurious far 'second peaks'.

      L = CE(soft Gaussian target)                         # smooth single peak at gt
        + far_weight * mean_t relu( far_margin*|t-gt|/N - (s_gt - s_t) )  # far must be low
        + lambda_mae * |soft_argmax - gt| / N
    The middle term requires every frame to sit at least (far_margin * normalized
    distance) BELOW the peak: near frames may be high, far frames are pushed down."""
    N = s_scaled.shape[-1]
    gt = int(max(0, min(N - 1, gt)))
    idx = torch.arange(N, device=s_scaled.device, dtype=torch.float32)
    d = (idx - gt).abs()
    # (1) soft positive Gaussian target CE
    tgt = torch.exp(-(d ** 2) / (2 * sigma ** 2)); tgt = tgt / tgt.sum()
    ce = -(tgt * F.log_softmax(s_scaled, dim=-1)).sum()
    # (2) distance-weighted far suppression
    s_gt = s_scaled[gt]
    req_gap = far_margin * (d / N)                          # 0 at gt, far_margin at edges
    far = F.relu(req_gap - (s_gt - s_scaled)).mean()
    # (3) soft-argmax MAE
    exp_idx = soft_argmax(s_scaled)
    mae = (exp_idx - gt).abs() / N
    loss = ce + far_weight * far + lambda_mae * mae
    stats = {"ce": float(ce.detach()), "far": float(far.detach()),
             "mae_frac": float(mae.detach()), "hard_frame": int(s_scaled.argmax())}
    return loss, stats


def detection_loss(s_scaled: torch.Tensor, event_frames, dilate: int = 2,
                   pos_weight: float = 15.0):
    """Multi-event per-frame detection (E2E-Spot style). s_scaled = cos/temp used as
    logits. Target y(t)=1 within +/-dilate of ANY same-type event frame, else 0
    (empty event_frames -> all 0 = event-absent window, rejection built in).
    pos_weight compensates the rare-positive imbalance."""
    N = s_scaled.shape[-1]
    y = torch.zeros(N, device=s_scaled.device)
    for e in event_frames:
        e = int(max(0, min(N - 1, e)))
        y[max(0, e - dilate): min(N, e + dilate + 1)] = 1.0
    pw = torch.tensor(pos_weight, device=s_scaled.device, dtype=s_scaled.dtype)
    return F.binary_cross_entropy_with_logits(s_scaled, y, pos_weight=pw)


def reject_loss(s_scaled: torch.Tensor, peak_mult: float = 3.0):
    """Negative window (event ABSENT) -> keep the distribution FLAT so the model emits
    no confident peak. SCALE-FREE: penalise the peak softmax probability above
    peak_mult x uniform (1/W). A flat window has peak ~1/W -> 0 loss; a peaked window
    is penalised regardless of the (tiny) absolute cosine scale."""
    p = F.softmax(s_scaled, dim=-1)
    W = s_scaled.shape[-1]
    return F.relu(p.max() - peak_mult / W)
