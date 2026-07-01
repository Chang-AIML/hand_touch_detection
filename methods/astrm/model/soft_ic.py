""" Soft Instance Contrastive (Soft-IC) loss.

From the ASTRM paper (eq.9-10), itself adapted from the Instance Contrastive
(IC) loss of Han et al. (CVPR 2022, "Expanding low-density latent regions for
open-set object detection").  Soft-IC makes per-instance features compact within
a class and separated across classes, and is designed to work with mixup, where
labels are soft.

Scope: the paper defines Soft-IC over the K *event* classes c_i (it targets
event-class imbalance). Background is NOT a class here -- it is never enqueued
and never participates in the contrastive term. This module therefore holds one
bank per *event* class and consumes event-only soft labels of shape (N, K).

Design / interpretation choices (the paper is light on implementation details):
  * One FIFO memory bank per event class, each holding up to `bank_size`
    L2-normalized feature vectors together with the soft class weight w.
  * With mixup, every non-zero entry w_ij in a sample's soft label contributes
    one Soft-IC term and is enqueued into that class bank, matching eq.9.
  * Per eq.10 the numerator pairs z_i with same-class bank items (pull together),
    the denominator normalizes over the other event banks A(c_i)=M\\M(c_i)
    (push apart).  The paper writes a log-ratio; this module returns its
    negative so minimizing the loss compacts same-class features and separates
    cross-class features.

The loss is a small auxiliary term (lambda_sic = 0.001 in the paper).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftICLoss(nn.Module):
    def __init__(self, num_event_classes, feat_dim=128, bank_size=256,
                 temperature=0.1, min_weight=1e-3):
        super().__init__()
        self.num_event_classes = num_event_classes
        self.feat_dim = feat_dim
        self.bank_size = bank_size
        self.tau = temperature
        self.min_weight = min_weight

        # FIFO memory banks (not learned), one per event class. Registered as
        # buffers so they move with .to(device) and are checkpointed.
        self.register_buffer(
            'bank_feat', torch.zeros(num_event_classes, bank_size, feat_dim))
        self.register_buffer(
            'bank_weight', torch.zeros(num_event_classes, bank_size))
        self.register_buffer(
            'bank_count', torch.zeros(num_event_classes, dtype=torch.long))
        self.register_buffer(
            'bank_ptr', torch.zeros(num_event_classes, dtype=torch.long))

    @torch.no_grad()
    def _enqueue(self, feats, soft_labels):
        """Store each non-zero soft-label pair (z_i, w_ij) in event bank j."""
        for c in range(self.num_event_classes):
            weights = soft_labels[:, c]
            sel = weights > self.min_weight
            n = int(sel.sum())
            if n == 0:
                continue
            f = feats[sel].detach()
            w = weights[sel].detach()
            ptr = int(self.bank_ptr[c])
            if n >= self.bank_size:
                f, w, n, ptr = f[-self.bank_size:], w[-self.bank_size:], \
                    self.bank_size, 0
            end = ptr + n
            if end <= self.bank_size:
                self.bank_feat[c, ptr:end] = f
                self.bank_weight[c, ptr:end] = w
            else:                                   # wrap around
                first = self.bank_size - ptr
                self.bank_feat[c, ptr:] = f[:first]
                self.bank_weight[c, ptr:] = w[:first]
                self.bank_feat[c, :end - self.bank_size] = f[first:]
                self.bank_weight[c, :end - self.bank_size] = w[first:]
            self.bank_ptr[c] = (ptr + n) % self.bank_size
            self.bank_count[c] = min(self.bank_size,
                                     int(self.bank_count[c]) + n)

    def forward(self, feats, soft_labels, update_bank=True):
        """feats: (N, feat_dim) projected per-frame features (pre-normalized ok).
        soft_labels: (N, num_event_classes) event-only soft (mixup)/one-hot
            labels -- background is NOT included.
        Returns a scalar loss and optionally enqueues the batch into the banks.
        """
        feats = F.normalize(feats, dim=1)

        # Need at least two event classes populated for a contrastive signal.
        populated = (self.bank_count > 0).sum().item()
        loss_terms = []
        if populated >= 2:
            bank_f = F.normalize(self.bank_feat, dim=2)    # (C, B, d)
            for c in range(self.num_event_classes):
                if int(self.bank_count[c]) == 0:
                    continue
                weights = soft_labels[:, c]
                sel = weights > self.min_weight
                if sel.sum() == 0:
                    continue
                zi = feats[sel]                            # (n_i, d)
                wi = weights[sel].clamp_min(self.min_weight)

                # same-class bank M(c)
                m_valid = self.bank_count[c]
                pos_f = bank_f[c, :m_valid]                # (P, d)
                pos_w = self.bank_weight[c, :m_valid]      # (P,)

                # other event banks A(c) = M \ M(c)
                neg_f, neg_w = [], []
                for c2 in range(self.num_event_classes):
                    if c2 == c or int(self.bank_count[c2]) == 0:
                        continue
                    nv = self.bank_count[c2]
                    neg_f.append(bank_f[c2, :nv])
                    neg_w.append(self.bank_weight[c2, :nv])
                if not neg_f:
                    continue
                neg_f = torch.cat(neg_f, dim=0)            # (Q, d)
                neg_w = torch.cat(neg_w, dim=0)            # (Q,)

                # weighted similarities (eq.10 uses w_j * z_j inside the dot)
                pos_logits = (zi @ (pos_w[:, None] * pos_f).t()) / self.tau  # (n_i,P)
                neg_logits = (zi @ (neg_w[:, None] * neg_f).t()) / self.tau  # (n_i,Q)

                # log [ exp(pos) / sum_neg exp(neg) ], averaged over same-class j
                neg_lse = torch.logsumexp(neg_logits, dim=1, keepdim=True)   # (n_i,1)
                log_ratio = pos_logits - neg_lse                            # (n_i,P)
                # mean over the same-class bank, then weight by 1/w_i (eq.10)
                per_inst = log_ratio.mean(dim=1) / wi                       # (n_i,)
                loss_terms.append(-per_inst)

        if update_bank:
            self._enqueue(feats, soft_labels)

        if not loss_terms:
            return feats.sum() * 0.0      # zero loss, keeps graph/device intact
        return torch.cat(loss_terms).mean()
