"""InfoNCE key-separation: break the connector's 'shared saliency key' collapse.

The vocab lens showed all event-frame tokens collapse to ~one direction (effective rank ≈1). This pushes
different-type/direction event-frame Q-Former tokens APART with a supervised (multi-positive) InfoNCE,
tapped at the Q-Former output z_t (before out.1 — the collapse is downstream, in out.1's readout, so we
fix the *extraction*). Trains the Q-Former (q,in_ln,a3) + two proj heads; out.1 / LLM / V-JEPA frozen.

命门 self-check: the HARD negatives are OTHER-type event-frame tokens (cross-batch, via the bank). A model
that emits one shared key CANNOT satisfy this loss (start/end tokens identical -> pos and neg overlap ->
loss stays high). If negatives were only background frames, a shared key WOULD satisfy it -> useless.
"""
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCEHeads(nn.Module):
    def __init__(self, d_qformer, d_query, d_proj=256):
        super().__init__()
        self.proj_frame = nn.Linear(d_qformer, d_proj)     # Q-Former token (n_q*768) -> 256
        self.proj_query = nn.Linear(d_query, d_proj)        # pooled query embed (d_llm) -> 256

    def frame(self, z):                                     # z (T, d_qformer) -> unit (T, 256)
        return F.normalize(self.proj_frame(z.float()), dim=-1)

    def query(self, qv):                                    # qv (d_query,) -> unit (256,)
        return F.normalize(self.proj_query(qv.float().unsqueeze(0)), dim=-1)[0]


class KeyBank:
    """MoCo-style queue of DETACHED event-frame proj tokens + type labels — supplies cross-batch hard
    negatives when batch_size=1 (otherwise there are no other-type events in a step)."""
    def __init__(self, cap=512):
        self.tok = deque(maxlen=cap)
        self.typ = deque(maxlen=cap)

    def add(self, toks, typ):                               # toks (P,256) detached
        for t in toks:
            self.tok.append(t); self.typ.append(typ)

    def negatives(self, exclude_type):
        """banked tokens whose type != exclude_type — the HARD other-type/direction negatives."""
        keep = [t for t, ty in zip(self.tok, self.typ) if ty != exclude_type]
        return torch.stack(keep) if keep else None

    def __len__(self):
        return len(self.tok)


def infonce(q_vec, pos_tok, neg_tok, temp):
    """Multi-positive InfoNCE (supervised-contrastive form): all positives in the numerator, averaged.
    q_vec (256,) unit; pos (P,256) unit; neg (Nn,256) unit. Returns scalar loss."""
    cand = torch.cat([pos_tok, neg_tok], 0)                 # (P+Nn, 256)
    sim = (cand @ q_vec) / temp                             # (P+Nn,)
    logp = sim - torch.logsumexp(sim, 0)
    return -(logp[:pos_tok.shape[0]]).mean()


@torch.no_grad()
def collapse_stats(Z):
    """Collapse metrics for event-frame tokens Z (N, d). Returns (mean_pairwise_cosine, effective_rank).

    mean_pairwise_cos : off-diagonal mean cosine of unit rows. ~1 => all tokens point one way (the collapsed
        shared key); DROPS as InfoNCE separates them. Robust, monotone, no high-d saturation — the headline.
    effective_rank    : exp(-Σ p_i log p_i), p_i=σ_i/Σσ, on RAW (unnormalized, uncentered) Z, so the common
        component dominates σ_1 (=> ~1 when collapsed) instead of saturating. Climbs as keys spread."""
    Z = Z.float()
    U = Z / (Z.norm(dim=-1, keepdim=True) + 1e-9)
    n = U.shape[0]
    g = U @ U.T
    mpc = float((g.sum() - g.diag().sum()) / max(1, n * (n - 1)))     # off-diagonal mean cosine
    s = torch.linalg.svdvals(Z)                                       # RAW: common component dominates
    p = s / (s.sum() + 1e-9)
    er = float(torch.exp(-(p * (p + 1e-12).log()).sum()))
    return mpc, er
