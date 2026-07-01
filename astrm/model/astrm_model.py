""" ASTRM precise event-spotting model.

Same skeleton as E2E-Spot's E2EModel but with:
  * RegNetY backbone refined by ASTRM (instead of GSM),
  * a Bi-GRU temporal block,
  * a classifier head + a 128-d projection head for the Soft-IC loss,
  * an ASAM two-step training loop with BCE + mixup + Soft-IC.

Label / class convention
------------------------
`num_classes` is the E2E-Spot convention K+1 = (background + K event classes).
The paper, however, treats classification and Soft-IC over the K *event*
classes only (Y in R^{Ts x K}, eq.11; Soft-IC is defined over event classes
c_i).  So internally:
  * BCE head outputs K event logits (NO background logit); background is implicit
    (low on every event sigmoid). For evaluation we rebuild a (K+1) score vector
    with bg = 1 - max_k sigmoid(event_k).
  * Soft-IC operates on the K event classes only; background is never enqueued.
  * `event_target` (N, K) is the foreground target shared by BCE and Soft-IC.
The CE branch is kept as a non-paper option and still uses a (K+1) softmax head.
"""

from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm import tqdm

from model.common import BaseRGBModel
from model.astrm import make_astrm
from model.asam import ASAM
from model.soft_ic import SoftICLoss

MAX_GRU_HIDDEN_DIM = 768


class ASTRMModel(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, num_classes, feature_arch, clip_len, modality='rgb',
                     proj_dim=128, cls_loss='bce', astrm_kwargs=None):
            super().__init__()
            astrm_kwargs = astrm_kwargs or {}
            is_rgb = modality == 'rgb'

            # ---- backbone ----
            base = feature_arch.rsplit('_', 1)[0]
            if base in ('rny002', 'rny008'):
                features = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny008': 'regnety_008',
                }[base], pretrained=is_rgb)
                feat_dim = features.head.fc.in_features
                features.head.fc = nn.Identity()
            else:
                raise NotImplementedError(feature_arch)

            self._require_clip_len = -1
            if feature_arch.endswith('_astrm'):
                make_astrm(features, clip_len, **astrm_kwargs)
                self._require_clip_len = clip_len     # ASTRM needs fixed T
            self._features = features
            self._feat_dim = feat_dim

            # ---- temporal block: Bi-GRU ----
            hidden_dim = min(feat_dim, MAX_GRU_HIDDEN_DIM)
            self._gru = nn.GRU(
                feat_dim, hidden_dim, num_layers=1, batch_first=True,
                bidirectional=True)
            self._dropout = nn.Dropout()

            # ---- heads ----
            # BCE: K event logits (no background). CE: K+1 logits (with bg).
            self._cls_dim = num_classes if cls_loss == 'ce' else num_classes - 1
            self._fc_cls = nn.Linear(2 * hidden_dim, self._cls_dim)
            self._fc_proj = nn.Linear(2 * hidden_dim, proj_dim)

        def forward(self, x, return_proj=False):
            B, true_clip_len, C, H, W = x.shape
            clip_len = true_clip_len
            if self._require_clip_len > 0:
                assert true_clip_len <= self._require_clip_len, \
                    'Expected <= {}, got {}'.format(
                        self._require_clip_len, true_clip_len)
                if true_clip_len < self._require_clip_len:
                    x = F.pad(
                        x, (0,) * 7 + (self._require_clip_len - true_clip_len,))
                    clip_len = self._require_clip_len

            im_feat = self._features(
                x.view(-1, C, H, W)).reshape(B, clip_len, self._feat_dim)
            if true_clip_len != clip_len:
                im_feat = im_feat[:, :true_clip_len, :]

            y, _ = self._gru(im_feat)
            y = self._dropout(y)
            logits = self._fc_cls(y)
            if return_proj:
                return logits, self._fc_proj(y)
            return logits

        def print_stats(self):
            print('Model params: {:.3f}M'.format(
                sum(p.numel() for p in self.parameters()) / 1e6))
            print('  Backbone (RegNetY+ASTRM): {:.3f}M'.format(
                sum(p.numel() for p in self._features.parameters()) / 1e6))
            print('  Temporal+heads: {:.3f}M'.format(
                (sum(p.numel() for p in self._gru.parameters())
                 + sum(p.numel() for p in self._fc_cls.parameters())
                 + sum(p.numel() for p in self._fc_proj.parameters())) / 1e6))

    def __init__(self, num_classes, feature_arch, clip_len, modality='rgb',
                 device='cuda', multi_gpu=False, cls_loss='bce', fg_weight=1,
                 use_soft_ic=True, lambda_sic=0.001, soft_ic_tau=0.1,
                 amp_dtype=torch.bfloat16, astrm_kwargs=None):
        self.device = device
        self._multi_gpu = multi_gpu
        self._num_classes = num_classes            # K+1 (with background)
        self._n_event = num_classes - 1            # K event classes
        self._cls_loss = cls_loss
        # head width: CE keeps the background logit, BCE drops it
        self._cls_dim = num_classes if cls_loss == 'ce' else self._n_event
        self._fg_weight = fg_weight
        self._use_soft_ic = use_soft_ic
        self._lambda_sic = lambda_sic
        self._amp_dtype = amp_dtype

        self._model = ASTRMModel.Impl(
            num_classes, feature_arch, clip_len, modality,
            cls_loss=cls_loss, astrm_kwargs=astrm_kwargs)
        self._model.print_stats()

        # Soft-IC is defined over the K event classes only (no background).
        self._soft_ic = SoftICLoss(
            self._n_event, temperature=soft_ic_tau) if use_soft_ic else None

        if multi_gpu:
            self._model = nn.DataParallel(self._model)
        self._model.to(device)
        if self._soft_ic is not None:
            self._soft_ic.to(device)

    # ASAM needs access to the underlying nn.Module's named parameters
    @property
    def _core(self):
        return self._model.module if isinstance(self._model, nn.DataParallel) \
            else self._model

    def get_optimizer(self, opt_args):
        return torch.optim.AdamW(self._model.parameters(), **opt_args), None

    def state_dict(self):
        model = self._core if isinstance(self._model, nn.DataParallel) \
            else self._model
        state = {'model': model.state_dict()}
        if self._soft_ic is not None:
            state['soft_ic'] = self._soft_ic.state_dict()
        return state

    def load(self, state_dict):
        # Backward-compatible with checkpoints saved before Soft-IC buffers were
        # included.
        if isinstance(state_dict, dict) and 'model' in state_dict:
            model_state = state_dict['model']
            soft_ic_state = state_dict.get('soft_ic')
        else:
            model_state = state_dict
            soft_ic_state = None

        if isinstance(self._model, nn.DataParallel):
            self._model.module.load_state_dict(model_state)
        else:
            self._model.load_state_dict(model_state)
        if self._soft_ic is not None and soft_ic_state is not None:
            self._soft_ic.load_state_dict(soft_ic_state)

    # ---- loss ----
    def _cls_loss_fn(self, logits, hard_label, soft_full, event_target):
        """logits: (N, cls_dim).
        hard_label: (N,) int over K+1, or None when targets are soft (mixup).
        soft_full:  (N, K+1) soft targets over background+events (for CE).
        event_target: (N, K) foreground soft targets (for BCE).
        """
        if self._cls_loss == 'ce':
            w = None
            if self._fg_weight != 1:
                w = torch.FloatTensor(
                    [1] + [self._fg_weight] * self._n_event).to(logits.device)
            if hard_label is not None:
                return F.cross_entropy(logits, hard_label, weight=w)
            # soft targets (mixup): weighted soft cross-entropy
            logp = F.log_softmax(logits, dim=1)
            ce = -(soft_full * logp)
            if w is not None:
                ce = ce * w[None, :]
            return ce.sum(dim=1).mean()
        elif self._cls_loss == 'bce':
            # K event logits only; background is implicit (no logit).
            pos_weight = torch.full(
                (self._n_event,), float(self._fg_weight), device=logits.device)
            return F.binary_cross_entropy_with_logits(
                logits, event_target, pos_weight=pos_weight)
        raise NotImplementedError(self._cls_loss)

    def _forward_loss(self, frame, hard_label, soft_full, event_target,
                      update_soft_ic=True):
        with torch.autocast('cuda', dtype=self._amp_dtype):
            if self._soft_ic is not None:
                logits, proj = self._model(frame, return_proj=True)
                logits = logits.reshape(-1, self._cls_dim)
                proj = proj.reshape(-1, proj.shape[-1])
            else:
                logits = self._model(frame).reshape(-1, self._cls_dim)
                proj = None
            loss = self._cls_loss_fn(logits, hard_label, soft_full, event_target)
        if self._soft_ic is not None:
            # Soft-IC over the K event classes only (background excluded).
            sic = self._soft_ic(
                proj.float(), event_target, update_bank=update_soft_ic)
            loss = loss + self._lambda_sic * sic
        return loss

    def epoch(self, loader, optimizer=None, minimizer=None, lr_scheduler=None,
              acc_grad_iter=1):
        train = optimizer is not None
        self._model.train(train)

        epoch_loss = 0.0
        with nullcontext() if train else torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = loader.dataset.load_frame_gpu(batch, self.device)
                label = batch['label'].to(self.device)
                # Build the three target views:
                #   hard_label (N,)    -> CE hard path (None under mixup)
                #   soft_full  (N,K+1) -> CE soft path
                #   event_target (N,K) -> BCE + Soft-IC (background dropped)
                if label.dim() == 3:                       # soft (mixup)
                    soft_full = label.reshape(-1, label.shape[-1]).float()
                    hard_label = None
                else:                                      # hard labels
                    hard_label = label.flatten()
                    soft_full = F.one_hot(
                        hard_label, self._num_classes).float()
                event_target = soft_full[:, 1:].contiguous()

                if not train:
                    loss = self._forward_loss(
                        frame, hard_label, soft_full, event_target,
                        update_soft_ic=False)
                    epoch_loss += loss.detach().item()
                    continue

                # ----- ASAM two-step (or plain step) -----
                do_step = (batch_idx + 1) % acc_grad_iter == 0
                use_asam_step = minimizer is not None and do_step
                # Clean pass: compute loss and refresh the memory bank with
                # clean-weight features (more representative than perturbed ones).
                loss = self._forward_loss(
                    frame, hard_label, soft_full, event_target,
                    update_soft_ic=True)
                (loss / acc_grad_iter).backward()
                if use_asam_step:
                    minimizer.ascent_step()
                    # Perturbed pass: gradient only, never touch the bank.
                    loss2 = self._forward_loss(
                        frame, hard_label, soft_full, event_target,
                        update_soft_ic=False)
                    (loss2 / acc_grad_iter).backward()
                    minimizer.descent_step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                elif do_step:
                    optimizer.step()
                    optimizer.zero_grad()
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                epoch_loss += loss.detach().item()
        return epoch_loss / len(loader)

    def predict(self, seq, use_amp=True):
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if seq.dim() == 4:
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        self._model.eval()
        with torch.no_grad():
            with torch.autocast('cuda', dtype=self._amp_dtype) if use_amp \
                    else nullcontext():
                logits = self._model(seq).float()
            if self._cls_loss == 'bce':
                # logits are the K event logits; rebuild a (K+1) score vector
                # with background = 1 - max_k sigmoid so argmax/NMS still work.
                fg = torch.sigmoid(logits)
                bg = 1.0 - fg.amax(dim=2, keepdim=True)
                pred = torch.cat([bg, fg], dim=2)
            else:
                pred = torch.softmax(logits, dim=2)
            return torch.argmax(pred, dim=2).cpu().numpy(), pred.cpu().numpy()
