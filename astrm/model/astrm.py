""" Adaptive Spatio-Temporal Refinement Module (ASTRM).

Reproduction of the module from "Precise Event Spotting in Sports Videos:
Solving Long-Range Dependency and Class Imbalance" (CVPR), built to drop into
the E2E-Spot RegNet backbone in place of the Gated Shift Module (GSM).

The backbone processes a clip as a batch of 2D frames, i.e. tensors of shape
(B*T, C, H, W).  ASTRM needs the temporal axis, so it folds the batch back into
(B, C, T, H, W), applies the refinement, and unfolds again.  This mirrors how
GSM/TSM rely on a known ``n_segment`` (= clip_len).

Refinement (paper eq.1):
    Psi(x) = ((x * (1 + F_s(x))) * (1 + F_t(x))) convolved with G_t(x)
where * before G_t is the Hadamard product and the final operation is temporal
convolution with the dynamic kernel produced by the global-temporal block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm


class LocalSpatial(nn.Module):
    """F_s(x): CBAM-style spatial attention (paper eq.2).

    Pool over the channel axis (avg & max), concat, 7x7 conv, sigmoid -> a
    per-location spatial weight broadcast over channels.  Operates per-frame
    (kernel is 1 along the temporal axis).
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(
            2, 1, kernel_size=(1, kernel_size, kernel_size),
            padding=(0, padding, padding), bias=False)

    def forward(self, x):
        # x: (B, C, T, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)            # (B,1,T,H,W)
        max_out = torch.amax(x, dim=1, keepdim=True)            # (B,1,T,H,W)
        attn = torch.cat([avg_out, max_out], dim=1)            # (B,2,T,H,W)
        attn = self.conv(attn)
        return torch.sigmoid(attn)                              # (B,1,T,H,W)


class LocalTemporal(nn.Module):
    """F_t(x): temporal attention (paper eq.3, TAM-inspired).

    3x1x1 temporal conv (C -> C/b), ReLU, BN, then 1x1x1 conv (C/b -> C),
    sigmoid.  Purely temporal: the spatial kernel is 1x1 so each spatial
    location keeps its own temporal response.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.conv1 = nn.Conv3d(
            channels, hidden, kernel_size=(3, 1, 1), padding=(1, 0, 0),
            bias=False)
        self.bn = nn.BatchNorm3d(hidden)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            hidden, channels, kernel_size=(1, 1, 1), bias=False)

    def forward(self, x):
        # x: (B, C, T, H, W)
        out = self.conv1(x)
        out = self.bn(self.relu(out))
        out = self.conv2(out)
        return torch.sigmoid(out)                              # (B,C,T,H,W)


class GlobalTemporal(nn.Module):
    """G_t(x): adaptive temporal kernel (paper eq.4, TAM-inspired).

    Spatial GAP -> (B,C,T); two FCs over the temporal axis (T -> 2T -> K);
    softmax over K -> a per-(batch,channel) temporal conv kernel.  The paper
    writes the output as CxK without giving K.  We back-inferred K from the
    paper's reported 60.25 GFLOPs: a small K (3-7) reproduces it within ~7%,
    whereas K=clip_len(=128) inflates compute to ~1.42x.  Default K=7.
    """

    def __init__(self, clip_len, kernel_size=7):
        super().__init__()
        self.clip_len = clip_len
        self.kernel_size = 7 if kernel_size is None else kernel_size
        self.fc1 = nn.Linear(clip_len, 2 * clip_len)
        self.fc2 = nn.Linear(2 * clip_len, self.kernel_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        g = x.mean(dim=(3, 4))                                  # (B,C,T)
        g = self.relu(self.fc1(g))                              # (B,C,2T)
        g = self.fc2(g)                                         # (B,C,K)
        kernel = torch.softmax(g, dim=-1)                      # (B,C,K)
        return kernel

    @staticmethod
    def apply_kernel(feat, kernel):
        """Depthwise temporal convolution with a per-(B,C) dynamic kernel.

        feat:   (B, C, T, H, W)
        kernel: (B, C, K)  (softmax-normalized, sums to 1 over K)
        """
        B, C, T, H, W = feat.shape
        K = kernel.shape[-1]
        # Treat (B*C) as conv groups, T as the conv axis, H*W as "width".
        x = feat.reshape(1, B * C, T, H * W)                   # (1, B*C,T,HW)
        pad_left = (K - 1) // 2
        pad_right = K // 2
        x = F.pad(x, (0, 0, pad_left, pad_right))
        w = kernel.reshape(B * C, 1, K, 1).to(x.dtype)         # (B*C,1,K,1)
        out = F.conv2d(x, w, groups=B * C)
        return out.reshape(B, C, T, H, W)


class ASTRM(nn.Module):
    """Full ASTRM refinement applied to a (B*T, C, H, W) feature map."""

    def __init__(self, channels, clip_len, reduction=4, spatial_kernel=7,
                 temporal_kernel=7):
        super().__init__()
        self.clip_len = clip_len
        self.local_spatial = LocalSpatial(spatial_kernel)
        self.local_temporal = LocalTemporal(channels, reduction)
        self.global_temporal = GlobalTemporal(clip_len, temporal_kernel)

    def forward(self, x):
        # x: (B*T, C, H, W) -> (B, C, T, H, W)
        nt, c, h, w = x.shape
        t = self.clip_len
        b = nt // t
        x5 = x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

        fs = self.local_spatial(x5)            # (B,1,T,H,W)
        ft = self.local_temporal(x5)           # (B,C,T,H,W)
        gk = self.global_temporal(x5)          # (B,C,K)

        out = x5 * (1.0 + fs)
        out = out * (1.0 + ft)
        out = GlobalTemporal.apply_kernel(out, gk)

        # back to (B*T, C, H, W)
        return out.permute(0, 2, 1, 3, 4).contiguous().view(nt, c, h, w)


class ASTRMWrap(nn.Module):
    """Run a conv (the bottleneck's first conv) then ASTRM, matching the
    paper's "add ASTRM after the first conv" placement."""

    def __init__(self, conv, out_channels, clip_len, **kwargs):
        super().__init__()
        self.conv = conv
        self.astrm = ASTRM(out_channels, clip_len, **kwargs)

    def forward(self, x):
        return self.astrm(self.conv(x))


def _conv_out_channels(conv):
    if isinstance(conv, nn.Conv2d):
        return conv.out_channels
    # timm ConvNormAct / ConvBnAct wraps a .conv
    if hasattr(conv, 'conv') and isinstance(conv.conv, nn.Conv2d):
        return conv.conv.out_channels
    if hasattr(conv, 'out_channels'):
        return conv.out_channels
    raise NotImplementedError('Cannot infer out_channels from {}'.format(
        type(conv)))


def make_astrm(net, clip_len, **astrm_kwargs):
    """Insert ASTRM after conv1 of every bottleneck block.

    Mirrors model.shift.make_temporal_shift but installs ASTRM instead of GSM.
    Currently supports timm RegNet (the backbone used in the paper).
    """

    def wrap_block_conv1(block):
        out_ch = _conv_out_channels(block.conv1)
        block.conv1 = ASTRMWrap(block.conv1, out_ch, clip_len, **astrm_kwargs)

    if isinstance(net, timm.models.regnet.RegNet):
        for stage in [net.s1, net.s2, net.s3, net.s4]:
            for block in stage.children():
                wrap_block_conv1(block)
            print('=> Inserted ASTRM into RegNet stage ({} blocks)'.format(
                len(list(stage.children()))))
    elif isinstance(net, torchvision.models.ResNet):
        for stage in [net.layer1, net.layer2, net.layer3, net.layer4]:
            for block in stage.children():
                wrap_block_conv1(block)
    else:
        raise NotImplementedError(
            'ASTRM insertion not implemented for {}'.format(type(net)))
