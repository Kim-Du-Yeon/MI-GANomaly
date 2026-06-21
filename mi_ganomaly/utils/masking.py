import warnings

import torch
import torch.nn as nn


class MAEMasker(nn.Module):
    def __init__(self, patch_size=16, mask_ratio=0.2, isize=32):
        super(MAEMasker, self).__init__()

        if patch_size >= isize:
            raise ValueError(
                f'patch_size({patch_size}) must be smaller than isize({isize})'
            )

        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        n_patches = (isize // patch_size) ** 2
        if int(n_patches * mask_ratio) == 0:
            warnings.warn(
                f'mask_ratio({mask_ratio}) * n_patches({n_patches}) rounds down to 0 '
                f'masked patches; falling back to max(1, n_mask) so at least 1 patch is masked.'
            )

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, 'H, W must be divisible by patch_size'

        nh, nw = H // p, W // p
        n_patches = nh * nw
        n_mask = max(1, int(n_patches * self.mask_ratio))

        noise = torch.rand(B, n_patches, device=x.device)
        mask_idx = noise.argsort(dim=1)[:, :n_mask]

        patch_mask = torch.zeros(B, n_patches, dtype=torch.bool, device=x.device)
        patch_mask.scatter_(1, mask_idx, True)
        patch_mask = patch_mask.view(B, 1, nh, nw)

        mask = patch_mask.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)

        masked_x = x.masked_fill(mask, 0.0)
        return masked_x, mask
