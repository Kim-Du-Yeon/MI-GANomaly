import torch.nn as nn
from pytorch_msssim import ssim


class ReconLoss(nn.Module):
    """MSE + SSIM. alpha=1.0 -> MSE only (Phase 1), alpha=0.5 -> blend (Phase 3)."""

    def __init__(self, alpha=0.5):
        super(ReconLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, real, fake):
        loss_mse = self.mse(fake, real)
        if self.alpha >= 1.0:
            return loss_mse
        loss_ssim = 1 - ssim(fake, real, data_range=2.0, size_average=True)
        return self.alpha * loss_mse + (1 - self.alpha) * loss_ssim


class ContextualLoss(nn.Module):
    def __init__(self):
        super(ContextualLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, feat_real, feat_fake):
        return self.l1(feat_fake, feat_real)


class EncoderLoss(nn.Module):
    def __init__(self):
        super(EncoderLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, z_real, z_fake):
        return self.mse(z_fake, z_real)


class TotalLoss(nn.Module):
    def __init__(self, opt, recon_alpha=0.5):
        super(TotalLoss, self).__init__()
        self.w_recon = opt.w_recon
        self.w_ctx = opt.w_ctx
        self.w_enc = opt.w_enc

        self.recon_loss = ReconLoss(alpha=recon_alpha)
        self.ctx_loss = ContextualLoss()
        self.enc_loss = EncoderLoss()

    def forward(self, real, fake, feat_real, feat_fake, z_real, z_fake):
        l_recon = self.recon_loss(real, fake)
        l_ctx = self.ctx_loss(feat_real, feat_fake)
        l_enc = self.enc_loss(z_real, z_fake)

        total = self.w_recon * l_recon + self.w_ctx * l_ctx + self.w_enc * l_enc
        return total, l_recon, l_ctx, l_enc
