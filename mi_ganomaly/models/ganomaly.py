import torch
import torch.nn as nn

from .networks import Encoder, Decoder, Discriminator
from .loss import TotalLoss


class GANomaly(nn.Module):
    def __init__(self, opt):
        super(GANomaly, self).__init__()
        self.e1 = Encoder(opt.isize, opt.nc, opt.ndf, opt.nz, opt.extralayers)
        self.g = Decoder(opt.isize, opt.nc, opt.ngf, opt.nz, opt.extralayers)
        self.e2 = Encoder(opt.isize, opt.nc, opt.ndf, opt.nz, opt.extralayers)
        self.d = Discriminator(opt.isize, opt.nc, opt.ndf, opt.extralayers)

        self.criterion = TotalLoss(opt)

    def forward(self, x):
        z = self.e1(x)
        x_hat = self.g(z)
        z_hat = self.e2(x_hat)

        score, feat_real = self.d(x)
        _, feat_fake = self.d(x_hat)

        return x_hat, z, z_hat, feat_real, feat_fake

    @torch.no_grad()
    def anomaly_score(self, x):
        x_hat, z, z_hat, feat_real, feat_fake = self.forward(x)
        total, l_recon, l_ctx, l_enc = self.criterion(x, x_hat, feat_real, feat_fake, z, z_hat)
        return total
