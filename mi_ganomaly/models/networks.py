import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, isize=32, nc=3, ndf=64, nz=500, extralayers=0):
        super(Encoder, self).__init__()
        assert isize % 16 == 0, 'isize has to be a multiple of 16'

        main = nn.Sequential()
        main.add_module('initial-conv', nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu', nn.LeakyReLU(0.2, inplace=True))

        csize, cndf = isize / 2, ndf

        for t in range(extralayers):
            main.add_module(f'extra-{t}-conv', nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            # GroupNorm: 소배치(64)에서 BatchNorm 통계 불안정 문제 해결
            main.add_module(f'extra-{t}-bn', nn.GroupNorm(8, cndf))
            main.add_module(f'extra-{t}-relu', nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat, out_feat = cndf, cndf * 2
            main.add_module(f'pyramid-{in_feat}-{out_feat}-conv', nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            # GroupNorm: 소배치(64)에서 BatchNorm 통계 불안정 문제 해결
            main.add_module(f'pyramid-{out_feat}-bn', nn.GroupNorm(8, out_feat))
            main.add_module(f'pyramid-{out_feat}-relu', nn.LeakyReLU(0.2, inplace=True))
            cndf, csize = cndf * 2, csize / 2

        # Dropout: 정상만 학습하는 Semi-Supervised 구조 과적합 방지 (마지막 두 Conv 이후)
        main.add_module('dropout-1', nn.Dropout(p=0.3))

        main.add_module('final-conv', nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))

        # Dropout: 정상만 학습하는 Semi-Supervised 구조 과적합 방지 (마지막 두 Conv 이후)
        main.add_module('dropout-2', nn.Dropout(p=0.3))

        self.main = main

    def forward(self, x):
        return self.main(x)


class Decoder(nn.Module):
    def __init__(self, isize=32, nc=3, ngf=64, nz=500, extralayers=0):
        super(Decoder, self).__init__()
        assert isize % 16 == 0, 'isize has to be a multiple of 16'

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf, tisize = cngf * 2, tisize * 2

        main = nn.Sequential()
        main.add_module('initial-convt', nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        # GroupNorm: 소배치(64)에서 BatchNorm 통계 불안정 문제 해결
        main.add_module('initial-bn', nn.GroupNorm(8, cngf))
        main.add_module('initial-relu', nn.ReLU(True))

        csize = 4
        while csize < isize // 2:
            main.add_module(f'pyramid-{cngf}-convt', nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            # GroupNorm: 소배치(64)에서 BatchNorm 통계 불안정 문제 해결
            main.add_module(f'pyramid-{cngf // 2}-bn', nn.GroupNorm(8, cngf // 2))
            main.add_module(f'pyramid-{cngf // 2}-relu', nn.ReLU(True))
            cngf, csize = cngf // 2, csize * 2

        for t in range(extralayers):
            main.add_module(f'extra-{t}-conv', nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            # GroupNorm: 소배치(64)에서 BatchNorm 통계 불안정 문제 해결
            main.add_module(f'extra-{t}-bn', nn.GroupNorm(8, cngf))
            main.add_module(f'extra-{t}-relu', nn.LeakyReLU(0.2, inplace=True))

        main.add_module('final-convt', nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final-tanh', nn.Tanh())

        self.main = main

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self, isize=32, nc=3, ndf=64, extralayers=0):
        super(Discriminator, self).__init__()
        assert isize % 16 == 0, 'isize has to be a multiple of 16'

        features = nn.Sequential()
        features.add_module('initial-conv', nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        features.add_module('initial-relu', nn.LeakyReLU(0.2, inplace=True))

        csize, cndf = isize / 2, ndf

        for t in range(extralayers):
            features.add_module(f'extra-{t}-conv', nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            # GroupNorm: 소배치(64)에서 BatchNorm 통계 불안정 문제 해결
            features.add_module(f'extra-{t}-bn', nn.GroupNorm(8, cndf))
            features.add_module(f'extra-{t}-relu', nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat, out_feat = cndf, cndf * 2
            features.add_module(f'pyramid-{in_feat}-{out_feat}-conv', nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            # GroupNorm: 소배치(64)에서 BatchNorm 통계 불안정 문제 해결
            features.add_module(f'pyramid-{out_feat}-bn', nn.GroupNorm(8, out_feat))
            features.add_module(f'pyramid-{out_feat}-relu', nn.LeakyReLU(0.2, inplace=True))
            cndf, csize = cndf * 2, csize / 2

        self.features = features
        self.classifier = nn.Sequential(
            nn.Conv2d(cndf, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.features(x)
        out = self.classifier(feat)
        return out.view(-1, 1).squeeze(1), feat
