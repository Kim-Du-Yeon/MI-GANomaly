import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score
from torch import optim

from models.ganomaly import GANomaly
from models.loss import TotalLoss
from options import get_options
from utils.dataloader import get_dataloader
from utils.masking import MAEMasker


def compute_latent_scores(model, loader, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _, z, z_hat, _, _ = model(x)
            scores.append((z - z_hat).pow(2).mean(dim=[1, 2, 3]).cpu())
    return torch.cat(scores)


def plot_loss_curve(history, save_dir):
    epochs = range(1, len(history['total']) + 1)
    plt.figure()
    for key in ['total', 'recon', 'ctx', 'enc']:
        plt.plot(epochs, history[key], label=key)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()


def main():
    opt = get_options()
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

    train_loader, _ = get_dataloader(opt, 'train')
    test_loader, test_labels = get_dataloader(opt, 'test')

    model = GANomaly(opt)
    model.criterion = TotalLoss(opt, recon_alpha=1.0)  # Phase 1 baseline: MSE only
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    masker = None
    if opt.mask_type != 'none':
        masker = MAEMasker(patch_size=opt.mask_size, mask_ratio=opt.mask_ratio, isize=opt.isize).to(device)

    ckpt_dir = os.path.join(opt.save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    history = {'total': [], 'recon': [], 'ctx': [], 'enc': []}
    best_auc = -1.0

    for epoch in range(opt.n_epochs):
        model.train()
        running = {'total': 0.0, 'recon': 0.0, 'ctx': 0.0, 'enc': 0.0}

        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)
            masked_x = masker(x)[0] if masker is not None else x

            x_hat, z, z_hat, feat_real, feat_fake = model(masked_x)
            total, l_recon, l_ctx, l_enc = model.criterion(x, x_hat, feat_real, feat_fake, z, z_hat)

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            running['total'] += total.item()
            running['recon'] += l_recon.item()
            running['ctx'] += l_ctx.item()
            running['enc'] += l_enc.item()

            if (i + 1) % opt.print_freq == 0:
                print(f'[Epoch {epoch + 1}/{opt.n_epochs}][{i + 1}/{len(train_loader)}] '
                      f'total={total.item():.4f} recon={l_recon.item():.4f} '
                      f'ctx={l_ctx.item():.4f} enc={l_enc.item():.4f}')

        n_batches = max(len(train_loader), 1)
        for k in history:
            history[k].append(running[k] / n_batches)

        scores = compute_latent_scores(model, test_loader, device)
        auc = roc_auc_score(test_labels.numpy(), scores.numpy())
        print(f'[Epoch {epoch + 1}/{opt.n_epochs}] test AUC={auc:.4f}')

        torch.save(model.state_dict(), os.path.join(ckpt_dir, f'epoch_{epoch + 1}.pth'))

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(opt.save_dir, 'best_model.pth'))

    plot_loss_curve(history, opt.save_dir)


if __name__ == '__main__':
    main()
