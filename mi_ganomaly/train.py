import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import wandb  # pip install wandb 필요 (requirements.txt 참고), 미설치 시 import 단계에서 실패
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.ganomaly import GANomaly
from models.loss import TotalLoss
from options import get_options
from utils.dataloader import get_dataloader
from utils.masking import MAEMasker
from utils.reproducibility import set_seed


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
    set_seed(42)
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

    if opt.use_wandb:
        run_name = os.path.basename(os.path.normpath(opt.save_dir))
        wandb.init(project=opt.wandb_project, name=run_name, config=vars(opt))

    train_loader, _ = get_dataloader(opt, 'train')
    test_loader, test_labels = get_dataloader(opt, 'test')

    model = GANomaly(opt)
    model.criterion = TotalLoss(opt, recon_alpha=0.5)  # Phase 3: MSE 0.5 + SSIM 0.5
    model = model.to(device)
    # Weight Decay: L2 정규화로 가중치 폭발 방지
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),
                            weight_decay=opt.weight_decay)
    # CosineAnnealing: 학습 후반 lr 점진적 감소로 미세 수렴
    scheduler = CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=1e-6)

    masker = None
    if opt.mask_type != 'none':
        masker = MAEMasker(patch_size=opt.mask_size, mask_ratio=opt.mask_ratio, isize=opt.isize).to(device)

    ckpt_dir = os.path.join(opt.save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    history = {'total': [], 'recon': [], 'ctx': [], 'enc': []}
    best_auc = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    converged_epoch = opt.n_epochs

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
            # Gradient Clipping: G+E+D 복잡 구조 gradient 폭발 방지
            clip_grad_norm_(model.parameters(), max_norm=opt.max_norm)
            optimizer.step()

            running['total'] += total.item()
            running['recon'] += l_recon.item()
            running['ctx'] += l_ctx.item()
            running['enc'] += l_enc.item()

            if (i + 1) % opt.print_freq == 0:
                print(f'[Epoch {epoch + 1}/{opt.n_epochs}][{i + 1}/{len(train_loader)}] '
                      f'total={total.item():.4f} recon={l_recon.item():.4f} '
                      f'ctx={l_ctx.item():.4f} enc={l_enc.item():.4f}')

        scheduler.step()

        n_batches = max(len(train_loader), 1)
        for k in history:
            history[k].append(running[k] / n_batches)

        print(f'[Epoch {epoch + 1}/{opt.n_epochs}] avg total={history["total"][-1]:.4f} '
              f'recon={history["recon"][-1]:.4f} ctx={history["ctx"][-1]:.4f} enc={history["enc"][-1]:.4f}')

        scores = compute_latent_scores(model, test_loader, device)
        auc = roc_auc_score(test_labels.numpy(), scores.numpy())
        print(f'[Epoch {epoch + 1}/{opt.n_epochs}] test AUC={auc:.4f}')

        if opt.use_wandb:
            wandb.log({
                'total_loss': history['total'][-1],
                'recon_loss': history['recon'][-1],
                'ctx_loss': history['ctx'][-1],
                'enc_loss': history['enc'][-1],
                'test_auc': auc,
                'learning_rate': optimizer.param_groups[0]['lr'],
            }, step=epoch + 1)

        torch.save(model.state_dict(), os.path.join(ckpt_dir, f'epoch_{epoch + 1}.pth'))

        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(opt.save_dir, 'best_model.pth'))
            if opt.use_wandb:
                wandb.log({'best_auc': auc}, step=epoch + 1)
        else:
            epochs_no_improve += 1

        # Early Stopping: 최적 수렴점 자동 탐지, 과적합 방지
        if epochs_no_improve >= opt.patience:
            converged_epoch = epoch + 1
            print(f'[Early Stopping] AUC 개선 없음 {opt.patience} epoch 연속 '
                  f'(best_epoch={best_epoch}, best_auc={best_auc:.4f}) -> epoch {converged_epoch}에서 종료')
            break
    else:
        converged_epoch = opt.n_epochs

    plot_loss_curve(history, opt.save_dir)

    with open(os.path.join(opt.save_dir, 'convergence.txt'), 'w') as f:
        f.write(f'converged_epoch={converged_epoch}\nbest_epoch={best_epoch}\nbest_auc={best_auc:.4f}\n')
    print(f'[Convergence] converged_epoch={converged_epoch} best_epoch={best_epoch} best_auc={best_auc:.4f}')

    if opt.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
