import os
import sys

import torch
import wandb
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ganomaly import GANomaly
from models.loss import TotalLoss
from options import get_options
from utils.dataloader import get_dataloader
from utils.masking import MAEMasker
from utils.reproducibility import set_seed

N_EPOCHS = 50
PATIENCE = 15

COMBINATIONS = [
    ('single_8', [8]),
    ('single_16', [16]),
    ('single_32', [32]),
    ('multi_8_16', [8, 16]),
    ('multi_8_32', [8, 32]),
    ('multi_8_16_32', [8, 16, 32]),
]


def build_opt():
    opt = get_options()
    opt.dataroot = 'mi_ganomaly/data/train_augmented'
    opt.isize = 64
    opt.batchsize = 64
    opt.n_epochs = N_EPOCHS
    opt.patience = PATIENCE
    opt.w_ctx = 8.46
    opt.w_enc = 16.31
    opt.recon_alpha = 0.793
    opt.lr = 1.16e-05
    opt.mask_ratio = 0.288
    opt.workers = 0  # Windows 멀티프로세싱 이슈 방지
    return opt


def apply_multiscale_mask(x, maskers):
    combined_mask = None
    for masker in maskers:
        _, mask = masker(x)
        combined_mask = mask if combined_mask is None else (combined_mask | mask)
    return x.masked_fill(combined_mask, 0.0)


def run_combination(name, mask_sizes, device):
    set_seed(42)
    opt = build_opt()

    train_loader, _ = get_dataloader(opt, 'train')
    n_train = len(train_loader.dataset)
    print(f'[data check] dataroot={opt.dataroot} train/normal images={n_train}')
    if n_train == 0:
        raise FileNotFoundError(f'train 이미지를 찾을 수 없습니다: {opt.dataroot}')

    test_loader, test_labels = get_dataloader(opt, 'test')

    model = GANomaly(opt)
    model.criterion = TotalLoss(opt, recon_alpha=opt.recon_alpha)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),
                            weight_decay=opt.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=1e-6)
    maskers = [MAEMasker(patch_size=ms, mask_ratio=opt.mask_ratio, isize=opt.isize).to(device)
               for ms in mask_sizes]

    wandb.init(project='MI-GANomaly-masking', name=name,
               config={**vars(opt), 'mask_sizes': mask_sizes}, reinit=True)

    best_auc = -1.0
    epochs_no_improve = 0

    for epoch in range(opt.n_epochs):
        model.train()
        for x, _ in train_loader:
            x = x.to(device)
            masked_x = apply_multiscale_mask(x, maskers)

            x_hat, z, z_hat, feat_real, feat_fake = model(masked_x)
            total, _, _, _ = model.criterion(x, x_hat, feat_real, feat_fake, z, z_hat)

            optimizer.zero_grad()
            total.backward()
            clip_grad_norm_(model.parameters(), max_norm=opt.max_norm)
            optimizer.step()

        scheduler.step()

        model.eval()
        scores = []
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                _, z, z_hat, _, _ = model(x)
                scores.append((z - z_hat).pow(2).mean(dim=[1, 2, 3]).cpu())
        auc = roc_auc_score(test_labels.numpy(), torch.cat(scores).numpy())

        wandb.log({'epoch': epoch + 1, 'test_auc': auc}, step=epoch + 1)
        print(f'[{name}][epoch {epoch + 1}/{opt.n_epochs}] test_auc={auc:.4f}')

        if auc > best_auc:
            best_auc = auc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= opt.patience:
            print(f'[{name}] early stopping at epoch {epoch + 1} (best_auc={best_auc:.4f})')
            break

    wandb.finish()
    return best_auc


def print_table(results):
    print(f"{'combination':>16} {'mask_sizes':>18} {'best_auc':>10}")
    for name, mask_sizes, auc in results:
        print(f"{name:>16} {str(mask_sizes):>18} {auc:>10.4f}")


def append_to_log(results, log_path):
    best = max(results, key=lambda r: r[2])
    lines = ['\n## Multi-scale 마스킹 실험 결과\n\n']
    lines.append('| 조합 | mask_sizes | best AUC |\n')
    lines.append('|---|---|---|\n')
    for name, mask_sizes, auc in results:
        lines.append(f'| {name} | {mask_sizes} | {auc:.4f} |\n')
    lines.append(f'\n**Best 조합**: {best[0]} ({best[1]}) AUC={best[2]:.4f}\n')

    with open(log_path, 'a', encoding='utf-8') as f:
        f.writelines(lines)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = []
    for name, mask_sizes in COMBINATIONS:
        print(f'--- {name} {mask_sizes} ---')
        auc = run_combination(name, mask_sizes, device)
        results.append((name, mask_sizes, auc))

    print()
    print_table(results)

    log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiments_log.md'))
    append_to_log(results, log_path)
    print(f'experiments_log.md updated: {log_path}')


if __name__ == '__main__':
    main()
