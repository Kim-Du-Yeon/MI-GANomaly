import os
import sys

import optuna
import torch
import wandb
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
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

# NOTE: optuna_integration.WeightsAndBiasesCallback은 trial이 끝난 "후"에만 호출되는
# 콜백이라 trial 진행 중 epoch별 실시간 로깅이 불가능함. 대신 train.py와 동일한 방식으로
# trial마다 wandb.init/log/finish를 직접 호출해 epoch 단위 곡선까지 기록한다.

N_TRIALS = 20
N_EPOCHS = 50
PATIENCE = 15


def build_trial_opt(trial):
    opt = get_options()
    opt.dataroot = 'mi_ganomaly/data/train_augmented'
    opt.workers = 0  # Windows 멀티프로세싱 이슈 방지
    opt.isize = 64
    opt.batchsize = 64
    opt.n_epochs = N_EPOCHS
    opt.patience = PATIENCE
    opt.mask_type = 'patch'
    opt.mask_size = 8

    opt.w_ctx = trial.suggest_float('w_ctx', 1.0, 500.0, log=True)
    opt.w_enc = trial.suggest_float('w_enc', 1.0, 50.0, log=True)
    opt.recon_alpha = trial.suggest_float('recon_alpha', 0.3, 1.0)
    opt.lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    opt.mask_ratio = trial.suggest_float('mask_ratio', 0.1, 0.4)

    return opt


def train_one_trial(opt, trial, device):
    set_seed(42)

    train_loader, _ = get_dataloader(opt, 'train')
    n_train = len(train_loader.dataset)
    print(f'[data check] dataroot={opt.dataroot} train/normal images={n_train}')
    if n_train == 0:
        raise FileNotFoundError(
            f'train 이미지를 찾을 수 없습니다: {os.path.join(opt.dataroot, "train", "normal")} '
            f'경로와 이미지 존재 여부를 확인하세요.'
        )

    test_loader, test_labels = get_dataloader(opt, 'test')

    model = GANomaly(opt)
    model.criterion = TotalLoss(opt, recon_alpha=opt.recon_alpha)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),
                            weight_decay=opt.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=1e-6)
    masker = MAEMasker(patch_size=opt.mask_size, mask_ratio=opt.mask_ratio, isize=opt.isize).to(device)

    best_auc = -1.0
    epochs_no_improve = 0

    for epoch in range(opt.n_epochs):
        model.train()
        for x, _ in train_loader:
            x = x.to(device)
            masked_x, _ = masker(x)

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

        wandb.log({'epoch': epoch + 1, 'test_auc': auc,
                   'learning_rate': optimizer.param_groups[0]['lr']}, step=epoch + 1)

        if auc > best_auc:
            best_auc = auc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        trial.report(auc, epoch + 1)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if epochs_no_improve >= opt.patience:
            break

    return best_auc


def objective(trial):
    opt = build_trial_opt(trial)
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

    wandb.init(project=f'{opt.wandb_project}-optuna', name=f'trial_{trial.number}',
               config={**vars(opt), 'trial_number': trial.number}, reinit=True)

    try:
        best_auc = train_one_trial(opt, trial, device)
    finally:
        wandb.finish()

    return best_auc


def print_top_trials(study, top_k=5):
    completed = [t for t in study.trials if t.value is not None]
    top = sorted(completed, key=lambda t: t.value, reverse=True)[:top_k]

    print(f"{'rank':>4} {'trial':>5} {'AUC':>8} {'w_ctx':>8} {'w_enc':>8} "
          f"{'recon_alpha':>11} {'lr':>10} {'mask_ratio':>10}")
    for rank, t in enumerate(top, start=1):
        p = t.params
        print(f"{rank:>4} {t.number:>5} {t.value:>8.4f} {p['w_ctx']:>8.2f} {p['w_enc']:>8.2f} "
              f"{p['recon_alpha']:>11.3f} {p['lr']:>10.2e} {p['mask_ratio']:>10.3f}")


def main():
    sampler = TPESampler(seed=42)
    pruner = MedianPruner()
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=N_TRIALS)

    print('--- Best trial ---')
    print(f'AUC: {study.best_value:.4f}')
    print(f'params: {study.best_params}')

    print()
    print('--- Top 5 trials ---')
    print_top_trials(study, top_k=5)


if __name__ == '__main__':
    main()
