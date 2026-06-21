import copy
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import collect_scores, compute_metrics, normalize_scores, sweep_threshold
from models.ganomaly import GANomaly
from models.loss import TotalLoss
from options import get_options
from utils.dataloader import get_dataloader
from utils.masking import MAEMasker

MASK_SIZES = [8, 16, 32]
MASK_RATIOS = [0.1, 0.2, 0.3]


def make_dummy_dataset(dataroot, n_train=10, n_test_normal=6, n_test_anomaly=6, img_size=40):
    for split, cls, n in [('train', 'normal', n_train),
                          ('test', 'normal', n_test_normal),
                          ('test', 'anomaly', n_test_anomaly)]:
        d = os.path.join(dataroot, split, cls)
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            arr = (np.random.rand(img_size, img_size, 3) * 255).astype('uint8')
            Image.fromarray(arr).save(os.path.join(d, f'{i}.png'))


def run_experiment(opt, mask_size, mask_ratio, device):
    opt = copy.deepcopy(opt)
    opt.mask_size = mask_size
    opt.mask_ratio = mask_ratio
    opt.mask_type = 'patch'

    train_loader, _ = get_dataloader(opt, 'train')
    test_loader, _ = get_dataloader(opt, 'test')

    model = GANomaly(opt)
    model.criterion = TotalLoss(opt, recon_alpha=1.0)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    masker = MAEMasker(patch_size=opt.mask_size, mask_ratio=opt.mask_ratio, isize=opt.isize).to(device)

    for _ in range(opt.n_epochs):
        model.train()
        for x, _ in train_loader:
            x = x.to(device)
            masked_x, _ = masker(x)

            x_hat, z, z_hat, feat_real, feat_fake = model(masked_x)
            total, _, _, _ = model.criterion(x, x_hat, feat_real, feat_fake, z, z_hat)

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

    raw_scores, labels = collect_scores(model, test_loader, device)
    scores = normalize_scores(raw_scores)
    auc = roc_auc_score(labels, scores)
    th = sweep_threshold(scores, labels)
    metrics = compute_metrics(scores, labels, th)

    return {'auc': auc, 'f1': metrics['f1']}


def print_table(results):
    print(f"{'mask_size':>10} {'mask_ratio':>11} {'AUC':>8} {'F1':>8}")
    for (ms, mr), r in results.items():
        print(f"{ms:>10} {mr:>11} {r['auc']:>8.4f} {r['f1']:>8.4f}")


def append_to_log(results, log_path, best_key):
    lines = ['\n## Phase 2 - 마스킹 크기/비율 Ablation (더미 데이터)\n\n']
    lines.append('| mask_size | mask_ratio | AUC | F1 |\n')
    lines.append('|---|---|---|---|\n')
    for (ms, mr), r in results.items():
        lines.append(f"| {ms} | {mr} | {r['auc']:.4f} | {r['f1']:.4f} |\n")

    bms, bmr = best_key
    lines.append(f"\n**Best combo (F1 기준)**: mask_size={bms}, mask_ratio={bmr} "
                 f"(AUC={results[best_key]['auc']:.4f}, F1={results[best_key]['f1']:.4f})\n")
    lines.append('\n> 더미(랜덤 노이즈) 데이터 기준 동작 검증용 결과이며, 실제 ELPV 데이터로 재실험 필요.\n')

    with open(log_path, 'a', encoding='utf-8') as f:
        f.writelines(lines)


def main():
    opt = get_options()
    opt.n_epochs = 3
    opt.batchsize = 2
    opt.workers = 0
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

    tmp = tempfile.mkdtemp(prefix='ablation_masking_')
    opt.dataroot = os.path.join(tmp, 'data')
    make_dummy_dataset(opt.dataroot)

    results = {}
    for mask_size in MASK_SIZES:
        for mask_ratio in MASK_RATIOS:
            print(f'--- mask_size={mask_size} mask_ratio={mask_ratio} ---')
            results[(mask_size, mask_ratio)] = run_experiment(opt, mask_size, mask_ratio, device)

    shutil.rmtree(tmp)

    print_table(results)

    best_key = max(results, key=lambda k: results[k]['f1'])
    print(f"Best combo (F1): mask_size={best_key[0]}, mask_ratio={best_key[1]} -> {results[best_key]}")

    log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiments_log.md'))
    append_to_log(results, log_path, best_key)
    print(f'experiments_log.md updated: {log_path}')


if __name__ == '__main__':
    main()
