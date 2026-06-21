import argparse


def get_options():
    parser = argparse.ArgumentParser(description='MI-GANomaly options')

    # 모델
    parser.add_argument('--nz', type=int, default=500, help='latent vector size')
    parser.add_argument('--isize', type=int, default=64, help='input image size')
    parser.add_argument('--nc', type=int, default=3, help='input image channels')
    parser.add_argument('--ndf', type=int, default=64, help='discriminator feature map size')
    parser.add_argument('--ngf', type=int, default=64, help='generator feature map size')
    parser.add_argument('--extralayers', type=int, default=0, help='number of extra layers')

    # 학습
    parser.add_argument('--batchsize', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='adam beta1')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--device', type=str, default='cuda', help='device: cuda or cpu')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='optimizer weight decay (L2)')
    parser.add_argument('--max_norm', type=float, default=1.0, help='gradient clipping max norm')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience (epochs, monitor=AUC)')

    # 데이터
    parser.add_argument('--dataset', type=str, default='elpv', help='dataset name')
    parser.add_argument('--dataroot', type=str, default='./data', help='dataset root path')
    parser.add_argument('--workers', type=int, default=4, help='number of dataloader workers')

    # 마스킹 (Phase2 대비)
    # mask_size는 isize보다 작아야 함 (MAEMasker.__init__에서 자동 ValueError 검증,
    # mask_ratio*n_patches가 0으로 내려가면 max(1,n_mask) 보정 + Warning 출력)
    # isize=64 기준 patch_size=8 -> n_patches=64로 더 세밀한 ratio 제어 가능 (isize=32 시절보다 개선)
    parser.add_argument('--mask_size', type=int, default=8, help='mask patch size')
    parser.add_argument('--mask_ratio', type=float, default=0.2, help='mask ratio')
    parser.add_argument('--mask_type', type=str, default='patch', help='mask type')

    # 손실 가중치
    parser.add_argument('--w_recon', type=float, default=1.0, help='reconstruction loss weight')
    parser.add_argument('--w_ctx', type=float, default=1.0, help='contextual loss weight')
    parser.add_argument('--w_enc', type=float, default=1.0, help='encoder loss weight')
    parser.add_argument('--recon_alpha', type=float, default=1.0,
                         help='ReconLoss MSE/SSIM blend: 1.0=MSE only, 0.5=MSE+SSIM')

    # 평가 (Threshold)
    parser.add_argument('--k', type=float, default=2.0, help='auto threshold: mu + k*sigma')

    # WandB
    parser.add_argument('--use_wandb', action=argparse.BooleanOptionalAction, default=True,
                         help='enable WandB experiment logging')
    parser.add_argument('--wandb_project', type=str, default='MI-GANomaly', help='WandB project name')

    # 기타
    parser.add_argument('--save_dir', type=str, default='./output', help='output save directory')
    parser.add_argument('--resume', type=str, default='', help='checkpoint path to resume from')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')

    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()
    print(opt)
