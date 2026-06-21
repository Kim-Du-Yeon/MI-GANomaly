import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

# ===== 채택 기법 (도메인 근거는 experiments_log.md 참고) =====
TECHNIQUES = {
    'hflip': transforms.RandomHorizontalFlip(p=0.5),                              # 패널 좌우 대칭 가능
    'vflip': transforms.RandomVerticalFlip(p=0.5),                                # 패널 상하 대칭 가능
    'rotate': transforms.RandomRotation(degrees=15),                             # 촬영 각도 미세 변동
    'blur': transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),            # 카메라 초점 변동
    'colorjitter': transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.0),  # 조명 변동만
}

# ===== 금지 기법 =====
# EXCLUDED - RandomErasing: 인위적 결함 생성 위험
# EXCLUDED - CutMix/MixUp: 존재하지 않는 패턴 생성
# EXCLUDED - ElasticTransform: 패널 격자 구조 파괴
# EXCLUDED - ColorJitter(saturation): 색조 변경은 결함 판단 기준 훼손
# EXCLUDED - Rotation>15deg: 패널 직사각형 구조 비현실적 왜곡

IMG_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp')


def _list_images(img_dir):
    paths = []
    for ext in IMG_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(img_dir, ext)))
    return sorted(paths)


def offline_augment(src_dir, dst_dir, target_multiplier=6):
    n_tech = len(TECHNIQUES)
    assert target_multiplier - 1 == n_tech, (
        f'target_multiplier({target_multiplier})는 원본 1 + 기법 수({n_tech})와 같아야 '
        f'기법별로 균등하게 배분됩니다.'
    )

    os.makedirs(dst_dir, exist_ok=True)
    src_paths = _list_images(src_dir)

    for i, p in enumerate(src_paths, start=1):
        Image.open(p).save(os.path.join(dst_dir, f'orig_{i:04d}_original.png'))

    for tech_name, tech in TECHNIQUES.items():
        for i, p in enumerate(src_paths, start=1):
            img = Image.open(p)
            aug = tech(img)
            aug.save(os.path.join(dst_dir, f'orig_{i:04d}_{tech_name}.png'))

    return len(src_paths) * target_multiplier


def plot_pixel_distribution(img_dir, save_path, title):
    pixels = [np.asarray(Image.open(p).convert('L'), dtype=np.float32).ravel()
              for p in _list_images(img_dir)]
    pixels = np.concatenate(pixels)

    plt.figure(figsize=(6, 4))
    plt.hist(pixels, bins=50, color='steelblue')
    plt.xlabel('pixel intensity (0-255)')
    plt.ylabel('count')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_technique_samples(src_dir, save_dir, n_samples=5):
    src_paths = _list_images(src_dir)[:n_samples]
    os.makedirs(save_dir, exist_ok=True)

    for tech_name, tech in TECHNIQUES.items():
        fig, axes = plt.subplots(1, len(src_paths), figsize=(3 * len(src_paths), 3))
        for ax, p in zip(axes, src_paths):
            img = Image.open(p)
            aug = tech(img)
            ax.imshow(aug, cmap='gray' if aug.mode == 'L' else None)
            ax.axis('off')
        fig.suptitle(tech_name)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'samples_{tech_name}.png'))
        plt.close(fig)


if __name__ == '__main__':
    SRC_DIR = 'mi_ganomaly/data/train/normal'
    DST_DIR = 'mi_ganomaly/data/train/normal_augmented'
    OUT_DIR = 'output/augmentation'

    os.makedirs(OUT_DIR, exist_ok=True)

    n_before = len(_list_images(SRC_DIR))
    offline_augment(SRC_DIR, DST_DIR, target_multiplier=6)
    n_after = len(_list_images(DST_DIR))

    plot_pixel_distribution(SRC_DIR, os.path.join(OUT_DIR, 'pixel_dist_before.png'),
                             'Pixel Distribution - Before Augmentation')
    plot_pixel_distribution(DST_DIR, os.path.join(OUT_DIR, 'pixel_dist_after.png'),
                             'Pixel Distribution - After Augmentation')
    save_technique_samples(SRC_DIR, OUT_DIR, n_samples=5)

    print(f'before: {n_before}')
    print(f'after: {n_after}')
    print(f"original (kept): {len(glob.glob(os.path.join(DST_DIR, 'orig_*_original.png')))}")
    for tech_name in TECHNIQUES:
        count = len(glob.glob(os.path.join(DST_DIR, f'orig_*_{tech_name}.png')))
        print(f'{tech_name}: {count}')
