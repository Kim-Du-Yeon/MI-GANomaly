import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_gradcam(img, heatmap, save_path, alpha=0.4):
    img_np = img.detach().cpu().numpy() if torch.is_tensor(img) else np.asarray(img)
    if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
        img_np = img_np.transpose(1, 2, 0)

    img_np = np.clip(img_np * 0.5 + 0.5, 0, 1)  # un-normalize from [-1, 1] to [0, 1]
    if img_np.shape[-1] == 1:
        img_np = np.repeat(img_np, 3, axis=-1)

    heatmap_color = plt.get_cmap('jet')(heatmap)[..., :3]
    overlay = np.clip((1 - alpha) * img_np + alpha * heatmap_color, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(img_np)
    axes[0].set_title('original')
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM++')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('overlay')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
