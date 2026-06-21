import glob
import os

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms

IMG_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp')


class ELPVDataset(Dataset):
    def __init__(self, root_dir, split='train', label=0, isize=32, nc=3):
        super().__init__()
        self.label = label
        self.nc = nc

        class_name = 'normal' if label == 0 else 'anomaly'
        img_dir = os.path.join(root_dir, split, class_name)

        self.paths = []
        for ext in IMG_EXTENSIONS:
            self.paths.extend(glob.glob(os.path.join(img_dir, ext)))
        self.paths.sort()

        mean, std = [0.5] * nc, [0.5] * nc

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((isize, isize)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.GaussianBlur(kernel_size=3),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((isize, isize)),
                transforms.CenterCrop(isize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        mode = 'RGB' if self.nc == 3 else 'L'
        img = Image.open(self.paths[idx]).convert(mode)
        img = self.transform(img)
        return img, self.label


def get_dataloader(opt, split):
    persistent_workers = opt.workers > 0

    if split == 'train':
        dataset = ELPVDataset(opt.dataroot, split='train', label=0, isize=opt.isize, nc=opt.nc)
        labels = torch.zeros(len(dataset), dtype=torch.long)
        loader = DataLoader(dataset, batch_size=opt.batchsize, shuffle=True, num_workers=opt.workers,
                             persistent_workers=persistent_workers, pin_memory=True)

    elif split == 'test':
        normal_ds = ELPVDataset(opt.dataroot, split='test', label=0, isize=opt.isize, nc=opt.nc)
        anomaly_ds = ELPVDataset(opt.dataroot, split='test', label=1, isize=opt.isize, nc=opt.nc)
        dataset = ConcatDataset([normal_ds, anomaly_ds])
        labels = torch.cat([
            torch.zeros(len(normal_ds), dtype=torch.long),
            torch.ones(len(anomaly_ds), dtype=torch.long),
        ])
        loader = DataLoader(dataset, batch_size=opt.batchsize, shuffle=False, num_workers=opt.workers,
                             persistent_workers=persistent_workers, pin_memory=True)

    else:
        raise ValueError(f'unknown split: {split}')

    return loader, labels
