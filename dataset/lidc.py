import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from torchvision import transforms
import glob


class LIDCDataset(Dataset):
    def __init__(self, root_dir='../LIDC', augmentation=False):
        self.root_dir = root_dir
        self.file_names = glob.glob(os.path.join(
            root_dir, './**/*.npy'), recursive=True)
        self.augmentation = augmentation

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        path = self.file_names[index]
        img = np.load(path)

        if self.augmentation:
            random_n = torch.rand(1)
            if random_n[0] > 0.5:
                img = np.flip(img, 2)

        imageout = torch.from_numpy(img.copy()).float()
        imageout = imageout.unsqueeze(0)

        return {'data': imageout}

if __name__ == "__main__":
    import pylidc as pl

    print("Converting to npy...")

    # Query for all CT scans with desired traits.
    scans = pl.query(pl.Scan)
    print(scans.count())

    for i, scan in enumerate(scans):
        print(scan.slice_thickness,scan.pixel_spacing )
    #scans = pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= 1,
    #                             pl.Scan.pixel_spacing <= 0.6)
    print(dir(scan))