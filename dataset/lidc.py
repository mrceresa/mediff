import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from torchvision import transforms
from glob import glob
from tqdm.auto import tqdm
import numpy as np

class LIDCDataset(Dataset):
    def __init__(self, root_dir='../LIDC', augmentation=False):
        self.root_dir = root_dir
        self.file_names = glob(os.path.join(
            root_dir, './npys/**/*.npy'), recursive=True)
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
    from pylidc.Scan import _get_dicom_file_path_from_config_file

    lidc_path = _get_dicom_file_path_from_config_file()
    npy_path = os.path.join(lidc_path,"npys")
    os.makedirs(npy_path, exist_ok=True)
    # Query for all CT scans with desired traits.
    scans = pl.query(pl.Scan)
    
    print("Converting to npy...")

    for i, scan in tqdm(enumerate(scans)):
        scan_path = os.path.join(npy_path,str(scan.id)) + ".npy"
        if not os.path.exists(scan_path):
            vol = scan.to_volume()
            np.save(scan_path, vol)

    ds = LIDCDataset(root_dir=lidc_path)
    print(ds.file_names)


        #print(scan.id, scan.slice_thickness,scan.pixel_spacing, len(glob( "%s/*"%scan.get_path_to_dicom_files() )) )
    #scans = pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= 1,
    #                             pl.Scan.pixel_spacing <= 0.6)
    