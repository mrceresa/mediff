from dataset import MRNetDataset, BRATSDataset, ADNIDataset, DUKEDataset, LIDCDataset, DEFAULTDataset
from torch.utils.data import WeightedRandomSampler
from omegaconf.dictconfig import DictConfig

def get_dataset(cfg:DictConfig):
    
    if not "severity" in cfg.dataset: 
        severity = "HGG"
    else:
        severity = cfg.dataset.severity
    if cfg.dataset.name == 'MRNet':
        train_dataset = MRNetDataset(
            root_dir=cfg.dataset.root_dir, task=cfg.dataset.task, plane=cfg.dataset.plane, split='train')
        val_dataset = MRNetDataset(root_dir=cfg.dataset.root_dir,
                                   task=cfg.dataset.task, plane=cfg.dataset.plane, split='valid')
        sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weight, num_samples=len(train_dataset.sample_weight))
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'BRATS':
        train_dataset = BRATSDataset(
            root_dir=cfg.dataset.root_dir, imgtype=cfg.dataset.imgtype, train=True, severity=severity)
        val_dataset = BRATSDataset(
            root_dir=cfg.dataset.root_dir, imgtype=cfg.dataset.imgtype, train=True, severity=severity)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'ADNI':
        train_dataset = ADNIDataset(
            root_dir=cfg.dataset.root_dir, augmentation=True)
        val_dataset = ADNIDataset(
            root_dir=cfg.dataset.root_dir, augmentation=True)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'DUKE':
        train_dataset = DUKEDataset(
            root_dir=cfg.dataset.root_dir)
        val_dataset = DUKEDataset(
            root_dir=cfg.dataset.root_dir)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'LIDC':
        train_dataset = LIDCDataset(
            root_dir=cfg.dataset.root_dir, augmentation=True)
        val_dataset = LIDCDataset(
            root_dir=cfg.dataset.root_dir, augmentation=True)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'DEFAULT':
        train_dataset = DEFAULTDataset(
            root_dir=cfg.dataset.root_dir)
        val_dataset = DEFAULTDataset(
            root_dir=cfg.dataset.root_dir)
        sampler = None
    raise ValueError(f'{cfg.dataset.name} Dataset is not available')
