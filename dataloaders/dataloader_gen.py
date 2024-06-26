from torch.utils.data import DataLoader
from dataloaders import IQA

def gen_dataloader(cfg):

    train_dataset = IQA.Train(cfg=cfg)
    test_ref_dataset = IQA.Ref(cfg=cfg)
    test_dataset = IQA.Test(cfg=cfg)

    test_ref_loader = DataLoader(test_ref_dataset, batch_size=1, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)

    return train_dataset, test_ref_loader, test_loader