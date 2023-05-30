"""Data loader."""

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from data.build import build_dataset
from data.collate import COLLATE_FN
from utils import distributed as du


def construct_loader(cfg, split):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    collate_fn = COLLATE_FN[cfg.MODEL.TASK]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / du.get_world_size())
        drop_last = True
        shuffle = True
    elif split in ["val"]:
        dataset_name = cfg.VAL.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / du.get_world_size())
        drop_last = False
        shuffle = False
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / du.get_world_size())
        drop_last = False
        shuffle = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)
    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if du.get_world_size() > 1 else None
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    return loader


def shuffle_dataset(loader, cur_epoch):
    """"
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    assert isinstance(
        loader.sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(loader.sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)
