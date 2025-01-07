from pathlib import Path

import numpy as np
import torch
from faiss.contrib.datasets import sanitize
from faiss.contrib.vecs_io import bvecs_mmap, fvecs_mmap, ivecs_mmap
from numpy.lib.format import open_memmap

NUM_WORKERS = 10


####################################################################
# Data classes for memory-mapped vector datasets
####################################################################


class LoopSubset(torch.utils.data.Dataset):
    r"""
    Subset of a dataset.
    When training, it will loop over all the dataset in blocks of size n.
    When evaluating, it will only use the first n elements
    """

    def __init__(
        self,
        cfg,
        dataset: torch.utils.data.Dataset,
        limit: int,
        train: bool,
        shuffle_train=True,
    ) -> None:
        self.cfg = cfg  # Use cfg as a register for current epoch (hacky solution, but it works)
        self.dataset = dataset
        self.loop_len = len(dataset)
        self.train = train
        self.limit = limit
        self.shuffling = shuffle_train and train
        if train and shuffle_train:
            self.rand_map = np.arange(self.loop_len)
            self.rand_map = np.random.permutation(self.rand_map)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self[i] for i in idx]
        cur_idx = idx
        if self.train:
            cur_idx = (cur_idx + self.limit * self.cfg._cur_epoch) % self.loop_len
        if self.shuffling:
            cur_idx = self.rand_map[cur_idx]
        return self.dataset[cur_idx]

    def __len__(self):
        return self.limit


class MMapDataset(torch.utils.data.Dataset):
    N_CACHE = 2_000_000

    def __init__(self, data_mmap, cfg, block_shuffle=True):
        self.mmap = data_mmap
        self.i_start_cache = -1
        self.block_shuffle = block_shuffle

    def __len__(self):
        return len(self.mmap)

    def __getitem__(self, idx):
        if self.i_start_cache == -1 or not (
            self.i_start_cache <= idx < self.i_start_cache + self.N_CACHE
        ):
            self.i_start_cache = idx - idx % self.N_CACHE
            cached_data = self.mmap[
                self.i_start_cache : self.i_start_cache + self.N_CACHE
            ]
            cached_data = sanitize(cached_data).astype(np.float32)
            cached_data = torch.from_numpy(cached_data).to(torch.float32)
            self.cached_data = cached_data

            if self.block_shuffle:
                self.rand_map = np.arange(len(self.cached_data))
                self.rand_map = np.random.permutation(self.rand_map)

        if self.block_shuffle:
            idx = self.rand_map[idx - self.i_start_cache]
        else:
            idx = idx - self.i_start_cache
        vec = self.cached_data[idx]
        return vec


####################################################################
# Creating & exploting vector datasets
####################################################################


def get_data_memmap(filepath, dataname):
    p_file = Path(filepath)
    if not filepath or not p_file.exists() or not p_file.is_file():
        raise FileNotFoundError(
            f"File {filepath} for data source {dataname} doesn't exist!"
        )

    if p_file.suffix == ".bvecs":
        return bvecs_mmap(filepath)
    elif p_file.suffix == ".fvecs":
        return fvecs_mmap(filepath)
    elif p_file.suffix == ".ivecs":
        return ivecs_mmap(filepath)
    elif p_file.suffix == ".npy":
        return open_memmap(filepath)
    else:
        raise ValueError(
            f"Unsuported file format '{p_file.suffix}' for data source {dataname} stored at {filepath}. Supported formats: npy, bvecs, fvecs, ivecs."
        )


def data_loader_for_mmap(data_mmap, cfg, loop, train=True, compute_stats=False):
    acc_print = cfg._accelerator.print
    mmap_ds = MMapDataset(data_mmap, cfg)
    acc_print(f"Got memory-mapped dataset of size {len(mmap_ds)}")
    if loop and cfg.ds.loop and cfg.ds.loop < len(mmap_ds):
        mmap_ds = LoopSubset(
            cfg, mmap_ds, limit=cfg.ds.loop, train=train, shuffle_train=False
        )
        acc_print(f"Will loop over it in chuncks of size {len(mmap_ds)}")
    data_loader = torch.utils.data.DataLoader(
        mmap_ds, batch_size=cfg.batch, shuffle=False, num_workers=0
    )
    return data_loader


####################################################################
# Dataset loading
####################################################################


def load_vec_trainset(cfg):
    assert (
        cfg.trainset is not None
    ), "Please provide a training dataset using the 'trainset' parameter, or use a default database with the 'db' parameter"

    # Training data
    xt = get_data_memmap(cfg.trainset, "training")
    cfg._accelerator.print(f"Max size of the training + val dataset: {len(xt)}")

    assert (
        cfg.ds.valset <= len(xt) // 2
    ), "Validation set larger than half of the training set, abnormal configuration"
    xt, xval = xt[: -cfg.ds.valset], xt[-cfg.ds.valset :]
    cfg.ds.trainset = min(cfg.ds.trainset or len(xt), len(xt))
    xt = xt[: cfg.ds.trainset]
    cfg._accelerator.print(
        f"Split into {len(xt)} training vectors and {len(xval)} validation vectors"
    )

    train_dataloader = data_loader_for_mmap(
        xt, cfg, loop=not cfg.qinco1_mode, train=True, compute_stats=True
    )
    val_dataloader = data_loader_for_mmap(xval, cfg, loop=False, train=False)

    return (xt, xval), (train_dataloader, val_dataloader)


def load_vec_db(cfg):
    assert (
        cfg.db is not None
    ), "Please provide a path to database or the name of a default dataset using the 'db' parameter"
    # Database & validation data
    xdb = get_data_memmap(cfg.db, "database")
    cfg._accelerator.print(f"Max size of the database: {len(xdb)}")

    cfg.ds.db = min(cfg.ds.db or len(xdb), len(xdb))
    if cfg.ds.db < len(xdb):
        cfg._accelerator.print(f"Will use only {cfg.ds.db} elements from the database")
    xdb = xdb[: cfg.ds.db]
    db_dataloader = data_loader_for_mmap(xdb, cfg, loop=False, train=False)

    return xdb, db_dataloader


def load_queries_data(cfg):
    # Queries
    assert (
        cfg.queries is not None
    ), "Please provide a set of queries using the 'queries' parameter, or use a default database with the 'db' parameter"
    xq = get_data_memmap(cfg.queries, "queries")
    cfg._accelerator.print(f"Size of the queries dataset: {len(xq)}")

    # Queries ground-truth
    assert (
        cfg.queries_gt is not None
    ), "Please provide ground-truth answer to queries using the 'queries_gt' parameter, or use a default database with the 'db' parameter"
    xq_gt = get_data_memmap(cfg.queries_gt, "queries_gt")
    cfg._accelerator.print(f"Size of the queries ground-truth dataset: {len(xq_gt)}")

    return xq, xq_gt
