# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed

from qinco.datasets import load_vec_db, load_vec_trainset

from .log import TestMetricLogger, get_metric_logger
from .metrics import Timer
from .model import IVFBook, QINCo, QINCoInferenceWrapper, initialize_qinco_codebooks
from .model.scheduler import RampCosineLRSchedule
from .utils import (
    QAccelerator,
    SharedCfgState,
    count_trainable_parameters,
    ensure_path,
    load_model,
    load_saved_model_data,
    log_mem_info,
    save_model,
)
from .vrq import train_rq_centroids

####################################################################
# Training setup
####################################################################


def build_optimizer(cfg, model):
    if cfg.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    elif cfg.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.wd
        )
    else:
        raise ValueError(f"Unkown {cfg.optimizer=}")

    if cfg._optimizer is not None:  # state dict for optimizer
        optimizer.load_state_dict(cfg._optimizer)
    cfg._optimizer = optimizer
    return optimizer


def build_scheduler(cfg, optimizer):
    lr = cfg.lr
    s_cfg = cfg.scheduler

    if s_cfg.name == "cosine":
        scheduler = RampCosineLRSchedule(
            optimizer, s_cfg.ramp_epochs, cfg.epochs, lr * s_cfg.lr_min_fact
        )
    elif s_cfg.name == "reduce_lr_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=s_cfg.reduce_fact,
            patience=s_cfg.patience,
            min_lr=lr * s_cfg.lr_min_fact,
            threshold=s_cfg.threshold_frac,
        )
    else:
        raise ValueError(f"Unkown {s_cfg.name=}")

    if cfg._scheduler is not None:  # state dict for scheduler
        scheduler.load_state_dict(cfg._scheduler)
    cfg._scheduler = scheduler
    return scheduler


####################################################################
# QINCo evaluation
####################################################################


@torch.no_grad
def compute_MSE(accelerator: QAccelerator, cfg, melog, model, val_dataset):
    """compute MSE on validation set"""

    model.eval()
    melog.start_eval(val_dataset)
    max_loader_size = accelerator.gather(
        torch.tensor([len(val_dataset)]).to(accelerator.device)
    ).max()
    max_loader_size = int(max_loader_size.detach().cpu())

    # If task is eval, do a few batches to warm-start evaluation (allows jit to do compilation, for more accurate timers)
    if "eval" in cfg.task:
        for i_batch, batch in enumerate(val_dataset):
            encoded_data = model(batch, step="encode")

            decoded = model(encoded_data, step="decode")
            if i_batch >= 10:
                break
        _ = decoded[-1][-1].item()  # Forces CUDA synchronisation
        accelerator.print(f"Warm-start with {i_batch} batches: done")

    # Evaluate on full dataloader
    t_encode, t_decode = Timer(), Timer()
    n_vecs = 0
    for i_batch, batch in enumerate(val_dataset):
        n_vecs += len(batch)
        with t_encode:
            encoded_data = model(batch, step="encode")
            if encoded_data is not None:
                _ = float(
                    encoded_data[-1].reshape(-1)[-1].cpu()
                )  # Forces to complete computation before leaving the timer

        with t_decode:
            xhat = model(encoded_data, step="decode")
            _ = float(
                xhat.reshape(-1)[-1].cpu()
            )  # Forces to complete computation before leaving the timer
        assert xhat.shape == batch.shape, f"{xhat.shape=} != {batch.shape=}"

        melog.step_eval(i_batch, batch, xhat, encoded_data)

    # Run fake calls to synchronize with processes with longer datasets
    for i_batch in range(len(val_dataset), max_loader_size):
        melog.step_eval(i_batch, None, None, None)

    # End eval
    melog.end_eval()
    if "eval_time" in cfg.task:
        accelerator.print(
            f"Encoding time: {t_encode.s()} | Decoding time: {t_decode.s()}"
        )
        accelerator.print(
            f"Encoding time / vector: {t_encode.get()/n_vecs*1000*1000:.1f}μs"
        )
        accelerator.print(
            f"Decoding time / vector: {t_decode.get()/n_vecs*1000*1000:.1f}μs"
        )
    model.train()

    return melog.metrics.last_m_vals["MSE"]


####################################################################
# QINCo training
####################################################################


def step_scheduler(
    cfg, scheduler, epoch, i_batch=None, train_set_length=None, MSE_val=None
):
    if i_batch is not None:  # Step during epoch loop
        assert train_set_length is not None
        if cfg.scheduler.name == "cosine":
            scheduler.step(epoch + (i_batch + 1) / train_set_length)
    else:  # Step during main loop
        if cfg.scheduler.name == "cosine":
            scheduler.step(epoch)
        elif cfg.scheduler.name == "reduce_lr_plateau":
            assert MSE_val is not None
            scheduler.step(MSE_val)


def aggregate_losses(cfg, losses):
    if isinstance(losses, dict):
        losses = torch.sum(
            torch.stack([loss_val for loss_name, loss_val in losses.items()])
        )
    return losses


def train_one_epoch_qinco(
    cfg, accelerator, model, train_set, optimizer, scheduler, melog
):
    """run one epoch of training"""
    melog.start_epoch(train_set, scheduler.get_last_lr()[0])
    model.train()

    for i_batch, batch in enumerate(train_set):
        with accelerator.accumulate(model):
            step_scheduler(
                cfg, scheduler, cfg._cur_epoch, i_batch, train_set_length=len(train_set)
            )

            encoded_data, quantized_batch, losses = model(batch, step="train")
            total_loss = aggregate_losses(cfg, losses)

            accelerator.backward(total_loss)
            if cfg.grad_clip and accelerator.sync_gradients:
                accelerator.clip_grad_value_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            melog.step_epoch_batch(
                i_batch,
                batch,
                encoded_data,
                total_loss,
                losses,
                scheduler.get_last_lr()[0],
            )
            accelerator.wait_for_everyone()

    if cfg.verbose:
        accelerator.print_nolog()
    melog.end_training_part_epoch()


def train_qinco(accelerator, cfg, train_dataset, val_dataset, model):
    set_seed(cfg.seed)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    melog = get_metric_logger(cfg)
    if cfg.task == "train":
        accelerator.print(f"optimizer {optimizer} and scheduler {scheduler}")

    # Prepare for training
    val_dataset = accelerator.prepare_test_data(val_dataset, cfg)
    model, optimizer, train_dataset, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataset,
        scheduler,
    )

    MSE_val = compute_MSE(accelerator, cfg, melog, model, val_dataset)
    melog.end_standalone_eval()
    if "eval" in cfg.task:
        return

    # Start directly at cfg._cur_epoch (even if > 0)
    while not melog.should_stop():
        step_scheduler(cfg, scheduler, cfg._cur_epoch, MSE_val=MSE_val)
        train_one_epoch_qinco(
            cfg, accelerator, model, train_dataset, optimizer, scheduler, melog
        )
        MSE_val = compute_MSE(accelerator, cfg, melog, model, val_dataset)
        melog.end_epoch(model, MSE_val)

        cfg._cur_epoch += 1
    melog.mark_end_training()


####################################################################
# Running the task
####################################################################


def setup_job_env(cfg):
    torch.set_default_dtype(torch.float32)
    for var_name, var_value in cfg.env.items():
        os.environ[var_name] = str(var_value)
    set_seed(cfg.seed)


def log_job_details(cfg, accelerator):
    accelerator.print(f"Running on: {os.environ.get('SLURM_JOB_NODELIST', 'local')}")
    accelerator.print("Running args:", sys.argv)
    log_mem_info(accelerator)


def initialize_model(cfg, train_dataset=None, val_dataset=None):
    if train_dataset is not None:
        cfg._D = next(iter(train_dataset)).shape[-1]
    rq_centroids, ivf_centroids = None, None

    cfg._qinco_jit = cfg.task in ["train"]

    if cfg.ivf_in_use:  # Load IVF centroids
        if cfg.ivf_centroids:
            ivf_centroids = cfg._ivf_centroids_preloaded
        elif not cfg.model:
            raise Exception(
                "When training a new model, please specify a path to IVF centroids"
            )

        cfg._ivf_book = IVFBook(cfg, ivf_centroids)

    if (
        not cfg.model and cfg.task == "train"
    ):  # Train RQ only to initialize new model, when not loading weights
        assert train_dataset is not None and val_dataset is not None
        rq_centroids = train_rq_centroids(cfg, train_dataset, val_dataset)

    if (
        ivf_centroids is not None and cfg.task != "convert"
    ):  # Loaded IVF, need to adapt to data scaling AFTER RQ training
        ivf_centroids = torch.tensor(
            (ivf_centroids - cfg._data_mean) / cfg._data_std,
            device=cfg._accelerator.device,
        )
        cfg._ivf_book.ivf_centroids.weight.copy_(ivf_centroids)

    model = QINCo(cfg)
    if rq_centroids is not None:
        initialize_qinco_codebooks(cfg, model, rq_centroids)

    load_model(cfg, model)

    if cfg.inference and cfg.task not in ["train", "convert"]:
        model = QINCoInferenceWrapper(cfg, model)

    model = model.to(cfg._accelerator.device)
    return model


####################################################################
# Define tasks
####################################################################


class BaseTask:
    USE_QINCO_MODEL = True

    def __init__(self, cfg):
        self.cfg = SharedCfgState(cfg)

        self.setup()
        self.load_data()
        model = self.load_model()

        if model is not None:
            if hasattr(model, "built") and not model.built:
                model.build()
            self.accelerator.print(f"Model:\n{model}")

    def setup(self):
        setup_job_env(self.cfg)

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = QAccelerator(
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=self.cfg.grad_accumulate,
            step_scheduler_with_optimizer=False,
            mixed_precision="fp16",
            cpu=self.cfg.cpu,
        )
        log_job_details(self.cfg, self.accelerator)
        self.init_config()
        self.accelerator.print(f"Configuration:\n{self.cfg._cfg}")

    def init_config(self):
        # Initialize registers
        self.cfg._accelerator = self.accelerator

        # Load data from model
        load_saved_model_data(self.cfg, load_qinco=self.USE_QINCO_MODEL)

        if self.cfg.ivf_centroids:
            self.cfg.ivf_in_use = True

        # Populate paths for standard databases
        if self.cfg.db in self.cfg.default_datasets:
            ds_preset = self.cfg.default_datasets[self.cfg.db]
            for key, val in ds_preset.items():
                if key == "limit_db":
                    self.cfg.ds.db = min(self.cfg.ds.db or val, val)
                elif key == "mse_scale":
                    self.cfg.mse_scale = val
                else:
                    assert key in ["db", "trainset", "queries", "queries_gt"]
                    self.cfg[key] = val
        del self.cfg._cfg.default_datasets  # Remove from logs

        # Initialize registers
        self.cfg._ivf_book = None
        self.cfg._cur_epoch = None

        # Include IVF in codebooks
        if self.cfg.M is not None:
            self.cfg._M_ivf = int(self.cfg.M) + 0
            self.cfg._K_vals = [self.cfg.K for _ in range(self.cfg.M)]
            if self.cfg.ivf_in_use:  # If uses IVF, insert a new codebook first
                self.cfg._M_ivf = self.cfg.M + 1
                self.cfg._K_vals.insert(0, self.cfg.ivf_K)

        if self.cfg.resume:
            assert (
                self.cfg.output
            ), "Please specify an output file to be able to resume from it"
        if self.cfg.output:
            ensure_path(self.cfg.output, parent=True)

    def load_model(self):
        if self.USE_QINCO_MODEL:
            self.qinco_model = initialize_model(self.cfg)
            return self.qinco_model

    def run(self):
        raise NotImplementedError


class QincoTrainTask(BaseTask):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.accelerator.print(
            f"nb trainable parameters {count_trainable_parameters(self.qinco_model)}"
        )

    def load_data(self):
        self.cfg._accelerator.print(f"Loading training data from {self.cfg.trainset}")
        (self.train_vecs, self.val_vecs), (self.train_dataset, self.val_dataset) = (
            load_vec_trainset(self.cfg)
        )

        self.cfg._accelerator.print(f"Training set: {self.train_vecs.shape}")
        self.cfg._accelerator.print(f"Validation set: {self.val_vecs.shape}")

        if self.cfg.qinco1_mode:
            self.cfg._accelerator.print(f"QINCo-1 mode, computing scale of DB...")
            d_min, d_max = float(self.train_vecs.min()), float(self.train_vecs.max())
            self.cfg._data_mean = np.ones_like(self.train_vecs[0]) * d_min + 1
            self.cfg._data_std = (d_max - d_min) / 2
            self.cfg._accelerator.print(
                f"Data scale is {self.cfg._data_std:.2g} (interval [{d_min:.2g};{d_max:.2g}]), rescale to interval [-1;1]"
            )
        else:
            self.cfg._accelerator.print(
                f"Computing mean and variance on training data..."
            )
            stats_vecs = self.train_vecs[:100_000]  # We don't need very exact stats
            self.cfg._data_mean = stats_vecs.mean(0)
            self.cfg._data_std = stats_vecs.std()
            self.cfg._accelerator.print(
                f"Mean of {self.cfg._data_mean.mean().item():.2g} and variance of {self.cfg._data_std.item():.2g}"
            )

    def load_model(self):
        self.qinco_model = initialize_model(
            self.cfg, self.train_dataset, self.val_dataset
        )
        return self.qinco_model

    def init_config(self):
        if self.cfg.resume:
            if Path(self.cfg.output).exists():
                self.cfg.model = self.cfg.output
            else:
                self.accelerator.print(
                    f"Can't resume from {self.cfg.output}, model file not found. Will start new training."
                )

        super().init_config()
        cfg = self.cfg

        # Initialize registers
        cfg._cur_epoch = 0
        cfg._melog = None  # Logger
        cfg._optimizer = None
        cfg._scheduler = None
        cfg._rq_mse = (
            None  # Used only if RQ centroids are trained, to write MSE to tensorboard
        )

        if cfg.task == "train":
            assert (
                cfg.output is not None
            ), "Please specify the outpout path to store the model weights using the 'output' argument"
            assert cfg.output.endswith(
                ".pt"
            ), "Please specify a .pt file for the model 'output' argument"
            if self.cfg.model is None:
                for model_arg in ["L", "dh", "M", "K", "A", "B"]:
                    assert (
                        self.cfg[model_arg] is not None
                    ), f"Please specify argument '{model_arg}' to train a model, or use the 'model_args' argument to use a pre-defined config"

    def run(self):
        train_qinco(
            self.accelerator,
            self.cfg,
            self.train_dataset,
            self.val_dataset,
            self.qinco_model,
        )


class QincoEvalTask(BaseTask):
    def setup(self):
        super().setup()

        if self.cfg.task == "eval_time":
            assert self.cfg.cpu, "Evaluation time should be run on CPU"
            torch.set_num_threads(32)

    def load_data(self):
        self.cfg._accelerator.print(f"Loading database from {self.cfg.db}")
        self.test_vecs, self.test_dataset = load_vec_db(self.cfg)
        self.cfg._accelerator.print(f"Test set: {self.test_vecs.shape}")

    def run(self):
        model = self.qinco_model
        melog = TestMetricLogger(self.cfg)
        test_dataset = self.accelerator.prepare_test_data(self.test_dataset, self.cfg)
        model = self.accelerator.prepare(model)

        compute_MSE(self.accelerator, self.cfg, melog, model, test_dataset)


class QincoConvertTask(BaseTask):
    DB_DIMS = {
        "bigann1M": 128,
        "deep1M": 96,
        "contriever1M": 768,
        "FB_ssnpp1M": 256,
    }
    # fmt: off
    DB_NORMS = {
        'bigann1M': [25.9134, 20.4491, 18.2318, 19.1883, 24.3351, 14.1938, 11.5445, 14.7300, 62.6836, 32.6022, 18.9913, 17.5646, 22.3281, 14.8167, 14.9237, 28.6546, 63.9660, 26.4173, 14.4791, 14.7710, 22.0409, 17.8848, 20.8229, 37.1614, 27.2970, 14.9248, 11.7714, 14.3163, 24.4522, 20.1203, 20.2677, 22.8355, 35.3779, 21.3080, 19.4423, 23.7519, 32.0448, 19.8099, 13.8373, 17.5912, 92.6548, 34.6038, 20.5830, 24.4384, 31.0192, 19.4205, 16.1397, 38.8999, 93.0131, 35.5026, 15.0127, 19.1778, 31.8407, 26.0465, 22.7495, 39.8944, 36.2062, 17.6875, 14.1436, 20.7015, 33.8140, 25.9258, 21.5489, 23.5974, 35.4830, 17.4590, 13.8317, 19.9067, 32.0118, 23.6363, 19.4878, 21.5059, 92.7804, 38.1845, 16.1485, 19.5100, 30.9866, 24.3355, 20.6082, 35.2542, 92.9100, 39.1926, 22.7319, 26.1329, 31.8714, 19.1142, 15.0054, 36.2066, 36.1230, 23.3903, 21.5036, 26.0033, 33.8631, 20.6559, 14.1621, 17.8198, 26.1366, 14.6955, 11.5557, 14.2477, 24.2761, 19.0726, 18.2171, 20.6076, 62.9775, 28.4197, 14.9306, 14.8592, 22.2685, 17.4710, 18.9724, 32.9799, 63.7405, 36.7992, 20.8523, 17.9852, 22.0976, 14.7318, 14.4730, 26.6433, 27.1191, 22.6914, 20.2779, 20.2332, 24.5079, 14.2745, 11.7760, 14.9576],
        'deep1M': [0.0629, -0.0321, -0.0542,  0.0635, -0.0246, -0.0008,  0.0249,  0.0189, 0.0075,  0.0024,  0.0147, -0.0039,  0.0478, -0.0378, -0.0279, -0.0181, 0.0225, -0.0026, -0.0050,  0.0193,  0.0084,  0.0221, -0.0071, -0.0185, 0.0385, -0.0080, -0.0486,  0.0269, -0.0103,  0.0435, -0.0038,  0.0159,-0.0388,  0.0105, -0.0052,  0.0468, -0.0370, -0.0163, -0.0405,  0.0058, 0.0520,  0.0555, -0.0606, -0.0433, -0.0106, -0.0038, -0.0151,  0.0098, 0.0071, -0.0308,  0.0195, -0.0188, -0.0347, -0.0205,  0.0147, -0.0081,-0.0025, -0.0428,  0.0215,  0.0176,  0.0052,  0.0161, -0.0264, -0.0202,-0.0008, -0.0166, -0.0121,  0.0103,  0.0119, -0.0560, -0.0285,  0.0178, 0.0028, -0.0428, -0.0015,  0.0187,  0.0011,  0.0257,  0.0087, -0.0198,-0.0072, -0.0207,  0.0272, -0.0094, -0.0184, -0.0281,  0.0207, -0.0186, 0.0040, -0.0134,  0.0496, -0.0184,  0.0147, -0.0340,  0.0021, -0.0016],
        'contriever1M': [-1.8548e-02, -4.4715e-03, -7.8606e-03, -1.6525e-02, -9.1010e-03, -1.1489e-02, -1.5377e-02, -2.7966e-02, -6.8126e-03, -1.3544e-02, -9.0354e-03,  3.1596e-03, -8.8681e-03, -2.1225e-02,  1.6184e-03, -3.4056e-02, -1.9883e-02, -6.2641e-03, -3.1986e-02, -1.6476e-02, -2.7378e-02, -1.1270e-02, -1.1680e-02, -2.9382e-02, -3.1108e-03, -2.7151e-02, -3.2559e-02, -9.0112e-03, -6.7624e-03, -2.2737e-02, -5.6188e-03, -1.3726e-02, -1.9977e-03, -1.7358e-02, -1.3866e-02, -1.8712e-02,  5.4525e-03, -2.0538e-02, -2.0163e-02, -1.2053e-02, -1.3322e-02, -8.2696e-03, -1.3555e-02, -3.4996e-02, -1.1764e-02, -1.7973e-02, -5.1385e-03, -1.9064e-02, -6.4903e-03,  1.8153e-03, -1.4812e-02, -9.5447e-03, -2.6954e-02, -1.5352e-02, -2.3357e-02, -2.1286e-02, -6.8270e-03, -2.7587e-02, -1.7704e-02, -1.5381e-02,  1.0892e-03,  6.9485e-04, -4.3766e-03, -2.5220e-02, -2.7870e-02, -1.7584e-02, -5.4447e-03, -3.4343e-02, -3.6299e-02, -1.8577e-02, -5.8307e-03, -1.6229e-02, -1.2690e-02, -1.8496e-02, -3.8690e-02, -1.3893e-02, -1.1176e-02, -5.1511e-03, -2.4513e-02, -1.8583e-02, -1.9152e-02, -2.9355e-03, -9.4021e-03, -4.6249e-03, -3.5601e-02, -2.2865e-02, -1.6857e-02, -1.1001e-03, -3.0527e-02, -1.9492e-02, -2.7742e-02, -1.8934e-02, -2.6228e-02, -1.4911e-02, -6.2043e-03, -2.0032e-02, -1.8071e-02, -1.8300e-02, -7.1160e-03, -2.2185e-02, -8.4225e-03, -1.0093e-02, -1.8947e-02, -1.7678e-02, -1.5925e-02, -2.2376e-02, -1.2198e-02,  4.7891e-03, -2.0816e-02, -2.8371e-02, -9.1960e-03, -1.2432e-02, -1.3390e-02, -2.2071e-02, -2.5889e-02, -7.0790e-03, -2.9367e-02, -1.2815e-02, -1.9602e-02,  1.0708e-02, -1.8453e-02, -2.4680e-02, -1.5410e-02,  2.0464e-03, -9.3015e-03, -2.0955e-02, -7.7865e-04, -1.2457e-02, -1.7925e-02, -1.9584e-02, -2.7156e-02, -2.4502e-02, -1.2979e-02, -2.7983e-02, -1.4121e-02, -6.1309e-03, -2.3556e-02, -1.4650e-02,  2.2621e-03, -1.1724e-02, -2.1905e-02, -3.8509e-03, -3.3054e-02, -8.6065e-03,  9.2797e-03, -3.9523e-03, -2.2889e-02, -1.4623e-02, -1.1786e-02, -2.5729e-03, -8.7308e-03, -1.9084e-02, -2.3125e-02, -1.3843e-02, -1.6081e-02, -2.9191e-02, -4.2723e-03, -1.1394e-03, -1.0030e-02, -7.3606e-03, -9.7051e-03, -1.1406e-02, -9.8063e-03, -8.5277e-03, -9.9575e-03, -1.2427e-02, -1.5683e-02, -2.8400e-02, -3.0735e-02, -2.5616e-02, -4.5112e-03, -1.7227e-02, -1.1085e-02, -2.0783e-02, -1.1742e-02, -1.3780e-03, -1.8778e-02, -2.4668e-02, -3.9114e-02, -1.3846e-02, -2.9446e-02, -8.4423e-03, -6.5679e-03, -1.9699e-02, -2.8233e-02, -8.4278e-03, -1.9949e-02, -1.2896e-02, -2.4336e-02, -9.5129e-03, -3.2688e-02,  1.6186e-03, -1.3618e-02, -1.0673e-02, -4.2787e-03, -2.1971e-02, -1.9237e-02, -5.7116e-03,  3.5081e-03, -4.1887e-02, -2.1021e-02, -1.5475e-02, -4.9672e-02, -2.0723e-03, -1.5686e-02, -1.8389e-02, -4.3606e-03, -2.8081e-02, -4.4218e-04, -1.5766e-02, -3.6392e-02, -1.5486e-02, -1.2507e-02, -2.0544e-03, -2.8700e-02, -2.1497e-02, -1.8892e-02, -1.0322e-02, -2.3992e-02, -1.0860e-02, -2.5386e-02, -1.6759e-02, -2.9426e-02, -1.4039e-02, -2.3384e-02, -1.5204e-02, -3.0683e-02, -6.8355e-03, -1.9878e-02, -1.7957e-02, -8.0987e-04, -5.0863e-03, -3.2916e-02, -2.1221e-02, -1.2345e-02,  9.4861e-03, -2.9101e-02, -1.4679e-02, -2.4798e-02, -1.8848e-02, -1.8433e-02, -1.7777e-02,  3.3134e-03, -1.5019e-02, -2.8465e-02, -6.5880e-03, -1.1754e-02, -8.9656e-04, -7.6170e-03, -2.3605e-02, -1.5190e-02, -1.3086e-02, -1.4293e-02, -2.0440e-02, -7.9569e-03, -1.9284e-02, -2.8436e-02, -2.2724e-02, -1.7096e-02, -9.6211e-03, -1.4095e-02,  1.5503e-03, -2.3499e-02, -2.1775e-02, -2.8000e-02, -4.5225e-03, -1.7281e-02, -2.8606e-02, -1.2046e-02, -3.2387e-02, -1.0526e-02, -1.7504e-02, -4.2499e-02, -2.4330e-01, -5.6533e-03, -3.5263e-02, -2.3841e-02, -2.6745e-02, -8.7770e-03, -2.4825e-02, -4.8565e-03, -5.0796e-03, -2.0604e-02, -7.8113e-03, -2.5137e-02, -1.9380e-02, -1.1721e-02, -2.4810e-02, -1.5535e-02, -2.7130e-02, -3.8451e-02, -1.5281e-02, -4.7508e-02,  5.6043e-03, -8.2462e-03, -1.7872e-02, -2.8102e-02, -8.6999e-03, -1.5247e-02, -1.5474e-02, -8.9804e-03, -3.6820e-02, -1.8548e-02, -1.0736e-02, -6.0579e-03, -1.9886e-02, -3.4906e-03, -8.4715e-03, -2.9928e-01, -2.1248e-02, -1.3108e-02, -1.8685e-02, -2.8163e-02, -2.7830e-02, -2.0698e-02, -1.6717e-02, -2.0394e-02, -6.8325e-03, -1.0117e-02, -1.3004e-02, -2.1088e-02, -1.7799e-02, -1.3030e-02, -8.5572e-03, -4.8866e-03, -7.9420e-03, -3.1288e-02, -1.8668e-02, -1.5866e-02, -2.0334e-03, -2.4359e-02, -2.4846e-02, -4.3134e-03, -2.3327e-03, -3.3671e-02, -3.4869e-02, -1.1910e-02, -2.0893e-02, -1.7506e-02, -9.3531e-03, -3.4323e-02, -1.2852e-02, -1.5955e-02, -1.1018e-02, -2.1730e-02, -1.0676e-02, -2.8878e-02, -1.7237e-02, -2.4785e-02,  6.7450e-03, -2.3562e-02, -2.2610e-02, -8.7025e-03, -5.9448e-03, -9.9253e-03, -1.7427e-02, -7.6727e-03, -1.3280e-02, -2.3334e-02, -5.7605e-03, -8.9700e-03, -1.4958e-02, -1.3003e-02, -2.5988e-02, -2.3345e-02, -2.1821e-02, -1.0870e-02, -8.6794e-03, -1.5131e-02, -2.0231e-02, -1.4282e-02, -6.6496e-03, -1.0595e-02, -2.5246e-02, -1.3141e-02, -1.2439e-02, -2.9468e-02, -8.7229e-03, -3.0653e-03, -2.5891e-02, -3.8470e-03, -2.9773e-02, -1.7091e-02, -1.4888e-02, -2.9646e-02, -1.6602e-02, -2.0207e-02, -4.7583e-02, -1.7453e-02, -4.7490e-03, -1.2098e-02, -5.3847e-04, -9.4589e-03,  5.3800e-03, -2.9858e-03, -1.1665e-02, -2.9130e-02, -1.5671e-02, -1.2450e-02, -8.1488e-03, -6.6592e-03, -4.7748e-03, -3.3805e-02, -3.1344e-03, -1.7078e-02,  2.9861e-03, -1.2371e-02, -1.1020e-02, -2.5876e-02, -2.1373e-02,  2.7545e-03, -2.1289e-02, -1.7375e-02, -3.0430e-02, -1.3080e-02, -3.0080e-02, -2.6563e-02, -2.4571e-02, -5.2720e-03, -2.3147e-02, -2.9082e-03, -1.7319e-02, -2.2475e-02, -1.1492e-02, -1.8132e-02, -2.8066e-02, -1.7866e-02, -1.6913e-02, -8.1702e-03, -2.6820e-02, -2.3123e-02, -1.1312e-02, -1.3432e-02, -1.7461e-02,  2.7786e-03, -3.5312e-02,  5.5144e-04, -1.6285e-02, -2.3908e-02, -2.4735e-02, -4.0787e-02, -5.0867e-03, -1.6177e-02, -8.2743e-03, -1.8681e-02, -1.3591e-02, -9.9860e-03, -6.2414e-03, -1.7564e-02, -5.9951e-03, -1.8023e-02, -1.8730e-02, -1.9000e-02, -2.2692e-02, -2.2792e-02, -1.5190e-02, -2.8206e-02, -1.0730e-02, -3.4800e-02, -2.0857e-02, -2.3298e-02, -2.2722e-02,  1.3207e-02, -3.3908e-03, -1.1357e-02, -1.2299e-02,  1.1659e-02, -2.0710e-02, -2.9200e-02,  2.7910e-05, -1.4185e-02, -2.9017e-02, -3.3581e-02, -1.8913e-02, -1.4997e-02, -1.3836e-02, -1.4885e-02, -1.0344e-02, -5.5680e-03, -1.1980e-02, -1.9320e-02, -1.1832e-02, -1.2168e-02, -4.9098e-03, -1.7489e-02, -1.6503e-02,  7.1721e-03, -2.1056e-02, -1.8407e-02, -1.0469e-02, -1.9785e-02, -2.6040e-02, -3.7982e-03, -1.3121e-02, -2.8448e-02, -2.0854e-02, -2.2074e-02, -1.2803e-02, -2.9820e-02, -1.0516e-02,  1.0120e-02, -1.9526e-02, -7.2131e-03, -2.7107e-02, -1.0709e-02, -1.3902e-02, -1.5600e-02, -7.5618e-03,  1.0961e-03, -2.1646e-02, -5.5583e-03, -5.5526e-03, -1.4756e-02, -8.0193e-03, -2.2760e-02, -3.0927e-02, -3.1929e-02, -7.9370e-03, -2.4009e-02, -2.1976e-02, -2.1203e-02, -2.9963e-02, -2.4829e-02, -1.7249e-02, -2.7227e-03, -2.6803e-02,  9.4339e-03, -1.6830e-02, -1.6652e-02, -2.1760e-02, -1.3956e-02, -1.6363e-02, -1.3245e-02, -4.3209e-03, -3.5949e-03, -4.2876e-03, -2.4825e-02, -1.9855e-02, -2.0568e-02, -7.0549e-03, -6.7759e-03, -4.5818e-02, -1.3813e-02, -2.0235e-02, -1.2308e-02, -3.8523e-03, -1.8733e-02, -1.4320e-02, -1.1733e-02, -1.5409e-03, -9.3518e-03, -1.9058e-02,  2.5846e-03, -2.2582e-02, -4.7712e-03, -1.7096e-02, -3.5481e-02, -2.3679e-02,  3.8131e-03, -2.6641e-02,  5.7476e-03, -3.4512e-02, -1.5417e-02, -9.7302e-03, -1.1599e-02,  9.2179e-03, -1.0874e-02, -1.5953e-02, -1.8172e-02, -8.1316e-03, -9.8407e-03, -8.8272e-03, -1.6632e-02, -1.0500e-02, -5.8138e-03, -2.7272e-02, -2.7899e-02, -2.0753e-02, -3.2868e-02, -9.6628e-03, -2.3061e-02,  5.2491e-03, -1.1957e-02, -2.0170e-03, -3.9878e-03, -2.8784e-02, -2.9979e-02, -2.4256e-02, -9.3201e-03, -2.1436e-02, -1.7982e-02, -2.0254e-02, -1.8550e-02, -1.3860e-02, -1.9592e-02, -3.1215e-02, -1.2545e-02, -3.6349e-02, -3.0801e-02, -2.3674e-02, -2.2939e-02, -1.6419e-02, -1.2392e-02, -2.0331e-02, -2.2356e-02, -9.5789e-03, -1.1791e-02, -2.3086e-02, -2.8671e-02, -2.0095e-02, -8.1478e-03, -6.0773e-04, -1.5266e-02, -1.6088e-02, -3.5746e-03, -1.7528e-02, -2.9595e-02, -1.7254e-02, -1.4546e-02, -3.3351e-02, -2.4503e-02, -7.3448e-03, -1.1574e-02, -1.8634e-02, -7.7941e-03, -2.0338e-02, -2.9447e-02, -1.8150e-02, -2.2716e-03, -1.8706e-02, -1.8350e-02, -9.9822e-03, -3.3087e-02,  1.7424e-02, -1.1577e-02,  2.4128e-03, -4.3282e-02, -2.5394e-02, -9.6857e-03, -1.7365e-02,  8.8108e-03, -1.4095e-02, -4.6571e-03, -1.3951e-02, -1.2105e-02, -1.2397e-02, -9.3097e-03, -2.7094e-02, -4.1489e-02, -1.7666e-02, -5.5750e-03, -1.2852e-02, -8.9969e-03, -1.3697e-02, -2.4014e-02, -2.9803e-02, -2.4198e-02, -3.7126e-02, -1.9556e-02,  2.0755e-03, -1.6222e-02, -1.0047e-02, -2.0980e-02, -2.3272e-02, -1.8366e-02, -1.1327e-02, -1.1471e-02, -3.5168e-02, -1.1118e-02, -3.1542e-02,  3.1493e-03, -2.0688e-02, -2.3538e-02, -2.7907e-02, -1.6831e-02, -2.5707e-02, -1.4724e-02, -5.1896e-03, -2.0858e-02, -1.7916e-02, -1.0001e-02, -1.9988e-02, -1.0746e-02, -1.5745e-02, -1.4327e-02, -1.2784e-02, -1.2299e-02, -8.8605e-03,  6.6481e-03, -5.1756e-03, -1.4676e-02, -2.8578e-02, -3.1914e-02, -1.7125e-02, -2.4576e-02, -9.7271e-03, -4.1204e-03, -8.8177e-03, -2.1193e-02,  2.1984e-04, -3.0370e-02, -1.8071e-02, -1.0207e-02, -9.0201e-03, -1.2315e-02, -2.8572e-02, -1.6898e-02, -1.3597e-02, -2.0747e-02, -8.7040e-03, -3.0936e-02, -4.3729e-03, -5.4079e-03, -1.7245e-02, -4.2024e-03, -9.4405e-03, -1.0699e-02, -1.3056e-02, -1.7772e-02, -7.8886e-03, -7.7061e-03, -2.4422e-02, -2.6609e-02, -1.3641e-02, -2.3620e-02,  7.3478e-03,  4.3310e-03, -1.5758e-02, -3.7182e-03, -2.7205e-02, -8.6334e-03, -2.1506e-02, -9.7153e-03, -3.2696e-02, -9.9869e-04,  1.3422e-04, -4.3911e-03, -2.2354e-02, -8.6295e-03, -1.3117e-02, -1.8357e-02, -1.8502e-02,  6.1643e-03, -3.2450e-03, -3.3989e-02, -3.2837e-04, -1.9665e-02, -2.5843e-02, -1.3782e-02, -1.8865e-02, -2.2984e-02, -8.3617e-03,  2.4935e-01, -3.6928e-02, -1.4777e-02, -8.8277e-03, -1.2544e-02, -2.8774e-02,  1.1355e-03, -7.5590e-03, -2.6839e-02, -1.2760e-02, -3.3658e-02, -1.6985e-02, -2.0401e-02, -2.9326e-02, -2.1376e-02, -1.9648e-02],
        'FB_ssnpp1M': [127.9273, 127.9845, 128.0516, 127.9894, 127.9856, 127.9844, 127.9943, 127.9666, 128.0200, 127.9898, 128.0356, 127.9765, 127.9828, 128.0022, 127.9793, 127.9546, 127.9892, 128.0219, 127.9733, 128.0594, 127.9662, 127.9892, 128.0480, 127.9967, 128.0497, 128.0621, 127.9378, 127.9864, 127.9635, 127.9769, 128.0147, 127.9669, 128.0281, 127.9933, 127.9733, 127.9686, 128.0474, 128.0172, 128.0013, 127.9785, 127.9826, 127.9958, 128.0575, 128.0544, 127.9791, 127.9501, 127.9601, 127.9253, 128.0488, 128.0011, 128.0588, 128.0603, 127.9746, 128.0101, 127.9848, 128.0094, 128.0081, 127.9462, 127.9921, 127.9838, 127.9571, 128.0340, 127.9697, 127.9841, 127.9774, 127.9688, 128.0047, 127.9845, 128.0294, 127.9374, 128.0361, 127.9591, 127.9779, 128.0390, 128.0506, 127.9939, 127.9625, 128.0061, 128.0053, 128.0549, 127.9962, 128.0215, 128.0442, 128.0019, 128.0206, 128.0255, 127.9872, 128.0012, 127.9726, 128.0734, 127.9641, 127.9775, 128.0221, 127.9842, 127.9833, 127.9932, 128.0200, 128.0409, 128.0506, 128.0390, 127.9815, 127.9954, 128.0109, 127.9800, 128.0320, 128.0250, 127.9632, 127.9471, 127.9908, 127.9794, 127.9578, 128.0424, 127.9836, 128.0562, 128.0518, 128.0048, 128.0095, 128.0492, 128.0284, 127.9923, 127.9908, 128.0341, 127.9106, 128.0006, 128.0252, 128.0214, 127.9853, 128.0015, 128.0243, 128.0030, 127.9821, 127.9161, 128.0291, 128.0348, 127.9127, 127.9565, 127.9630, 127.9830, 127.9489, 127.9962, 127.9924, 127.9931, 128.0056, 127.9836, 127.9479, 127.9548, 127.9848, 127.9708, 127.9664, 128.0298, 128.0116, 127.9473, 128.0253, 127.9879, 127.9927, 128.0161, 127.9897, 128.0234, 127.9604, 127.9951, 128.0270, 127.9561, 128.0589, 128.0484, 128.0571, 128.0592, 127.9738, 127.9959, 127.9235, 128.0139, 127.9227, 128.0401, 128.0334, 127.9974, 128.0541, 128.0458, 127.9993, 128.0209, 128.0271, 127.9810, 128.0474, 127.9934, 128.0543, 127.9691, 128.0154, 128.0138, 128.0013, 127.9979, 127.9838, 128.0013, 127.9420, 127.9798, 127.9159, 127.9855, 127.9950, 127.9962, 128.0030, 128.0095, 128.0014, 128.0127, 127.9872, 127.9870, 128.0149, 128.0181, 127.9968, 128.0108, 128.0235, 127.9864, 127.9967, 127.9724, 127.9985, 127.9617, 127.9897, 127.9927, 128.0550, 128.0285, 127.9752, 128.0162, 127.9575, 127.9871, 127.9538, 128.0050, 128.0635, 128.0474, 128.0386, 127.9850, 127.9721, 127.9928, 127.9627, 127.9881, 127.9886, 127.9838, 128.0328, 127.9703, 127.9809, 128.0382, 128.0553, 127.9934, 127.9852, 127.9538, 127.9644, 128.0291, 128.0133, 127.9576, 127.9918, 128.0235, 127.9947, 128.0025, 127.9720, 127.9305, 128.0552, 127.9812, 128.0349, 128.0409, 127.9706, 127.9993],
    }
    DB_STD = {
        'bigann1M': 36.5888,
        'deep1M': 0.1020,
        'contriever1M': 0.0583,
        'FB_ssnpp1M': 22.1006,
    }
    # fmt: on

    def init_config(self):
        self.db_name = self.cfg.db.replace("1B", "1M")

        assert self.db_name in self.DB_DIMS, f"Can' find DB {self.db_name}"
        assert self.cfg.model is not None
        assert self.cfg.output is not None

        super().init_config()

    def load_data(self):
        self.cfg._D = self.DB_DIMS[self.db_name]

    def run(self):
        self.cfg._optimizer = None
        self.cfg._scheduler = None
        self.cfg._melog = None

        model = self.qinco_model
        state_dict = self.cfg._ckpt_state_dict["model"]
        state_dict = {
            re.sub(r"residual_blocks.[0-9]+.(in_proj|out_proj)", "\\1", key): w
            for key, w in state_dict.items()
        }
        state_dict["data_mean"] = torch.tensor(self.DB_NORMS[self.db_name]).to(
            torch.float32
        )
        state_dict["data_std"] = torch.tensor(self.DB_STD[self.db_name]).to(
            torch.float32
        )

        if "steps.0.substep.codebook.weight" in state_dict:
            del state_dict["steps.0.substep.codebook.weight"]

        if self.cfg.ivf_in_use:  # Load IVF centroids
            ivf_centroids = torch.tensor(
                self.cfg._ivf_centroids_preloaded
            )  # IVF centroids are already normalized using these mean / std
            state_dict["steps.0.ivf_centroids.weight"] = ivf_centroids

        load_r = model.load_state_dict(state_dict, strict=True)
        self.accelerator.print(load_r)
        self.accelerator.print(f"Saving model to {self.cfg.output}")

        save_model(self.cfg, self.accelerator, model)
