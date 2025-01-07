# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import time

import faiss
import numpy as np
import torch
import torch.distributed
import torch.multiprocessing as mp
from faiss.contrib.inspect_tools import get_additive_quantizer_codebooks
from torch import optim
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

import datasets
import model_qinco
from utils import assign_to_codebook, mean_squared_error


def fix_random_seed(seed):
    torch.random.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


####################################################################
# RQ training
#####################################################################


def train_rq(args, xt, xval):
    """trains a residual quantizer and retuns its codebook tables"""
    nbit = int(np.log2(args.K))
    print(f"training RQ {args.M}x{nbit}, beam_size={args.rq_beam_size}")
    t0 = time.time()
    rq = faiss.ResidualQuantizer(xt.shape[1], args.M, nbit)
    rq.max_beam_size
    rq.max_beam_size = args.rq_beam_size
    rq.train(xt)
    print(f"[{time.time() - t0:.2f} s] training done")
    MSE = mean_squared_error(rq.decode(rq.compute_codes(xt)), xt)
    MSE_val = mean_squared_error(rq.decode(rq.compute_codes(xval)), xval)
    print(f"train set {MSE=:g} validation MSE={MSE_val:g}")
    rq_centroids = np.array(get_additive_quantizer_codebooks(rq))
    print(f"RQ centroids size {rq_centroids.shape}")
    return rq_centroids


####################################################################
# QINCo training -- single GPU
####################################################################


def initialize_model(model, rq_centroids, ivf_centroids):
    """copy the RQ centroids into the codebooks. Leave the
    linear layers with their default pytorch initialization"""
    with torch.no_grad():
        assert len(rq_centroids) == model.M
        if ivf_centroids is None:
            model.codebook0.weight.copy_(
                torch.from_numpy(rq_centroids[0]) / model.db_scale
            )
            m = 1
        else:
            model.codebook0.weight.copy_(
                torch.from_numpy(ivf_centroids) / model.db_scale
            )
            m = 0

        for step in model.steps:
            step.codebook.weight.copy_(
                torch.from_numpy(rq_centroids[m]) / model.db_scale
            )
            m += 1


class Scheduler:
    """updates the learning rate, decides whether to stop and stores the model
    if it is the best one so far"""

    def __init__(self, lr0, filename):
        self.lr = lr0  # will be updated depending on loss values
        self.filename = filename
        self.verbose = True
        self.loss_values = []
        self.last_lr_update = 0

    def quiet(self):
        "not verbose and do not save models"
        self.verbose = False
        self.filename = None

    def append_loss(self, loss, model):
        self.loss_values.append(loss)
        loss_values = np.array(self.loss_values, dtype=float)
        epoch = len(loss_values)

        best_loss_value = loss_values.min()
        if self.filename and loss_values[-1] == best_loss_value:
            print("Best validation loss so far, storing", self.filename)
            torch.save(model, self.filename)

        # check if we need to stop optimization
        if epoch > self.last_lr_update + 50 and np.all(
            loss_values[-50:] > best_loss_value
        ):
            if self.verbose:
                print("Val loss did not improve for 50 epochs, stopping")
            self.last_lr_update = epoch
            self.lr = 0
        elif epoch > self.last_lr_update + 10 and np.all(
            loss_values[-10:] > best_loss_value
        ):
            # check if we need to reduce the learning rate
            if self.verbose:
                print("Val loss did not improve for 10 epochs, reduce LR")
            self.last_lr_update = epoch
            self.lr /= 10
            if self.lr < 2e-6:
                if self.verbose:
                    print("LR too small, stopping")
                self.lr = 0

    def should_stop(self):
        return self.lr == 0


def train_one_epoch(model, xt, ivf_xt_assign, idx_batches, optimizer, verbose=True):
    """run one epoch of training"""
    device = next(model.parameters()).device
    d = xt.shape[1]
    sum_loss = 0
    t0 = time.time()
    for i, idx_batch in enumerate(idx_batches):
        model.zero_grad()
        batch = xt[idx_batch]
        batch = torch.from_numpy(batch).to(device) / model.db_scale
        if ivf_xt_assign is None:
            batch_code0 = None
        else:
            batch_code0 = torch.from_numpy(ivf_xt_assign[idx_batch]).to(device)

        codes, xhat, losses = model(batch, code0=batch_code0)
        loss = losses.sum() / len(batch) / d
        loss.backward()
        optimizer.step()

        loss = loss.item()
        sum_loss += loss
        if not verbose:
            continue
        print(
            f"[{time.time() - t0:.2f} s] train {i} / {len(idx_batches)} "
            f"loss={loss:g}",
            end="\r",
            flush=True,
        )
    return sum_loss / len(idx_batches)


def compute_MSE(model, x, bs):
    """compute MSE on validation set"""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        sum_errs = 0
        n = 0
        for i0 in range(0, len(x), bs):
            batch = torch.from_numpy(x[i0 : i0 + bs]).to(device) / model.db_scale
            codes, xhat = model.encode(batch)
            sum_errs += ((xhat - batch) ** 2).sum().item() * model.db_scale**2
            n += len(batch)
            print(
                f"[{time.time() - t0:.2f} s] inference {n} / {len(x)} "
                f"MSE={sum_errs / n:g}",
                end="\r",
                flush=True,
            )
    model.train()
    return sum_errs / len(x)


def train(args, xt, xval, model, ivf_xt_assign=None):
    seed = 123

    bs = args.batch_size
    t0 = time.time()

    MSE_val = compute_MSE(model, xval, bs=args.batch_size)
    print(f"Before optimization: val MSE={MSE_val:g}")
    scheduler = Scheduler(args.lr, args.model)

    for epoch in range(args.max_epochs):
        lr = scheduler.lr
        print(f"[{time.time() - t0:.2f} s] epoch {epoch} {lr=:g}")

        rs = np.random.RandomState(epoch + seed)
        perm = rs.permutation(len(xt))
        idx_batches = [perm[i0 : i0 + bs] for i0 in range(0, len(xt), bs)]

        optimizer = optim.Adam(model.parameters(), lr=lr)
        mean_loss = train_one_epoch(model, xt, ivf_xt_assign, idx_batches, optimizer)

        MSE_val = compute_MSE(model, xval, bs=args.batch_size)

        print(f"End of epoch {epoch} train loss {mean_loss:g} val MSE={MSE_val:g}")
        scheduler.append_loss(MSE_val, model)

        if scheduler.should_stop():
            break

    print("Training done")


####################################################################
# QINCo training -- multiple GPUs
####################################################################


def train_job(rank, port, args, xt, xval, model, ivf_xt_assign=None):
    """function that gets called on one process per GPU"""
    world_size = args.ngpu

    print(f"Start train_job {rank=:}")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # move the model to the GPU and set up distributed dataparallel
    torch.cuda.set_device(rank)
    device = "cuda:%d" % rank
    model.to(device)
    model = DDP(model, device_ids=[rank])
    model.db_scale = model.module.db_scale

    bs = args.batch_size
    bs2 = bs // world_size  # batch size per GPU
    seed = 123

    t0 = time.time()

    # prepare buffers for communication
    mean_loss = torch.zeros(1).to(device)
    MSE_val = torch.zeros(1).to(device)

    # initial MSE
    if rank == 0:
        print(f"Setting up distribtued data parallel bs={bs}")
        MSE_val[0] = compute_MSE(model.module, xval, bs=args.batch_size)
        print(f"Before optimization: val MSE={MSE_val[0]:g}", flush=True)

    torch.distributed.broadcast(MSE_val, src=0)
    scheduler = Scheduler(args.lr, args.model)

    epoch0 = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        if rank == 0:
            print("Restarting from checkpoint", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        epoch0 = checkpoint["epoch"] + 1
        scheduler = checkpoint["scheduler"]
        state_dict = {k: v.to(device) for k, v in checkpoint["state_dict"].items()}
        model.module.load_state_dict(state_dict)

    if rank != 0:
        scheduler.quiet()

    # start training
    for epoch in range(epoch0, args.max_epochs):
        lr = scheduler.lr
        if rank == 0:
            print(f"[{time.time() - t0:.2f} s] epoch {epoch} {lr=:g}")

        # prepare batch indexes
        rs = np.random.RandomState(epoch + seed)
        perm = rs.permutation(len(xt))
        # round to batch size so that all procs have the same number of batches
        perm = perm[: len(perm) // bs * bs]
        idx_batches = [perm[i0 + bs2 * rank :][:bs2] for i0 in range(0, len(perm), bs)]

        optimizer = optim.Adam(model.parameters(), lr=lr)

        mean_loss[0] = train_one_epoch(
            model, xt, ivf_xt_assign, idx_batches, optimizer, verbose=rank == 0
        )

        torch.distributed.all_reduce(mean_loss, op=torch.distributed.ReduceOp.SUM)
        mean_loss /= world_size

        if rank == 0:
            MSE_val[0] = compute_MSE(model.module, xval, bs=args.batch_size)
            print(
                f"End of epoch {epoch} train loss {mean_loss[0]:g} val MSE={MSE_val[0]:g}"
            )

        torch.distributed.broadcast(MSE_val, src=0)

        scheduler.append_loss(MSE_val[0].item(), model.module)

        if rank == 0 and args.checkpoint:
            print("storing checkpoint", args.checkpoint)
            state_dict = {k: v.cpu() for k, v in model.module.state_dict().items()}
            torch.save(
                {"epoch": epoch, "scheduler": scheduler, "state_dict": state_dict},
                args.checkpoint,
            )

        if scheduler.should_stop():
            break

    print(f"Stop train_job {rank=:}")
    destroy_process_group()


####################################################################
# Driver
####################################################################


def main():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group("what to do")
    aa(
        "--todo",
        default=["train_rq", "train", "train_ivf"],
        choices=["train_rq", "train"],
        nargs="+",
        help="what to do",
    )

    group = parser.add_argument_group("database")
    aa(
        "--db",
        default="bigann1M",
        choices=datasets.available_names,
        help="Dataset to handle",
    )
    aa("--training_data", default="", help="flat npy array with training vectors")
    aa("--nt", default=500_000, type=int, help="nb training vectors to use")
    aa("--nval", default=10_000, type=int, help="additional validation vectors")
    aa(
        "--db_scale",
        default=-1,
        type=float,
        help="force database scaling. If not set, the maximum is determined automatically from the training set.",
    )

    group = parser.add_argument_group("model parameters")
    aa("--ivf", default=False, action="store_true", help="train an IVF model")
    aa(
        "--lowrank",
        default=False,
        action="store_true",
        help="Train QINCo-LR that has a low-rank projection in the concatenation block",
    )
    aa("--M", default=8, type=int, help="number of sub-quantizers")
    aa("--L", default=4, type=int, help="number of residual blocks")
    aa("--K", default=256, type=int, help="codebook sizes")
    aa("--h", default=256, type=int, help="hidden dimension of residual blocks")
    aa(
        "--rq_beam_size",
        default=1,
        type=int,
        help="beam size for the initial residual quantizer",
    )

    group = parser.add_argument_group("optimization parameters")
    aa("--ngpu", default=1, type=int, help="number of GPUs to use")
    aa("--lr", default=1e-4, type=float, help="base learning rate")
    aa("--max_epochs", default=1000, type=int, help="max nb of epochs")
    aa("--batch_size", default=1024, type=int, help="batch size")

    group = parser.add_argument_group("files")
    aa("--RQ_filename", default="", help="Residual quantizer centroids (npy)")
    aa("--IVF_centroids", default="", help="IVF centroids file")
    aa("--model", default="", help="file to store the best trained model")
    aa(
        "--checkpoint",
        default="",
        help="checkpoint file to load/store during optimization",
    )

    args = parser.parse_args()

    print("args:", args)
    os.system(
        'echo -n "nb processors "; '
        "cat /proc/cpuinfo | grep ^processor | wc -l; "
        'cat /proc/cpuinfo | grep ^"model name" | tail -1'
    )
    os.system("nvidia-smi")
    fix_random_seed(1234)

    if args.training_data:
        print("Loading training data from", args.training_data)
        xt = np.load(args.training_data, mmap_mode="r")
        if len(xt) >= args.nt + args.nval:
            print(f"   Size {xt.shape} -> restrict to {args.nt} + {args.nval}")
            xt = np.array(xt[: args.nt + args.nval])
        else:
            raise RuntimeError("not enough training data")
        d = xt.shape[1]
    else:
        print(f"Loading dataset {args.db}")
        ds = datasets.dataset_from_name(args.db)
        print(f"   {ds}")
        xt = ds.get_train(maxtrain=args.nt + args.nval)
        d = ds.d

    xt, xval = xt[: -args.nval], xt[-args.nval :]

    print(f"Training set: {xt.shape}, validation: {xval.shape}")
    rq_centroids = None

    if not args.ivf:
        ivf_centroids = None
        ivf_xt_assign = None
    else:
        assert args.IVF_centroids, "IVF centroids should be provided"
        print("loading IVF centroids from", args.IVF_centroids)
        ivf_centroids = np.load(args.IVF_centroids)
        print("assigning training vectors to IVF centroids (on GPU)")
        xt_t = torch.from_numpy(xt).to("cuda:0")
        ivf_centroids_t = torch.from_numpy(ivf_centroids).to("cuda:0")
        ivf_xt_assign = assign_to_codebook(xt_t, ivf_centroids_t)
        ivf_xt_assign = ivf_xt_assign.cpu().numpy()

    if "train_rq" in args.todo:
        print("====================== residual quantizer training")

        if not args.ivf:
            rq_centroids = train_rq(args, xt, xval)
        else:
            xt_residual = xt - ivf_centroids[ivf_xt_assign]
            rq_centroids = train_rq(args, xt_residual, xval)

        if args.RQ_filename:
            print("storing RQ centroids to", args.RQ_filename)
            np.save(args.RQ_filename, rq_centroids)

    if "train" in args.todo:
        if rq_centroids is None:
            print("reading RQ from", args.RQ_filename)
            rq_centroids = np.load(args.RQ_filename)

        print("====================== training")

        print("Initializing model from RQ")

        if args.ivf:
            model = model_qinco.IVFQINCo(
                d, len(ivf_centroids), args.K, args.L, args.M, args.h
            )
        elif args.lowrank:
            model = model_qinco.QINCoLR(d, args.K, args.L, args.M, args.h)
        else:
            model = model_qinco.QINCo(d, args.K, args.L, args.M, args.h)
        print(model)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"   nb trainable parameters {num_params}")
        if args.db_scale > 0:
            model.db_scale = args.db_scale
        else:
            model.db_scale = xt.max()
        print(f"Setting scaling factor to {model.db_scale}")

        initialize_model(model, rq_centroids, ivf_centroids)

        if args.ngpu == 0:
            print("Running single GPU training")
            model.to("cpu")
            train(args, xt, xval, model, ivf_xt_assign)
        elif args.ngpu == 1:
            print("Running single GPU training")
            model.to("cuda:0")
            train(args, xt, xval, model, ivf_xt_assign)
        else:
            print(f"Running on {args.ngpu} GPUs")
            assert torch.cuda.device_count() >= args.ngpu
            port = np.random.randint(50000, 65000)
            mp.spawn(
                train_job,
                args=(port, args, xt, xval, model, ivf_xt_assign),
                nprocs=args.ngpu,
            )


if __name__ == "__main__":
    main()
