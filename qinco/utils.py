import logging
from pathlib import Path

import accelerate
import numpy as np
import torch
from accelerate import Accelerator


class SharedCfgState:
    """
    Wrapper for configuation + interval registers
    """

    def __init__(self, cfg):
        self._cfg = cfg

    def __getattribute__(self, __name: str):
        if __name.startswith("_"):
            return super().__getattribute__(__name)
        else:
            return self._cfg[__name]

    def __setattr__(self, __name: str, __value) -> None:
        if __name.startswith("_"):
            super().__setattr__(__name, __value)
        else:
            self._cfg[__name] = __value

    def __setitem__(self, __name: str, __value) -> None:
        return self.__setattr__(__name, __value)

    def __getitem__(self, __name: str) -> None:
        return self.__getattribute__(__name)


def ensure_path(path, parent=False):
    path = Path(path)
    if parent:
        path = path.parent
    path.mkdir(parents=True, exist_ok=True)


def format_memory(mem):
    if mem >= 2**40:
        return f"{mem/(2**40):.1f}T"
    if mem >= 2**30:
        return f"{mem/(2**30):.1f}G"
    if mem >= 2**20:
        return f"{mem/(2**20):.1f}M"
    if mem >= 2**10:
        return f"{mem/(2**10):.1f}K"
    return f"{mem:.1f}b"


def format_time(t):
    hours, t = divmod(t, 3600)
    minutes, seconds = divmod(t, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


###############################################################
# Torch utils
###############################################################


def torch_sum(tensor_list):
    """Takes a list of tensors of same shape, and returns the sum of the tensors with the shape of the first tensor"""
    if isinstance(tensor_list, torch.Tensor):
        if not tensor_list.numel():
            return 0.0
        return torch.sum(tensor_list, dim=0)
    if tensor_list is None or not tensor_list:
        return 0.0
    t0 = tensor_list[0]
    for t in tensor_list[1:]:
        t0 = t0 + t
    return t0


def corrected_mean_squared_error(cfg, x, y):
    """Compute the MSE between 2 sets of vectors, corrected with the normalization std"""
    if isinstance(x, torch.Tensor) != isinstance(y, torch.Tensor):
        if isinstance(x, torch.Tensor):
            x = x.cpu().detach().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()
    assert x.shape == y.shape
    err = float(((x - y) ** 2).mean(0).sum())
    err = err * cfg.mse_scale
    return err


def save_model(cfg, accelerator, model: torch.nn.Module):
    if accelerator.is_main_process:
        path = cfg.output
        ensure_path(path, parent=True)
        model = unwrap(model)
        saved_parameters = [
            "K",
            "M",
            "de",
            "dh",
            "L",
            "A",
            "B",
            "ivf_in_use",
            "ivf_K",
            "qinco1_mode",
        ]

        torch.save(
            {
                "epoch": (
                    cfg._cur_epoch + 1 if cfg._cur_epoch is not None else None
                ),  # Add +1 to count the number of COMPLETED epochs (and start from next one!)
                "model": model.state_dict(),
                "optimizer": (
                    cfg._optimizer.state_dict() if cfg._optimizer is not None else None
                ),
                "scheduler": (
                    cfg._scheduler.state_dict() if cfg._scheduler is not None else None
                ),
                "logger": cfg._melog.state_dict() if cfg._melog is not None else None,
                "parameters": {
                    p: cfg[p] for p in saved_parameters if cfg[p] is not None
                },
                "data_dim": cfg._D,
            },
            str(path),
        )


def load_saved_model_data(cfg, load_qinco):
    path = cfg.model

    # Load IVF centroids
    if cfg.ivf_centroids:
        cfg._ivf_centroids_preloaded = np.load(cfg.ivf_centroids)
        cfg.ivf_K, cfg._D = cfg._ivf_centroids_preloaded.shape

    # Load QINCo model weights
    cfg._ckpt_state_dict = None
    if load_qinco:
        assert (
            path is not None or cfg.task == "train"
        ), "Please provied a path to model weights using the 'model' argument"
        if cfg.task == "train":
            if path is None:
                return  # Start from scratch
            cfg._accelerator.print(f"Resuming training from {path}")
        cfg._accelerator.print(f"Load model checkpoint from {path}")
        assert Path(path).exists(), f"Can't find path {path}"

        state_dict = torch.load(
            str(cfg.model), map_location=torch.device("cpu"), weights_only=True
        )

        if "parameters" in state_dict:
            for arg, val in state_dict["parameters"].items():
                if cfg[arg] is None:
                    cfg[arg] = val
                elif arg == "A" and cfg[arg] > 0 and not val:
                    raise ValueError(
                        "Can't evaluate a model trained with A=0 (no candidates pre-selection) using a non-zero A value."
                    )
            cfg._D = state_dict["data_dim"]
        else:
            assert (
                cfg.task == "convert"
            ), "Missing model parameters is acceptable only for converting a model!"

        cfg._ckpt_state_dict = state_dict


def load_model(cfg, model: torch.nn.Module):
    state_dict = cfg._ckpt_state_dict
    if state_dict is not None:
        model_state_dict = state_dict["model"]
        if cfg.task in ["train"]:
            if state_dict["epoch"] is not None:
                cfg._cur_epoch = state_dict["epoch"]
            if state_dict["optimizer"] is not None:
                cfg._optimizer = state_dict["optimizer"]
            if state_dict["scheduler"] is not None:
                cfg._scheduler = state_dict["scheduler"]
            if state_dict["logger"] is not None:
                cfg._melog = state_dict["logger"]
        if not hasattr(model, "module"):
            model_state_dict = {
                key.replace("module.", ""): tensor
                for key, tensor in model_state_dict.items()
            }
        model.load_state_dict(model_state_dict, strict=(cfg.task != "convert"))


def extract_data_block(dataloader, max_elements=None, map=None):
    n_elements = 0
    all_blocks = []
    for block in dataloader:
        n_elements += len(block)
        if all_blocks and max_elements and n_elements > max_elements:
            break
        all_blocks.append(block)
    concat = np.concatenate if isinstance(block, np.ndarray) else torch.concat
    return concat(all_blocks)


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def merge_losses(dest, src):
    for k, v in src.items():
        dest[k] = dest.get(k, 0) + v
    return dest


###############################################################
# CUDA and Distributed computing
###############################################################


def unwrap(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return unwrap(model.module)
    elif hasattr(torch.distributed, "fsdp") and isinstance(
        model, torch.distributed.fsdp.FullyShardedDataParallel
    ):
        return unwrap(model.module)
    return model


class QAccelerator(Accelerator):
    h_log = logging.getLogger(__name__)

    def print(self, *args, sep=" ", level="info"):
        if self.is_local_main_process:
            msg = sep.join([str(a) for a in args])
            if level == "info":
                self.h_log.info(msg)
            else:
                raise ValueError(f"{level=} unknown")

    def print_nolog(self, *args, **kwargs):
        if self.is_local_main_process:
            print(*args, **kwargs)

    def prepare_test_data(self, dataloader, cfg):
        if dataloader is None:
            return dataloader
        return accelerate.data_loader.prepare_data_loader(
            dataloader,
            device=self.device,
            put_on_device=True,
            even_batches=(cfg.inference),
        )


def log_mem_info(accelerator):
    if accelerator.is_main_process:
        n_gpu = torch.cuda.device_count()
        if not n_gpu:
            accelerator.print(f"No GPU available")
        else:
            accelerator.print(f"{n_gpu} GPU(s) available")
        accelerator.print(
            f"Will use device [{accelerator.device}] on main process (with {accelerator.num_processes} processes)"
        )

        format_mem = lambda mem: f"{mem/1024**3:.2f}"
        for gpu_id in range(n_gpu):
            free, total = torch.cuda.mem_get_info(gpu_id)
            allocated = torch.cuda.memory_allocated(device=gpu_id)
            reserved = torch.cuda.memory_reserved(device=gpu_id)

            used = total - free
            total_free = format_mem(free + reserved - allocated)
            free, total, used = format_mem(free), format_mem(total), format_mem(used)
            device_name = torch.cuda.get_device_name(device=gpu_id)
            allocated = format_mem(allocated)
            reserved = format_mem(reserved)
            accelerator.print(
                f"Device {gpu_id}: {used} / {total} GB used ({free} GB free) | allocated={allocated} GB over reserved={reserved} GB | total_free={total_free} GB [{device_name}]"
            )


###############################################################
# Distance computations and nearest neighbor assignment (in pytorch)
###############################################################

AUTO_MAX_K_VALUE = 32


def pairwise_distances(a, b, approx="auto"):
    """
    a (torch.Tensor): Shape [na_1...na_k, d]
    b (torch.Tensor): Shape [nb_1...nb_l, d]

    Returns (torch.Tensor): Shape [na,nb]
    """
    assert a.shape[-1] == b.shape[-1]
    d = a.shape[-1]
    a_dims, b_dims = a.shape[:-1], b.shape[:-1]
    a, b = a.reshape(-1, d), b.reshape(-1, d)

    if approx == "auto":
        na, nb = len(a), len(b)
        approx = na > AUTO_MAX_K_VALUE and nb > AUTO_MAX_K_VALUE

    if approx:
        dists = approx_pairwise_distance(a, b)
    else:
        dists = exact_pairwise_distance(a, b)
    dists = dists.reshape(a_dims + b_dims)
    return dists


@torch.jit.script
def exact_pairwise_distance(a: torch.Tensor, b: torch.Tensor):
    """
    a (torch.Tensor): Shape [a, d]
    b (torch.Tensor): Shape [b, d]

    Returns (torch.Tensor): Shape [a,b]
    """
    return torch.pow(a.unsqueeze(-2) - b.unsqueeze(-3), 2).sum(-1)


@torch.jit.script
def approx_pairwise_distance(a: torch.Tensor, b: torch.Tensor):
    """
    a (torch.Tensor): Shape [a, d]
    b (torch.Tensor): Shape [b, d]

    Returns (torch.Tensor): Shape [a,b]
    """
    anorms = (a**2).sum(-1)
    bnorms = (b**2).sum(-1)
    return anorms[:, None] + bnorms - 2 * a @ b.T


def compute_batch_distances(a, b, approx="auto", max_b_els=1024 * 2048) -> torch.Tensor:
    """
    a (torch.Tensor): Shape [n, a, d]
    b (torch.Tensor): Shape [n, b, d]

    Returns (torch.Tensor): Shape [n,a,b]
    """
    assert a.ndim == 3 == b.ndim
    N, A, D = a.shape
    N, B, D = b.shape

    if approx == "auto":
        approx = A > AUTO_MAX_K_VALUE or B > AUTO_MAX_K_VALUE

    if not approx:
        return exact_compute_batch_distances(a, b)

    if N * B <= max_b_els:
        return approx_compute_batch_distances(a, b)
    else:  # For large sets of data
        dists = torch.zeros((N, A, B), device=a.device)
        bs = max_b_els // N
        for ib1 in range(0, B, bs):
            ib2 = min(B, ib1 + bs)
            dists[:, :, ib1:ib2] = approx_compute_batch_distances(a, b[:, ib1:ib2])
        return dists


@torch.jit.script
def approx_compute_batch_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    anorms = (a**2).sum(-1)
    bnorms = (b**2).sum(-1)
    return (
        anorms.unsqueeze(-1) + bnorms.unsqueeze(1) - 2 * torch.bmm(a, b.transpose(2, 1))
    )


@torch.jit.script
def exact_compute_batch_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.pow(a.unsqueeze(-2) - b.unsqueeze(-3), 2).sum(-1)
