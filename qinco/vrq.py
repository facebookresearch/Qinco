from typing import Tuple

import torch
import torch.nn as nn
import numpy as np
import faiss
from faiss.contrib.inspect_tools import get_additive_quantizer_codebooks
from accelerate.utils import broadcast

from .utils import pairwise_distances, corrected_mean_squared_error, extract_data_block
from .metrics import Timer

MAX_ENCODE_BS = 20_000
MAX_DATA_BLOCK = 200_000

class TorchSingleVQ(nn.Module):
    def __init__(self, codebook_size_K, features_d, device=None):
        super().__init__()
        self.device = device
        self.codebook_size_K = codebook_size_K
        self.features_d = features_d
        self.codebook = nn.Parameter(torch.rand((codebook_size_K, features_d), dtype=torch.float32).to(device))

    def init_weight(self, x_BD):
        x_BD = x_BD[:MAX_DATA_BLOCK].to(self.device)
        self.mean_1D = x_BD.mean(0).unsqueeze(0)
        self.std_1D = x_BD.std(0).unsqueeze(0)
        self.codebook.copy_(torch.rand_like(self.codebook).to(self.device))
        self.codebook.copy_(self.codebook * self.std_1D + self.mean_1D)

    def encode_pairwise_distances(self, batched_x_BD, codebook_KD):
        dists_BK = pairwise_distances(batched_x_BD, codebook_KD, approx='auto')
        return dists_BK

    def encode(self, x_BD):
        codes_B = torch.zeros(len(x_BD), dtype=int, device=x_BD.device)
        for i in range(0, len(x_BD), MAX_ENCODE_BS):
            batched_x_BD = x_BD[i:i+MAX_ENCODE_BS].to(self.device)
            codebook_KD = self.codebook
            dists_BK: torch.Tensor = self.encode_pairwise_distances(batched_x_BD, codebook_KD)
            codes_B[i:i+MAX_ENCODE_BS] = dists_BK.argmin(-1)
        return codes_B

    def quantize(self, x_BD):
        codes_B = self.encode(x_BD)
        quant_x_BD = self.codebook[codes_B].to(x_BD.device)
        return quant_x_BD, codes_B

    def decode(self, codes):
        return self.codebook[codes].to(codes.device)

    def _static_train_step(self, x_BD):
        # Registers for sums of codes assigned to each centroid
        sum_codes = torch.zeros_like(self.codebook, device=self.device)
        code_xcount = torch.zeros(self.codebook_size_K, device=self.device, dtype=int)

        # Assign points to centroids
        for i in range(0, len(x_BD), MAX_DATA_BLOCK):
            batched_x_BD = x_BD[i:i+MAX_DATA_BLOCK].to(self.device)
            codes_B = self.encode(batched_x_BD)
            code_xcount.index_add_(0, codes_B, torch.ones(len(batched_x_BD), dtype=int, device=self.device))
            sum_codes.index_add_(0, codes_B, batched_x_BD)

        # Fill empty centroids with random codebooks
        rand_centroids = torch.rand_like(self.codebook).to(self.device)
        sum_codes += rand_centroids * (code_xcount < 0.5).unsqueeze(-1)
        code_xcount = torch.maximum(code_xcount, torch.tensor(1, device=self.device))

        # Assign mean of assign vectors to each centroid
        self.codebook.copy_(sum_codes / code_xcount.unsqueeze(-1))

        return code_xcount

    def train_static_rq(self, x_BD, steps):
        self.init_weight(x_BD)
        self.total_steps = steps
        for i_step in range(steps):
            self.cur_step = i_step
            self._static_train_step(x_BD)

    def get_centroids(self):
        return self.codebook.cpu()


class TorchRQ(nn.Module):
    def __init__(self, cfg, features_d):
        super().__init__()
        self.cfg = cfg
        self.device =  cfg._accelerator.device

        self.stages = nn.ModuleList()
        for s in range(cfg._M_ivf):
            if s == 0 and cfg._ivf_book is not None:
                step_vq = cfg._ivf_book
            else:
                step_vq = TorchSingleVQ(
                    cfg.K,
                    features_d,
                    self.device,
                )
            self.stages.append(step_vq)

    def train_static_rq(self, x, steps):
        if len(x) <= MAX_DATA_BLOCK:
            x = x.to(self.device)
        for i_stage, vq in enumerate(self.stages):
            if i_stage == 0 and self.cfg._ivf_book:
                self.cfg._accelerator.print(f"{self.__class__.__name__} - IVF stage {i_stage}")
            else:
                self.cfg._accelerator.print(f"{self.__class__.__name__} - training stage {i_stage}")
                vq.train_static_rq(x, steps)

            quantized_x = vq.quantize(x)[0].to(x.device)
            x = x - quantized_x

    def quantize(self, x, compute_entropy=False, return_codes=False):
        if len(x) > MAX_DATA_BLOCK:
            xq = torch.zeros_like(x, device='cpu')
            for i in range(0, len(x), MAX_DATA_BLOCK):
                xq[i:i+MAX_DATA_BLOCK] = self.quantize(x[i:i+MAX_DATA_BLOCK], compute_entropy)
                compute_entropy = False # Compute on first data block
            return xq

        x_device = x.device
        x = x.to(self.device)
        all_codes = []
        xq = torch.zeros_like(x, device=self.device)
        for i_stage, vq in enumerate(self.stages):
            quantized_x, codes = vq.quantize(x - xq)
            xq += quantized_x
            all_codes.append(codes)
            if compute_entropy and not (i_stage == 0 and self.cfg._ivf_book):
                usage = torch.zeros(vq.codebook_size_K, device=self.device)
                code_ids, counts = codes.unique(return_counts=True)
                usage[code_ids] += counts
        if return_codes:
            return all_codes
        return xq.to(x_device)

    def decode(self, codes):
        quant_x = None
        for i_stage, vq in enumerate(self.stages):
            remain = vq.decode(codes[i_stage])
            quant_x = remain if quant_x is None else (quant_x + remain)
        return quant_x

    def get_centroids(self):
        return [c for c in [vq.get_centroids() for vq in self.stages] if c != 'ivf']

    ##### Use as main model #####

    def forward(self, x_in, *args, step='train', **kwargs):
        assert step in ['train', 'encode', 'decode']

        if step == 'train':
            return self._train_encode_decode(x_in, *args, **kwargs)
        elif step == 'encode':
            codes = self.quantize(x_in, return_codes=True)
            return codes
        elif step == 'decode':
            assert isinstance(x_in, list)
            x_quantized = self.decode(x_in)
            return x_quantized
        else:
            raise ValueError(f"{step=}")

    def _train_encode_decode(self, x_in):
        with torch.no_grad():
            codes = self(x_in, step='encode')
        codes = [c.clone() for c in codes]
        x_hat = self(codes, step='decode')

        losses = {'mse': ((x_in - x_hat)**2).mean()}
        return codes, None, losses


class StackOfFaissRQ:
    def __init__(self, cfg, xt_shape):
        bs, features = xt_shape
        self.rq_stack = []
        nbits = int(np.log2(cfg.K))
        for _ in range(cfg._M_ivf):
            rq = faiss.ResidualQuantizer(features, 1, nbits)
            self.rq_stack.append(rq)

    def train_static_rq(self, xt):
        quant_xt = torch.zeros_like(xt)
        for rq in self.rq_stack:
            step_remain = xt - quant_xt
            rq.train(step_remain) # pylint: disable=all
            remain_quant = rq.decode(rq.compute_codes(step_remain)) # pylint: disable=all
            quant_xt += remain_quant
        return quant_xt

    def quantize(self, x):
        quant_x = torch.zeros_like(x)
        for rq in self.rq_stack:
            step_remain = x - quant_x
            remain_quant = rq.decode(rq.compute_codes(step_remain)) # pylint: disable=all
            quant_x += remain_quant
        return quant_x

    def get_centroids(self):
        return [get_additive_quantizer_codebooks(rq)[0] for rq in self.rq_stack]

def rq_faiss(cfg, xt, xval):
    rq = StackOfFaissRQ(cfg, xt.shape)

    # Faiss training
    quant_xt = rq.train_static_rq(xt) # pylint: disable=all
    quant_xval = rq.quantize(xval)
    rq_centroids = rq.get_centroids()

    return quant_xt, quant_xval, rq_centroids

def apply_rq_on_vectors(cfg, xt: torch.tensor, xval: torch.tensor, return_rq_model: bool = False) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """Train RQ on a set of vectors, and apply it on the train and test datasets for evaluation.

    Args:
        cfg: Configuration object
        xt (torch.tensor): training set
        xval (torch.tensor): validation set

    Returns:
       Tuple[torch.tensor, torch.tensor, torch.tensor]:
        - First tensor: quantized training set (xt)
        - Second tensor: quantized validation set (xval)
        - Third tensor: centroids of the RQ
    """
    if cfg.qinco1_mode:
        cfg._accelerator.print(f"QINCo-1 mode, using faiss to train RQ")
        quant_xt, quant_xval, rq_centroids = rq_faiss(cfg, xt, xval)
        rq_centroids = [torch.tensor(c) for c in rq_centroids]
    else:
        B, features = xt.shape
        rq = TorchRQ(cfg, features).to(cfg._accelerator.device)
        with torch.no_grad():
            xt = xt.to(torch.float32)
            xval = xval.to(torch.float32)
            rq.train_static_rq(xt, 10)

            quant_xt = rq.quantize(xt, compute_entropy=True)
            quant_xval = rq.quantize(xval)
            if return_rq_model:
                rq_centroids = rq
            else:
                rq_centroids = rq.get_centroids()

    return quant_xt, quant_xval, rq_centroids


####################################################################
# RQ on QINCo dataset
####################################################################

RQ_MAX_ELEMS = 1_000_000


@torch.no_grad
def train_rq(cfg, train_dataset, val_dataset, return_rq_model=False):
    """trains a residual quantizer and retuns its codebook tables"""
    print = cfg._accelerator.print
    # Load data
    rq_timer = Timer()
    with rq_timer(reset=True):
        xt_BD = extract_data_block(train_dataset, RQ_MAX_ELEMS)
        xval_BD = extract_data_block(val_dataset, RQ_MAX_ELEMS)

    print(f"Loading data took {rq_timer}")
    assert xt_BD.shape[1:] == xval_BD.shape[1:]
    print(f"Train on data of shape {xt_BD.shape} (and {xt_BD.shape} before mapping)")

    B, D = xt_BD.shape
    print(f"RQ applied with {D} features")

    # RQ training & eval
    with rq_timer(reset=True):
        print(f"training RQ {cfg.K}")
        quant_xt_BD, quant_xval_BD, rq_centroids = apply_rq_on_vectors(cfg, xt_BD, xval_BD, return_rq_model=return_rq_model)
    print(f"Training & decoding took {rq_timer}")

    MSE = corrected_mean_squared_error(cfg, quant_xt_BD, xt_BD)
    MSE_val = corrected_mean_squared_error(cfg, quant_xval_BD, xval_BD)

    print(f"train set MSE={MSE:g} validation MSE={MSE_val:g} (avg={MSE_val / D:g})")
    if not return_rq_model:
        print(f"RQ centroids size {[tuple(c.shape) for c in rq_centroids]}")
    else:
        print(f"Trained RQ model: {rq_centroids}")
    return rq_centroids, float(MSE_val)

@torch.no_grad
def train_rq_centroids(cfg, train_dataset, val_dataset):
    accelerator = cfg._accelerator
    rq_centroids = None

    accelerator.print("====================== residual quantizer training")

    rq_centroids, cfg._rq_mse = train_rq(cfg, train_dataset, val_dataset)
    accelerator.wait_for_everyone()

    # Ensure everyone has the same centroids
    if accelerator.num_processes > 1:
        if not accelerator.is_main_process:
            rq_centroids = [torch.zeros_like(c) for c in rq_centroids]
        rq_centroids = [broadcast(c.to(accelerator.device)).cpu() for c in rq_centroids]

    return rq_centroids