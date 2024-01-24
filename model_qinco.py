# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from utils import assign_batch_multiple, assign_to_codebook

####################################################################
# The base QINCo model
#####################################################################


class QINCoStep(nn.Module):
    """
    One quantization step for QINCo.
    Contains the codebook, concatenation block, and residual blocks
    """

    def __init__(self, d, K, L, h):
        nn.Module.__init__(self)

        self.d, self.K, self.L, self.h = d, K, L, h

        self.codebook = nn.Embedding(K, d)
        self.MLPconcat = nn.Linear(2 * d, d)

        self.residual_blocks = []
        for l in range(L):
            residual_block = nn.Sequential(
                nn.Linear(d, h, bias=False), nn.ReLU(), nn.Linear(h, d, bias=False)
            )
            self.add_module(f"residual_block{l}", residual_block)
            self.residual_blocks.append(residual_block)

    def decode(self, xhat, codes):
        zqs = self.codebook(codes)
        cc = torch.concatenate((zqs, xhat), 1)
        zqs = zqs + self.MLPconcat(cc)

        for residual_block in self.residual_blocks:
            zqs = zqs + residual_block(zqs)

        return zqs

    def encode(self, xhat, x):
        # we are trying out the whole codebook
        zqs = self.codebook.weight
        K, d = zqs.shape
        bs, d = xhat.shape

        # repeat so that they are of size bs * K
        zqs_r = zqs.repeat(bs, 1, 1).reshape(bs * K, d)
        xhat_r = xhat.reshape(bs, 1, d).repeat(1, K, 1).reshape(bs * K, d)

        # pass on batch of size bs * K
        cc = torch.concatenate((zqs_r, xhat_r), 1)
        zqs_r = zqs_r + self.MLPconcat(cc)

        for residual_block in self.residual_blocks:
            zqs_r = zqs_r + residual_block(zqs_r)

        # possible next steps
        zqs_r = zqs_r.reshape(bs, K, d) + xhat.reshape(bs, 1, d)
        codes, xhat_next = assign_batch_multiple(x, zqs_r)

        return codes, xhat_next - xhat


class QINCo(nn.Module):
    """
    QINCo quantizer, built from a chain of residual quantization steps
    """

    def __init__(self, d, K, L, M, h):
        nn.Module.__init__(self)

        self.d, self.K, self.L, self.M, self.h = d, K, L, M, h

        self.codebook0 = nn.Embedding(K, d)

        self.steps = []
        for m in range(1, M):
            step = QINCoStep(d, K, L, h)
            self.add_module(f"step{m}", step)
            self.steps.append(step)

    def decode(self, codes):
        xhat = self.codebook0(codes[:, 0])
        for i, step in enumerate(self.steps):
            xhat = xhat + step.decode(xhat, codes[:, i + 1])
        return xhat

    def encode(self, x, code0=None):
        """
        Encode a batch of vectors x to codes of length M.
        If this function is called from IVF-QINCo, codes are 1 index longer,
        due to the first index being the IVF index, and codebook0 is the IVF codebook.
        """
        M = len(self.steps) + 1
        bs, d = x.shape
        codes = torch.zeros(bs, M, dtype=int, device=x.device)

        if code0 is None:
            # at IVF training time, the code0 is fixed (and precomputed)
            code0 = assign_to_codebook(x, self.codebook0.weight)

        codes[:, 0] = code0
        xhat = self.codebook0.weight[code0]

        for i, step in enumerate(self.steps):
            codes[:, i + 1], toadd = step.encode(xhat, x)
            xhat = xhat + toadd

        return codes, xhat

    def forward(self, x, code0=None):
        with torch.no_grad():
            codes, xhat = self.encode(x, code0=code0)
        # then decode step by step and collect losses
        losses = torch.zeros(len(self.steps) + 1)
        xhat = self.codebook0(codes[:, 0])
        losses[0] = ((xhat - x) ** 2).sum()
        for i, step in enumerate(self.steps):
            xhat = xhat + step.decode(xhat, codes[:, i + 1])
            losses[i + 1] = ((xhat - x) ** 2).sum()
        return codes, xhat, losses


####################################################################
# IVF-QINCo model
#####################################################################


class IVFQINCo(QINCo):

    """
    QINCo quantizer with a pre-trained (and non-trainable) IVF in front of the first quantization step
    """

    def __init__(self, d, K_IVF, K, L, M, h):
        nn.Module.__init__(self)

        self.d, self.K_IVF, self.K = d, K_IVF, K
        self.L, self.M, self.h = L, M, h

        self.codebook0 = nn.Embedding(K_IVF, d, _freeze=True)

        self.steps = []
        for m in range(M):
            step = QINCoStep(d, K, L, h)
            self.add_module(f"step{m}", step)
            self.steps.append(step)


####################################################################
# QINCo-LR model
#####################################################################`


class QINCoLR(QINCo):
    """
    QINCo quantizer with an additional low-rank projection in the concatenation block.
    Especially useful for high-dimensional embeddings (i.e. large d).
    """

    def __init__(self, d, K, L, M, h):
        QINCo.__init__(self, d, K, L, M, h)

        for step in self.steps:
            # override the MLPconcat to avoid the d^2 Linear layer
            step.MLPconcat = nn.Sequential(
                nn.Linear(2 * d, h, bias=False), nn.Linear(h, d, bias=False)
            )


####################################################################
# PQ-QINCo model
#####################################################################`


class PQ_QINCo(nn.Module):
    def __init__(self, sub_quantizers, opq_matrix=None):
        """
        sub_quantizers: list of sub-quantizers, each of which is a QINCo model
        opq_matrix: if provided, this is the OPQ matrix to be used
        """
        nn.Module.__init__(self)
        self.db_scale = 1  # the db_scale is set per sub-quantizer
        self.sub_quantizers = sub_quantizers
        for m, q in enumerate(self.sub_quantizers):
            self.add_module(f"sub_quantizer_{m}", q)

        if opq_matrix is not None:
            self.opq_matrix = nn.Parameter(torch.from_numpy(opq_matrix))
        else:
            self.opq_matrix = None

    def encode(self, x):
        d0 = 0
        codes = []
        xhat = torch.zeros_like(x)

        if self.opq_matrix is not None:
            x = x @ self.opq_matrix.T

        for q in self.sub_quantizers:
            d1 = d0 + q.d
            code, xhat_sub = q.encode(x[:, d0:d1] / q.db_scale)
            codes.append(code)
            xhat[:, d0:d1] = xhat_sub * q.db_scale
            d0 = d1

        if self.opq_matrix is not None:
            xhat = xhat @ self.opq_matrix

        return torch.concatenate(codes, 1), xhat

    def decode(self, codes):
        c0 = 0
        x = []
        for q in self.sub_quantizers:
            c1 = c0 + q.M
            x.append(q.decode(codes[:, c0:c1]) * q.db_scale)
            c0 = c1
        x = torch.concatenate(x, 1)

        if self.opq_matrix is not None:
            x = x @ self.opq_matrix

        return x
