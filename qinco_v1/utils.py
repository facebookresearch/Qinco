# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from multiprocessing.pool import ThreadPool

import numpy as np
import torch

try:
    import faiss
except ImportError:
    print("Faiss missings, some functionality will not work")


def mean_squared_error(x, y):
    """one-liner to compute the MSE between 2 sets of vectors"""
    return float(((x - y) ** 2).sum(1).mean())


###############################################################
# Distance computations and nearest neighbor assignment (in pytorch)
###############################################################


def pairwise_distances(a, b):
    """
    a (torch.Tensor): Shape [na, d]
    b (torch.Tensor): Shape [nb, d]

    Returns (torch.Tensor): Shape [na,nb]
    """
    anorms = (a**2).sum(-1)
    bnorms = (b**2).sum(-1)
    return anorms[:, None] + bnorms - 2 * a @ b.T


def compute_batch_distances(a, b):
    """
    a (torch.Tensor): Shape [n, a, d]
    b (torch.Tensor): Shape [n, b, d]

    Returns (torch.Tensor): Shape [n,a,b]
    """
    anorms = (a**2).sum(-1)
    bnorms = (b**2).sum(-1)
    # return anorms.unsqueeze(-1) + bnorms.unsqueeze(1) - 2 * torch.einsum('nad,nbd->nab',a,b)
    return (
        anorms.unsqueeze(-1) + bnorms.unsqueeze(1) - 2 * torch.bmm(a, b.transpose(2, 1))
    )


def assign_batch_multiple(x, zqs):
    """
    Assigns a batch of vectors to a batch of codebooks

    x (torch.Tensor) Shape: [bs x d]
    zqs (torch.Tensor) All possible next quantization vectors per elements in batch. Shape: [bs x ksq x d]

    Returns:
    codes (torch.int64) Indices of selected quantization vector per batch element. Shape: [bs]
    quantized (torch.Tensor) The selected quantization vector per batch element. Shape: [bs x d]
    """
    bs, d = x.shape
    bs, K, d = zqs.shape

    L2distances = compute_batch_distances(x.unsqueeze(1), zqs).squeeze(1)  # [bs x ksq]
    idx = torch.argmin(L2distances, dim=1).unsqueeze(1)  # [bsx1]
    quantized = torch.gather(zqs, dim=1, index=idx.unsqueeze(-1).repeat(1, 1, d))
    return idx.squeeze(1), quantized.squeeze(1)


def assign_to_codebook(x, c, bs=16384):
    """find the nearest centroid in matrix c for all the vectors
    in matrix x. Compute by batches if necessary to spare GPU memory
    (bs is the batch size)"""
    nq, d = x.shape
    nb, d2 = c.shape
    assert d == d2
    if nq * nb < bs * bs:
        # small enough to represent the whole distance table
        dis = pairwise_distances(x, c)
        return dis.argmin(1)

    # otherwise tile computation to avoid OOM
    res = torch.empty((nq,), dtype=torch.int64, device=x.device)
    cnorms = (c**2).sum(1)
    for i in range(0, nq, bs):
        xnorms = (x[i : i + bs] ** 2).sum(1, keepdim=True)
        for j in range(0, nb, bs):
            dis = xnorms + cnorms[j : j + bs] - 2 * x[i : i + bs] @ c[j : j + bs].T
            dmini, imini = dis.min(1)
            if j == 0:
                dmin = dmini
                imin = imini
            else:
                (mask,) = torch.where(dmini < dmin)
                dmin[mask] = dmini[mask]
                imin[mask] = imini[mask] + j
        res[i : i + bs] = imin
    return res


###############################################################
# Least-squares solution of Additive Quantization tables (in numpy)
###############################################################


def one_hot(codes, k):
    """return a one-hot matrix where each code is represented as a 1"""
    nt, M = codes.shape
    tab = np.zeros((nt * M, k), dtype="float32")
    tab[np.arange(nt * M), codes.ravel()] = 1
    return tab.reshape(nt, M, k)


def compute_fixed_codebooks(xt, train_codes, k=256):
    """estimate fixed codebooks that minimize the reconstruction loss
    w.r.t. xt given the train_codes"""
    nt, M = train_codes.shape
    nt2, d = xt.shape
    assert nt2 == nt
    onehot_codes = one_hot(train_codes, k).reshape((nt, -1))
    codebooks, _, _, _ = np.linalg.lstsq(onehot_codes, xt, rcond=None)
    codebooks = codebooks.reshape((M, k, d))
    return codebooks


def reconstruct_from_fixed_codebooks(codes, codebooks):
    """reconstruct vectors from thier codes and the fixed codebooks"""
    M = codes.shape[1]
    assert codebooks.shape[0] == M
    for m in range(M):
        xi = codebooks[m, codes[:, m]]
        if m == 0:
            recons = xi
        else:
            recons += xi
    return recons


def reconstruct_from_fixed_codebooks_parallel(all_codes, codebooks, nt=16):
    """parallel implementation of the fixed codebook reconstrcution"""
    n, M = all_codes.shape
    assert codebooks.shape[0] == M
    d = codebooks.shape[2]
    all_recons = np.empty((n, d), dtype=codebooks.dtype)

    def recons_slice(t):
        i0, i1 = t * n // nt, (t + 1) * n // nt
        codes = all_codes[i0:i1]
        recons = all_recons[i0:i1]
        for m in range(M):
            xi = codebooks[m, codes[:, m]]
            if m == 0:
                recons[:] = xi
            else:
                recons += xi

    with ThreadPool(nt) as pool:
        pool.map(recons_slice, range(nt))
    return all_recons


###############################################################
# Additional Faiss functions
###############################################################


def refine_distances(xq, xb, I):
    """Recompute distances between xq[i] and xb[I[i, :]]"""
    nq, k = I.shape
    xq = np.ascontiguousarray(xq, dtype="float32")
    nq2, d = xq.shape
    xb = np.ascontiguousarray(xb, dtype="float32")
    nb, d2 = xb.shape
    I = np.ascontiguousarray(I, dtype="int64")
    assert nq2 == nq
    assert d2 == d
    D = np.empty(I.shape, dtype="float32")
    D[:] = np.inf
    faiss.fvec_L2sqr_by_idx(
        faiss.swig_ptr(D),
        faiss.swig_ptr(xq),
        faiss.swig_ptr(xb),
        faiss.swig_ptr(I),
        d,
        nq,
        k,
    )
    return D


def add_to_ivfaq_index(index, xb_codes, Icoarse, xb_norms, i_base=0):
    """
    Fill in a Faiss IVFAdditiveQuantizer index with pre-computed codes.

    index: IVFAdditiveQuantizer to fill in
    xb_codes: codes to add
    Icoarse: corresponding invlist indexes
    xb_norms: squared norms of the vectors to index
    """
    n, M = xb_codes.shape
    (n2,) = Icoarse.shape
    assert n2 == n
    (n2,) = xb_norms.shape
    assert n2 == n
    assert M == index.aq.M

    o = np.argsort(Icoarse)
    counts = np.bincount(Icoarse, minlength=index.nlist)
    i0 = 0
    for list_no in range(index.nlist):
        i1 = i0 + counts[list_no]
        ids = o[i0:i1].astype("int64")
        assert np.all(Icoarse[ids] == list_no)
        codes = xb_codes[ids]
        norms = xb_norms[ids]
        codes = codes.astype("int32")
        n = len(ids)
        packed_codes = np.zeros((n, index.rq.code_size), dtype="uint8")
        index.rq.pack_codes(
            n,
            faiss.swig_ptr(codes),
            faiss.swig_ptr(packed_codes),
            -1,
            faiss.swig_ptr(norms),
            None,
        )
        ids += i_base
        index.invlists.add_entries(
            list_no, n, faiss.swig_ptr(ids), faiss.swig_ptr(packed_codes)
        )
        i0 = i1
    index.ntotal = index.invlists.compute_ntotal()
