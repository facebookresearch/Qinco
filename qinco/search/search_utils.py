# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import faiss
import numpy as np
import torch


###############################################################
# Utilities
###############################################################


def batched_db(db, bs):  # Returns (i_batch, batch, n_batches, log_step)
    for i_bs in range(0, db.shape[0], bs):
        batch_bD = db[i_bs : i_bs + bs]
        i_batch = i_bs // bs
        yield i_batch, batch_bD


def show_mem():
    free, tot = torch.cuda.mem_get_info()
    free = free / 2**30
    tot = tot / 2**30
    print(f"GPU memory, {free:.2f}G free / {tot:.2f}G ({tot-free:.2f}G in use)")


class EncodedDBIterator:
    def __init__(self, cfg, base_path):
        assert base_path.endswith(".npz")
        self.part_base_path = base_path[:-4]

        data_infos = np.load(base_path)
        self.n_parts = int(data_infos["n_parts"])

        assert cfg.K is None or cfg.K == int(data_infos["K"])
        cfg.K = int(data_infos["K"])
        assert cfg.M is None or cfg.M == int(data_infos["M"])
        cfg.M = int(data_infos["M"])
        assert cfg._D is None or cfg._D == int(data_infos["D"])
        cfg._D = int(data_infos["D"])

        self.cur_i_part = None
        self.cur_i_batch = None
        self.part_n_batches = None
        self.total_n_batches = None
        self.batch_start_id = None
        self.batch_end_id = None
        self.n_samples = None

    def iter(self, batch_size=None):
        self.batch_start_id = 0
        for i_part in range(self.n_parts):
            path = self.part_base_path + f".part_{i_part}.npz"
            db_codes = np.load(path)["codes"]

            bs = batch_size or len(db_codes)
            self.cur_i_part = i_part + 1
            self.part_n_batches = math.ceil(len(db_codes) / bs)
            self.total_n_batches = self.part_n_batches * self.n_parts
            self.n_samples = self.n_parts * len(db_codes)

            for ib in range(0, len(db_codes), bs):
                self.cur_i_batch = i_part * self.part_n_batches + ib // bs
                batch = db_codes[ib : ib + bs]
                self.batch_end_id = self.batch_start_id + len(batch)
                yield batch

                self.batch_start_id += len(batch)

    def load_all(self):
        parts = [p for p in self.iter()]
        return np.concatenate(parts, axis=0)


##### Least-squares solution of Additive Quantization tables (in numpy) #####


def one_hot_matrix_codes(codes, k):
    """return a one-hot matrix where each code is represented as a 1"""
    nt, M = codes.shape
    tab = np.zeros((nt * M, k), dtype="float32")
    tab[np.arange(nt * M), codes.ravel()] = 1
    return tab.reshape(nt, M, k)


def compute_fixed_aq_codebooks(xt, train_codes, k):
    """estimate fixed codebooks that minimize the reconstruction loss
    w.r.t. xt given the train_codes"""
    nt, M = train_codes.shape
    nt2, d = xt.shape
    assert nt2 == nt

    onehot_codes = one_hot_matrix_codes(train_codes, k).reshape((nt, -1))
    codebooks, _, _, _ = np.linalg.lstsq(onehot_codes, xt, rcond=None)
    codebooks = codebooks.reshape((M, k, d))
    return codebooks.astype(np.float32)


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


##### Additional Faiss functions #####


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
