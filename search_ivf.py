# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import time

import faiss
import numpy as np
import torch
from faiss.contrib.evaluation import OperatingPointsWithRanges

import datasets
from codec_qinco import encode
from utils import (
    add_to_ivfaq_index,
    compute_fixed_codebooks,
    mean_squared_error,
    reconstruct_from_fixed_codebooks,
    reconstruct_from_fixed_codebooks_parallel,
)

####################################################################
# Centroids training phase
####################################################################


def train_ivf_centroids(args, ds):
    print(f"load training vectors")
    xt = ds.get_train(args.nt)
    print(f"   xt shape {xt.shape=:}")
    d = ds.d
    km = faiss.Kmeans(
        d, args.n_centroids, niter=args.kmeans_iter, verbose=True, gpu=True
    )
    km.train(xt)
    return km.centroids


####################################################################
# Codes training phase
####################################################################


def run_train(args, model, ds, res):
    print(f"load training vectors")
    xt = ds.get_train(args.nt)
    print(f"   xt shape {xt.shape=:}")
    d = ds.d
    M = model.M
    if args.xt_codes:
        print("loading pretrained codes from", args.xt_codes)
        xt_codes = np.load(args.xt_codes)
        print(f"    shape", xt_codes.shape)
        assert xt_codes.shape == (len(xt), M + 1)
    else:
        print("encode trainset")
        t0 = time.time()
        xt_codes = encode(model, xt, bs=args.bs, is_float16=args.float16)
        res.t_xt_encode = time.time() - t0
        print(f"   done in {res.t_xt_encode:.3f} s")

    print("get IVF centroids from model")
    ivf_codebook = model.codebook0.weight.cpu().numpy() * model.db_scale
    print(f"{ivf_codebook.shape=:} {ivf_codebook.max()=:g}")
    print("train fixed codebook on residuals")
    t0 = time.time()
    print("   compute residuals")
    xt_residuals = xt - ivf_codebook[xt_codes[:, 0]]
    print("   train")
    codebooks = compute_fixed_codebooks(xt_residuals, xt_codes[:, 1:])
    res.t_train_codebook = time.time() - t0
    print(f"train done in {res.t_train_codebook:.2f} s")
    xt_fixed_recons = reconstruct_from_fixed_codebooks(xt_codes[:, 1:], codebooks)
    MSE = mean_squared_error(xt_fixed_recons, xt_residuals)
    res.MSE_train_residuals = MSE
    print(f"MSE on training set {MSE:g}")
    print("Train norms")
    # not really useful for _Nfloat encoding and even less for ST_decompress
    norms = ((xt_fixed_recons - xt_residuals) ** 2).sum(1)

    print("construct the index", args.index_key)
    index = faiss.index_factory(d, args.index_key)
    quantizer = faiss.downcast_index(index.quantizer)

    if args.quantizer_efConstruction > 0:
        print("set quantizer efConstruction to", args.quantizer_efConstruction)
        quantizer.hnsw.efConstruction = args.quantizer_efConstruction

    print("setting IVF centroids and RQ codebooks")
    t0 = time.time()
    print("    IVF centroids")
    assert ivf_codebook.shape[0] == index.nlist
    quantizer.add(ivf_codebook)
    print("    set codebook")
    assert codebooks.shape[0] == index.rq.M
    assert codebooks.shape[2] == index.rq.d
    rq_Ks = list(2 ** faiss.vector_to_array(index.rq.nbits))
    assert rq_Ks == [codebooks.shape[1]] * index.rq.M
    faiss.copy_array_to_vector(codebooks.ravel(), index.rq.codebooks)
    print("    train norms")
    index.rq.train_norm(len(norms), faiss.swig_ptr(norms))
    res.t_add_codebook = time.time() - t0
    index.rq.is_trained = True
    index.is_trained = True
    print(f"index ready in {res.t_add_codebook:.2f} s")

    return index


####################################################################
# Adding phase
####################################################################


def run_add(args, model, ds, index, res):
    quantizer = faiss.downcast_index(index.quantizer)
    codebooks = faiss.vector_to_array(index.rq.codebooks)
    ivf_codebook = quantizer.reconstruct_n()
    k = 1 << index.rq.nbits.at(0)
    M = index.rq.M
    d = ds.d
    codebooks = codebooks.reshape(M, k, d)

    if len(args.quantizer_efSearch) > 0:
        ef = args.quantizer_efSearch[0]
        print("set quantizer efSearch to", ef)
        quantizer.hnsw.efSearch = ef

    t0 = time.time()

    if args.xb_codes:
        print("adding from precomputed codes")

        def yield_codes():
            for fname in args.xb_codes:
                print(f"   [{time.time() - t0:.2f} s] load", fname)
                xb_codes = np.load(fname)
                yield xb_codes

    else:
        print("computing codes and adding")

        def yield_codes():
            for xb in ds.database_iterator(bs=args.add_bs):
                print(f"    [{time.time() - t0:.2f} s]    encode batch", xb.shape)
                xb_codes = encode(model, xb, bs=args.bs, is_float16=args.float16)
                yield xb_codes

    i0 = 0
    for xb_codes in yield_codes():
        print(f"    codes shape", xb_codes.shape)
        assert xb_codes.shape[1] == M + 1
        i1 = i0 + xb_codes.shape[0]
        xb_fixed_recons = reconstruct_from_fixed_codebooks_parallel(
            xb_codes[:, 1:], codebooks, nt=args.nthreads
        )
        # xb_residuals = xb - ivf_codebook[xb_codes[:, 0]]
        # MSE = mean_squared_error(xb_fixed_recons, xb_residuals)
        xb_norms = (xb_fixed_recons**2).sum(1)
        print(f"    add {i0}:{i1}")
        add_to_ivfaq_index(index, xb_codes[:, 1:], xb_codes[:, 0], xb_norms, i_base=i0)
        i0 = i1

    assert index.ntotal == ds.nb
    res.t_add = time.time() - t0
    print(f"add done in {res.t_add:.3f} s")


####################################################################
# Searching
####################################################################


def run_search(args, ds, index, res):
    quantizer = faiss.downcast_index(index.quantizer)
    assert index.ntotal == ds.nb

    print("loading CPU version of the model")
    model_cpu = torch.load(args.model, map_location="cpu")

    db_scale = model_cpu.db_scale
    print(f"   {db_scale=:g}")

    print("preparing index")
    index.parallel_mode
    index.parallel_mode = 3
    print("loading queries")
    xq = ds.get_queries()

    gt = ds.get_groundtruth()

    def compute_recalls(I):
        recalls = {}
        for rank in 1, 10, 100:
            recall = (I[:, :rank] == gt[:, :1]).sum() / gt.shape[0]
            recalls[rank] = float(recall)
        return recalls

    print(f"    {xq.shape=:} {gt.shape=:}")

    res.search_results = ivf_real_res = []
    cc = index.coarse_code_size()
    cc1 = index.sa_code_size()
    nq, d = xq.shape
    M = model_cpu.M
    listno_mask = index.nlist - 1
    print("start experiments")

    op = OperatingPointsWithRanges()
    op.add_range("nprobe", args.nprobe)
    if len(args.quantizer_efSearch) > 0:
        op.add_range("quantizer_efSearch", args.quantizer_efSearch)
    op.add_range("nshort", args.nshort)

    experiments = op.sample_experiments(args.n_autotune, rs=np.random.RandomState(123))
    print(f"Total nb experiments {op.num_experiments()}, running {len(experiments)}")

    for cno in experiments:
        key = op.cno_to_key(cno)
        parameters = op.get_parameters(key)
        print(f"{cno=:4d} {str(parameters):50}", end=": ", flush=True)

        if args.n_autotune == 0:
            pass  # don't optimize
        else:
            (max_perf, min_time) = op.predict_bounds(key)
            if not op.is_pareto_optimal(max_perf, min_time):
                print(
                    f"SKIP, {max_perf=:.3f} {min_time=:.3f}",
                )
                continue

        index.nprobe = parameters["nprobe"]
        nshort = parameters["nshort"]
        if "quantizer_efSearch" in parameters:
            quantizer.hnsw.efSearch = parameters["quantizer_efSearch"]

        t0 = time.time()
        D, I, codes = index.search_and_return_codes(xq, nshort, include_listnos=True)
        t1 = time.time()

        # decode
        codes2 = codes.reshape(nshort * nq, cc1)

        codes_int32 = np.zeros((nshort * nq, M + 1), dtype="int32")
        if cc == 2:
            codes_int32[:, 0] = codes2[:, 0] | (codes2[:, 1].astype(np.int32) << 8)
        elif cc == 3:
            codes_int32[:, 0] = (
                codes2[:, 0]
                | (codes2[:, 1].astype(np.int32) << 8)
                | (codes2[:, 2].astype(np.int32) << 16)
            )
        else:
            raise NotImplementedError

        # to avoid decode errors on -1 (missing shortlist result)
        # will be caught later because the id is -1 as well.
        codes_int32[:, 0] &= listno_mask

        codes_int32[:, 1:] = codes2[:, cc : M + cc]
        with torch.no_grad():
            shortlist = []
            for i in range(0, len(codes_int32), args.bs):
                code_batch = torch.from_numpy(codes_int32[i : i + args.bs])
                x_batch = model_cpu.decode(code_batch)
                shortlist.append(x_batch.numpy() * db_scale)

        t2 = time.time()
        shortlist = np.vstack(shortlist)
        shortlist = shortlist.reshape(nq, nshort, d)
        D_refined = ((xq.reshape(nq, 1, d) - shortlist) ** 2).sum(2)

        idx = np.argsort(D_refined, axis=1)
        I_refined = np.take_along_axis(I, idx[:, :100], axis=1)
        t3 = time.time()

        recalls_orig = compute_recalls(I)
        recalls = compute_recalls(I_refined)

        print(f"times {t1-t0:.3f}s + {t2-t1:.3f}s + {t3-t2:.3f}s " f"recalls {recalls}")

        op.add_operating_point(key, recalls[1], t3 - t0)

        ivf_real_res.append(
            dict(
                parameters=parameters,
                cno=cno,
                t_search=t1 - t0,
                t_decode=t2 - t1,
                t_dis=t3 - t2,
                recalls=recalls,
                recalls_orig=recalls_orig,
            )
        )


####################################################################
# Driver
####################################################################


def main():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    ivf_models_dir = "/checkpoint/ihuijben/ImplicitCodebookPredIVF/231026_fixH256/"

    group = parser.add_argument_group("what and how to compute")
    aa(
        "--todo",
        default=["train", "add", "search"],
        choices=["train_centroids", "train", "add", "search"],
        nargs="+",
        help="what to do",
    )
    aa("--bs", default=4096, type=int, help="batch size")
    aa("--add_bs", default=1000_000, type=int, help="batch size at add time")
    aa("--nthreads", default=32, type=int, help="number of OMP threads")

    group = parser.add_argument_group("QINCo model")
    aa("--model", default="", help="Model to load")
    aa("--device", default="cuda:0", help="pytorch device")
    aa("--float16", default=False, action="store_true", help="convert model to float16")

    group = parser.add_argument_group("database")
    aa(
        "--db",
        default="bigann1M",
        choices=datasets.available_names,
        help="Dataset to handle",
    )
    aa("--nt", default=1000_000, type=int, help="nb training vectors to use")
    aa("--xt_codes", default="", help="npy file with pre-encoded training vectors")
    aa(
        "--xb_codes",
        default=[],
        nargs="*",
        help="npy file with pre-encoded database vectors",
    )

    group = parser.add_argument_group("IVF centroids training")
    aa("--n_centroids", default=65536, type=int, help="number of centroids to train")
    aa("--kmeans_iter", default=10, type=int, help="number of k-means iterations")
    aa("--IVF_centroids", default="", help="where to store the IVF centroids")

    group = parser.add_argument_group("build index")
    aa("--index_key", default="IVF65536,RQ8x8_Nfloat", help="Faiss index key")
    aa("--trained_index", default="", help="load / store trained index")
    aa("--index", default="", help="load / store full index")
    aa("--quantizer_efConstruction", default=-1, type=int)
    aa(
        "--quantizer_efSearch",
        default=[],
        nargs="+",
        type=int,
        help="efSearch for the quantizer (used at add and search time)",
    )

    group = parser.add_argument_group("search parameters to try")
    aa(
        "--nprobe",
        default=[1, 4, 16, 64, 256, 1024],
        type=int,
        nargs="+",
        help="nprobe settings to try",
    )
    aa(
        "--nshort",
        default=[10, 20, 50, 100, 200, 500, 1000],
        type=int,
        nargs="+",
        help="shortlist sizes to try",
    )
    aa(
        "--n_autotune",
        default=0,
        type=int,
        help="number of autotune experiments (0=exhaustive exploration)",
    )

    args = parser.parse_args()

    print("args:", args)
    os.system(
        'echo -n "nb processors "; '
        "cat /proc/cpuinfo | grep ^processor | wc -l; "
        'cat /proc/cpuinfo | grep ^"model name" | tail -1'
    )
    os.system("nvidia-smi")

    # object to collect various stats
    res = argparse.Namespace()
    res.args = args.__dict__
    res.cpu_model = [l for l in open("/proc/cpuinfo", "r") if "model name" in l][0]

    res.cuda_devices = [
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    ]

    if args.nthreads != -1:
        print(f"set nb threads to {args.nthreads}")
        faiss.omp_set_num_threads(args.nthreads)

    ds = datasets.dataset_from_name(args.db)
    print(f"Prepared dataset {ds}")

    if "train_centroids" in args.todo:
        print("======== k-means clustering to compute IVF centroids")
        ivf_centroids = train_ivf_centroids(args, ds)

        if args.IVF_centroids:
            print("storing centroids in", args.IVF_centroids)
            np.save(args.IVF_centroids, ivf_centroids)

        # it does not make much sense to go further as the centroids
        # are an input of model training
        if args.todo == ["train_centroids"]:
            return

    print("loading model", args.model)
    model = torch.load(args.model)
    print("  database normalization factor", model.db_scale)
    model.eval()
    model.to(args.device)
    if args.float16:
        model.half()

    d = model.d
    k = model.K
    M = model.M

    index = None
    if "train" in args.todo:
        print("====================== training")

        index = run_train(args, model, ds, res)

        if args.trained_index:
            print("storing trained index in", args.trained_index)
            faiss.write_index(index, args.trained_index)

    if "add" in args.todo:
        print("====================== adding")

        if index is None and args.trained_index:
            print("loading pretrained index", args.trained_index)
            index = faiss.read_index(args.trained_index)
        elif index is None:
            raise RuntimeError("no pretrained index provided")

        run_add(args, model, ds, index, res)

        if args.index:
            print("storing index in", args.index)
            faiss.write_index(index, args.index)

    if "search" in args.todo:
        print("====================== searching")

        if index is None and args.index:
            print("loading pretrained index", args.index)
            index = faiss.read_index(args.index)
        elif index is None:
            raise RuntimeError("no index provided")

        run_search(args, ds, index, res)

    print("JSON results:", json.dumps(res.__dict__))


if __name__ == "__main__":
    main()
