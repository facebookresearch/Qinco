# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import faiss
import numpy as np
import torch

import datasets
from codec_qinco import decode, encode
from utils import (
    compute_fixed_codebooks,
    mean_squared_error,
    reconstruct_from_fixed_codebooks,
    refine_distances,
)


def main():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group("model")
    aa("--model", required=True, help="Model to load")
    aa("--float16", default=False, action="store_true", help="convert model to float16")

    group = parser.add_argument_group("database")
    aa(
        "--db",
        default="bigann1M",
        choices=datasets.available_names,
        help="Dataset to handle",
    )
    aa("--nt", default=500_000, type=int, help="nb training vectors")

    group = parser.add_argument_group("how to compute")
    aa("--bs", default=4096, type=int, help="batch size")
    aa("--device", default="cuda:0", help="pytorch device")
    aa("--nthreads", default=-1, type=int, help="number of OpenMP threads")

    args = parser.parse_args()

    print("args:", args)
    os.system(
        'echo -n "nb processors "; '
        "cat /proc/cpuinfo | grep ^processor | wc -l; "
        'cat /proc/cpuinfo | grep ^"model name" | tail -1'
    )
    os.system("nvidia-smi")

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

    ds = datasets.dataset_from_name(args.db)
    print(f"Prepared dataset {ds}")

    # to collect various stats
    res = argparse.Namespace()

    gt = ds.get_groundtruth()
    print(f"ground truth shape {gt.shape=:}")

    def compute_recalls(I):
        recalls = {}
        for rank in 1, 10, 100:
            recall = (I[:, :rank] == gt[:, :1]).sum() / gt.shape[0]
            recalls[rank] = float(recall)
        return recalls

    if args.nthreads != -1:
        print(f"set nb threads to {args.nthreads}")
        faiss.omp_set_num_threads(args.nthreads)

    xq = ds.get_queries()

    print(f"load training vectors")
    xt = ds.get_train(args.nt)
    print(f"xt shape {xt.shape=:}")

    print("encode trainset")
    xt_codes = encode(model, xt, bs=args.bs, is_float16=args.float16)

    print("decode")
    xt_decoded = decode(model, xt_codes, bs=args.bs, is_float16=args.float16)

    MSE = mean_squared_error(xt, xt_decoded)

    print(f"xt decoded {MSE=:g}")

    print("estimate fixed codebooks")

    codebooks = compute_fixed_codebooks(xt, xt_codes, k)

    MSE = mean_squared_error(xt, reconstruct_from_fixed_codebooks(xt_codes, codebooks))

    print(f"MSE fixed codebooks on trainset {MSE=:g}")

    print("Load database vectors")
    xb = ds.get_database()
    print(f"xb shape {xb.shape=:}")

    print("encode")
    xb_codes = encode(model, xb, bs=args.bs, is_float16=args.float16)

    print("Accurate decode")
    xb_decoded = decode(model, xb_codes, bs=args.bs, is_float16=args.float16)
    MSE = mean_squared_error(xb, xb_decoded)
    print(f"decode with full rerank {MSE=:.3f}")

    D, I = faiss.knn(xq, xb_decoded, 100)

    print("recalls", compute_recalls(I))

    print("AQ decode")
    xb_recons_fixed = reconstruct_from_fixed_codebooks(xb_codes, codebooks)
    MSE = mean_squared_error(xb, xb_recons_fixed)
    print(f"AQ decode {MSE=:.3f}")

    # do search on decoded vectors
    # (same result as search in IndexAdditiveQuantizer)
    D, I = faiss.knn(xq, xb_recons_fixed, 100)
    print("recalls", compute_recalls(I))

    # compute largest shortlist that we'll need
    _, Ishort_large = faiss.knn(xq, xb_recons_fixed, 1000)

    for kshort in 10, 20, 50, 100, 200, 500, 1000:
        assert kshort <= Ishort_large.shape[1]
        Ishort = Ishort_large[:, :kshort]

        # refine distances
        D_refined = refine_distances(xq, xb_decoded, Ishort)
        idx = np.argsort(D_refined, axis=1)
        I_refined = np.take_along_axis(Ishort, idx, axis=1)
        recalls = compute_recalls(I_refined)
        print(f"{kshort=:} {recalls=:}")


if __name__ == "__main__":
    main()
