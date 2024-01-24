# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import faiss
import numpy as np
import torch

import datasets
import model_qinco
import train_qinco

####################################################################
# Driver
####################################################################


def main():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group("what to do")
    aa(
        "--prepare",
        default=False,
        action="store_true",
        help="Prepare the training set and possibly train OPQ",
    )
    aa(
        "--recombine",
        default=False,
        action="store_true",
        help="recombine the QINCo models (and OPQ) into one PQ_QINCo model",
    )
    aa("--opq", default=False, action="store_true", help="Also train OPQ")
    aa("--OPQMatrix", default="", help="Location to store or load OPQ matrix")

    group = parser.add_argument_group("database")
    aa(
        "--db",
        default="bigann1M",
        choices=datasets.available_names,
        help="Dataset to handle",
    )
    aa("--training_data", default="", help="flat npy array with training vectors")
    aa(
        "--nt",
        default=510_000,
        type=int,
        help="nb training vectors to use. Defaults to 510k "
        "such that 10k can be used as the validation set later.",
    )

    group = parser.add_argument_group("for prepare")
    aa(
        "--training_subvectors",
        default=[],
        nargs="+",
        help="where to store the training vectors",
    )

    group = parser.add_argument_group("model parameters")
    aa("--nsub", default=4, type=int, help="number of sub-vectors")
    aa(
        "--opq_dim",
        default=-1,
        type=int,
        help="OPQ intermediate dim, if not provided, it is set to the data dimension",
    )

    group = parser.add_argument_group("for recombine")
    aa("--in_models", default=[], nargs="+", help="Trained QINCo models to be combined")
    aa("--out_model", default="", help="output PQ-QINCo model")

    args = parser.parse_args()

    print("args:", args)

    if args.prepare:
        assert args.nsub == len(args.training_subvectors)

        if args.training_data:
            print("Loading training data from", args.training_data)
            xt = np.load(args.training_data, mmap_mode="r")
            if len(xt) >= args.nt:
                print(f"   Size {xt.shape} -> restrict to {args.nt}")
                xt = np.array(xt[: args.nt])
            else:
                raise RuntimeError("not enough training data")
        else:
            print(f"Loading dataset {args.db}")
            ds = datasets.dataset_from_name(args.db)
            print(f"   {ds}")
            xt = ds.get_train(maxtrain=args.nt)

        print(f"Training set: {xt.shape}")
        d = xt.shape[1]
        if args.opq:
            print("training OPQ")
            opq_dim = args.opq_dim if args.opq_dim > 0 else d
            opq = faiss.OPQMatrix(d=d, M=args.nsub, d2=opq_dim)
            opq.train(xt)
            mat = faiss.vector_to_array(opq.A).reshape(opq.d_out, opq.d_in)
            xt_ref = opq.apply_py(xt[:100])
            xt_new = xt[:100] @ mat.T
            np.testing.assert_allclose(xt_ref, xt_new)
            xt = opq.apply_py(xt)
            np.save(args.OPQMatrix, mat)

        assert (
            d % args.nsub == 0
        ), f"The number of blocks must be a divisor of the data dimension. {d} is not divisible by {args.nsub}"
        dsub = d // args.nsub
        for PQblock in range(args.nsub):
            fname = args.training_subvectors[PQblock]
            print("writing training vectors", fname)
            np.save(fname, xt[:, (PQblock * dsub) : ((PQblock + 1) * dsub)])

    elif args.recombine:
        sub_quantizers = []
        for sub_model in args.in_models:
            sub_quantizers.append(torch.load(sub_model, map_location="cpu"))

        if args.opq:
            opq_matrix = np.load(args.OPQMatrix)
            model = model_qinco.PQ_QINCo(sub_quantizers, opq_matrix=opq_matrix)
        else:
            model = model_qinco.PQ_QINCo(sub_quantizers)

        print("storing PQ-QINCo model ", args.out_model)
        torch.save(model, args.out_model)

    else:
        assert not args.ivf, "Combining PQ-QINCo with IVF-QINCo is not yet supported"
        train_qinco.main(args)


if __name__ == "__main__":
    main()
