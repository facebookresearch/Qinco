# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import faiss
import torch

import datasets
from codec_qinco import decode, encode

####################################################################
# Driver
####################################################################


def main():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group("files")
    aa("--model", default="", help="QINCo model to use")

    group = parser.add_argument_group("computation options")
    aa("--batch_size", default=4096, type=int, help="batch size")
    aa("--device", default="cuda:0", help="pytorch device to use")
    aa("--float16", default=False, action="store_true", help="convert model to float16")

    group = parser.add_argument_group("database")
    aa(
        "--db",
        default="bigann1M",
        choices=datasets.available_names,
        help="Dataset to handle",
    )

    args = parser.parse_args()

    print("args:", args)

    print("loading model", args.model)
    model = torch.load(args.model, map_location=args.device)
    print("  database normalization factor", model.db_scale)
    model.eval()
    if args.float16:
        model.half()

    ds = datasets.dataset_from_name(args.db)
    print(f"Prepared dataset {ds}")

    xb = ds.get_database()
    print(f"Encoding database vectors of size {xb.shape}")
    codes = encode(model, xb, bs=args.batch_size, is_float16=args.float16)
    print("Decoding")
    xb_recons = decode(model, codes, bs=args.batch_size, is_float16=args.float16)
    print("Performing search")
    D, I = faiss.knn(ds.get_queries(), xb_recons, 100)

    gt = ds.get_groundtruth()
    for rank in 1, 10, 100:
        recall = (gt[:, :1] == I[:, :rank]).sum() / ds.nq
        print(f"1-recall@{rank}: {recall:.4f}")


if __name__ == "__main__":
    main()
