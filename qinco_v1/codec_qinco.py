# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time

import numpy as np
import torch

try:
    import faiss
    from faiss.contrib.vecs_io import bvecs_mmap, fvecs_mmap
except ImportError:
    print("Faiss missings, some functionality will not work")


####################################################################
# encoding
#####################################################################


def encode(model, x, bs, is_float16):
    output = []
    err_sum = 0
    device = next(model.parameters()).device
    t0 = time.time()
    with torch.no_grad():
        for i0 in range(0, len(x), bs):
            batch = x[i0 : i0 + bs]
            batch = torch.from_numpy(batch).to(device) / model.db_scale
            if is_float16:
                batch = batch.half()
            codes, recons = model.encode(batch)
            err_sum += ((recons - batch) ** 2).sum().item() * model.db_scale**2
            output.append(codes.cpu().numpy())
            print(
                f"[{time.time() - t0:.2f} s] {len(batch) + i0} / {len(x)}",
                end="\r",
                flush=True,
            )
    MSE = err_sum / len(x)
    print(f"Encoding done in {time.time() - t0:.2f} s, {MSE=:g}")
    return np.concatenate(output)


####################################################################
# decoding
####################################################################


def decode(model, codes, bs, is_float16):
    output = []
    device = next(model.parameters()).device
    t0 = time.time()
    with torch.no_grad():
        for i0 in range(0, len(codes), bs):
            batch = codes[i0 : i0 + bs]
            batch = torch.from_numpy(batch).to(device)
            if is_float16:
                batch = batch.half()
            recons = model.decode(batch) * model.db_scale
            output.append(recons.cpu().numpy())
            print(
                f"[{time.time() - t0:.2f} s] {len(batch) + i0} / {len(codes)}",
                end="\r",
                flush=True,
            )
    print(f"Decoding done in {time.time() - t0:.2f} s")
    return np.concatenate(output)


####################################################################
# Driver
####################################################################


def main():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group("what to do")
    aa("--encode", default=False, action="store_true")
    aa("--decode", default=False, action="store_true")

    group = parser.add_argument_group("files")
    aa("--model", default="", help="QINCo model to use")
    aa("--i", required=True, help="input vectors (npy or fvecs/bvecs format)")
    aa("--o", required=True, help="output format (npy or raw)")
    aa(
        "--raw",
        default=False,
        action="store_true",
        help="codes are in raw format (no header), to directly measure sizes",
    )

    group = parser.add_argument_group("computation options")
    aa("--batch_size", default=4096, type=int, help="batch size")
    aa("--device", default="cuda:0", help="pytorch device to use")
    aa("--float16", default=False, action="store_true", help="convert model to float16")

    args = parser.parse_args()

    print("args:", args)

    assert args.encode ^ args.decode, "one of encode or decode must be selected"

    print("loading model", args.model)
    model = torch.load(args.model, map_location=args.device)
    print("  database normalization factor", model.db_scale)
    model.eval()
    if args.float16:
        model.half()

    if args.encode:
        print("reading", args.i)
        if args.i.endswith(".npy"):
            x = np.load(args.i).astype(np.float32)
        elif args.i.endswith(".bvecs"):
            x = np.array(bvecs_mmap(args.i), dtype="float32")
        elif args.i.endswith(".fvecs"):
            x = np.array(fvecs_mmap(args.i), dtype="float32")
        else:
            raise RuntimeError("unrecognized format")
        print(f"encoding intput vectors of size {x.shape}")
        codes = encode(model, x, bs=args.batch_size, is_float16=args.float16)
        if not args.raw:
            print(f"Storing result of size {codes.shape} in {args.o}")
            np.save(args.o, codes)
        else:
            nbits = int(np.ceil(np.log2(model.K)))
            print(f"Packing result of size {codes.shape} to {model.M} * {nbits} bits")
            packed_codes = faiss.pack_bitstrings(codes, nbits)
            print(f"Packed size {packed_codes.shape} bytes, storing in {args.o}")
            packed_codes.tofile(args.o)

    if args.decode:
        print("reading", args.i)
        if args.raw:
            packed_codes = np.fromfile(args.i, dtype="uint8")
            nbits = int(np.ceil(np.log2(model.K)))
            code_size = (nbits * model.M + 7) // 8
            print(f"inferred {code_size=:} from {model.M} * {nbits} bits")
            packed_codes = packed_codes.reshape(-1, code_size)
            codes = faiss.unpack_bitstrings(packed_codes, [nbits] * model.M)
        else:
            if args.i.endswith(".npy"):
                codes = np.load(args.i)
            else:
                raise RuntimeError("unrecognized format")
        print(f"Decoding intput codes of size {codes.shape}")
        y = decode(model, codes, bs=args.batch_size, is_float16=args.float16)
        print(f"Storing result of size {y.shape} in {args.o}")
        np.save(args.o, y)


if __name__ == "__main__":
    main()
