# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

import numpy as np

infile, outfile = sys.argv[1:]


def xbin_mmap(fname, dtype, maxn=-1):
    """mmap the competition file format for a given type of items"""
    n, d = map(int, np.fromfile(fname, dtype="uint32", count=2))
    assert os.stat(fname).st_size == 8 + n * d * np.dtype(dtype).itemsize
    if maxn > 0:
        n = min(n, maxn)
    return np.memmap(fname, dtype=dtype, mode="r", offset=8, shape=(n, d))


print("mmapping", infile)

M = xbin_mmap(infile, "float32")

print(f"   fbin of size", M.shape)

bs = 8192

print(f"writing {outfile} by batches of size {bs}")

with open(outfile, "wb") as f:
    for i0 in range(0, len(M), bs):
        block = M[i0 : i0 + bs]
        sizes = np.ones(len(block), dtype="int32")
        sizes[:] = block.shape[1]
        block1 = np.hstack((sizes[:, None].view("float32"), block))
        block1.tofile(f)
        print(f"{i0} / {len(M)}", end="\r", flush=True)
