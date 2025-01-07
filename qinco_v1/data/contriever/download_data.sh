#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

curl https://dl.fbaipublicfiles.com/QINCo/datasets/ContrieverEmb/database1M.npy --output data/contriever/database1M.npy
curl https://dl.fbaipublicfiles.com/QINCo/datasets/ContrieverEmb/queries.npy --output data/contriever/queries.npy
curl https://dl.fbaipublicfiles.com/QINCo/datasets/ContrieverEmb/ground_truth1M.npy --output data/contriever/ground_truth1M.npy
curl https://dl.fbaipublicfiles.com/QINCo/datasets/ContrieverEmb/training_set.npy --output data/contriever/training_set.npy
