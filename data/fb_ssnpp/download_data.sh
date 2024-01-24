#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

curl https://dl.fbaipublicfiles.com/QINCo/datasets/FB_ssnpp/database1M.npy --output data/fb_ssnpp/database1M.npy
curl https://dl.fbaipublicfiles.com/QINCo/datasets/FB_ssnpp/queries.npy --output data/fb_ssnpp/queries.npy
curl https://dl.fbaipublicfiles.com/QINCo/datasets/FB_ssnpp/ground_truth1M.npy --output data/fb_ssnpp/ground_truth1M.npy
curl https://dl.fbaipublicfiles.com/QINCo/datasets/FB_ssnpp/training_set10010k.npy --output data/fb_ssnpp/training_set10010k.npy
