#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

small_set=false
if [ "$1" == "-small" ]; then
  small_set=true
fi

mkdir -p data/bigann/gnd

# # Download queries
curl ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz | gunzip > data/bigann/bigann_query.bvecs

if [ "$small_set" = true ]; then
  echo "Only downloading the first 10M training vectors and the first 1M database vectors"
  curl https://dl.fbaipublicfiles.com/QINCo/datasets/bigann/bigann_learn_10M10k.bvecs --output data/bigann/bigann_learn.bvecs
  curl https://dl.fbaipublicfiles.com/QINCo/datasets/bigann/bigann1M.bvecs --output data/bigann/bigann_base.bvecs
  curl https://dl.fbaipublicfiles.com/QINCo/datasets/bigann/gnd/idx_1M.ivecs --output data/bigann/gnd/idx_1M.ivecs
else
  curl ftp://ftp.irisa.fr/local/texmex/corpus/bigann_learn.bvecs.gz | gunzip > data/bigann/bigann_learn.bvecs
  curl ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz | gunzip > data/bigann/bigann_base.bvecs
  curl ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz | gunzip > data/bigann/gnd/idx_1000M.ivecs
fi
