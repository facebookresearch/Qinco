#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
set -e

small_set=false
if [ "$1" == "-small" ]; then
  small_set=true
fi

mkdir -p data/deep1b

# Download queries
# these links are broken, will use the bigann site ones

curl https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin \
    --output data/deep1b/query.public.10K.fbin
python data/deep1b/fbin_to_fvecs.py data/deep1b/query.public.10K.fbin data/deep1b/deep1B_queries.fvecs
rm data/deep1b/query.public.10K.fbin

# curl https://disk.yandex.ru/d/11eDCm7Dsn9GA/deep1B_queries.fvecs --output data/deep1b/deep1B_queries.fvecs

if [ "$small_set" = true ]; then
  echo "Only downloading the first 10M training vectors and the first 1M database vectors"
  curl https://dl.fbaipublicfiles.com/QINCo/datasets/deep1b/learn10M10k.fvecs --output data/deep1b/learn.fvecs
  curl https://dl.fbaipublicfiles.com/QINCo/datasets/deep1b/base1M.fvecs --output data/deep1b/base.fvecs
  curl https://dl.fbaipublicfiles.com/QINCo/datasets/deep1b/deep1M_groundtruth.ivecs --output data/deep1b/deep1M_groundtruth.ivecs

else

  # Download ground truths
  curl https://dl.fbaipublicfiles.com/QINCo/datasets/deep1b/deep1B_groundtruth.ivecs \
    --output data/deep1b/deep1B_groundtruth.ivecs

  curl https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/learn.350M.fbin \
    --output data/deep1b/learn.350M.fbin

  python data/deep1b/fbin_to_fvecs.py data/deep1b/learn.350M.fbin data/deep1b/learn.fvecs
  rm data/deep1b/learn.350M.fbin

  curl https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin \
    --output data/deep1b/base.1B.fbin

  python data/deep1b/fbin_to_fvecs.py data/deep1b/base.1B.fbin data/deep1b/base.fvecs
  rm data/deep1b/base.1B.fbin

fi
