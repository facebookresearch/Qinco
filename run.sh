# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash

accelerate launch --multi_gpu --main_process_port `shuf -i 2900-65535 -n 1` run.py "$@"