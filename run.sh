#!/bin/bash

accelerate launch --multi_gpu --main_process_port `shuf -i 2900-65535 -n 1` run.py "$@"