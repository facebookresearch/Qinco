# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -e


function run_on ()
{
    sys="$1"
    shift
    name="$1"
    shift
    script="$logdir/$name.sh"

    if [ -e "$script" ]; then
        echo script "$script" exists
        return
    fi

    # srun handles special characters fine, but the shell interpreter
    # does not
    escaped_cmd=$( printf "%q " "$@" )

    cat > $script <<EOF
#! /bin/bash
srun $escaped_cmd
EOF

    echo -n "$logdir/$name.stdout "
    sbatch -n1 -J "$name" \
           $sys \
            --comment='priority is the only one that works'  \
           --output="$logdir/$name.stdout" \
           "$script"

}

logdir=/checkpoint/matthijs/QINCo/pq_training/logs

mkdir -p $logdir


function run_on_4gpu {
    run_on "--cpus-per-task=80 --gres=gpu:4  --mem=240G --time=70:00:00 --constraint=volta --partition=learnlab" "$@"
}

function run_on_1gpu {
    run_on "--cpus-per-task=10 --gres=gpu:1  --mem=80G --time=70:00:00 --constraint=volta --partition=learnlab" "$@"
}


###############################
# To enable / disable some experiments
###############################

function SKIP () {
    echo -n
}

###############################
# First wave
###############################


for i in {0..1}; do
    SKIP run_on_4gpu QTPQ.$i.b \
     python -u train_qinco.py \
        --training_data models/pq_training/trainset_$i.npy \
        --M 3 --L 1 --h 256 --lr 0.001 --nt 510_000 \
        --max_epochs 50 \
        --ngpu 4 --model models/pq_training/quick_sub_model_$i.pt
done


for i in {0..1}; do
    SKIP run_on_4gpu TPQ.$i.b \
     python -u train_qinco.py \
        --training_data models/pq_training/trainset_$i.npy \
        --M 16 --L 2 --h 256 --lr 0.0001 \
        --ngpu 4 --model models/pq_training/sub_model_$i.pt
done

###############################
# Systematic
###############################

# run prepare locally
pq_data_dir=/checkpoint/matthijs/QINCo/pq_training

for nsub in 16 8 4 2; do
    train_vecs="$( seq --format=$pq_data_dir/trainset_nsub${nsub}_%g.npy 0 $((nsub-1)) )"
    SKIP python pq_qinco.py   \
      --prepare --db FB_ssnpp1M \
      --nt 10_010_000 --nsub $nsub \
      --training_subvectors $train_vecs
done


for nsub in 16 8 4 2; do
    M=$((32/nsub))
    for((i=0;i<nsub;i++)); do
        SKIP run_on_4gpu TPQ.nsub$nsub.$i.c \
         python -u train_qinco.py \
            --training_data $pq_data_dir/trainset_nsub${nsub}_$i.npy \
            --M $M --L 2 --h 256 --lr 0.0001 --nt 10000000 --nval 10000 \
            --ngpu 4 --model $pq_data_dir/sub_model_nsub${nsub}_${i}_M${M}_L2.pt
    done

done

# for nsub in 16 8 4 2; do
for nsub in 2 ; do
    M=$((32/nsub))
    models="$( seq --format=$pq_data_dir/sub_model_nsub${nsub}_%g_M${M}_L2.pt 0 $((nsub-1)) )"
    python pq_qinco.py   \
      --recombine  --nsub $nsub \
      --in_models $models \
      --out_model $pq_data_dir/model_${nsub}x${M}x8.pt
done

# run evaluation

for nsub in 2; do
# for nsub in 16 8 4 2; do
    M=$((32/nsub))
    run_on_1gpu EPQ.nsub$nsub.a \
         python -u eval_qinco.py \
            --db FB_ssnpp1M \
            --model $pq_data_dir/model_${nsub}x${M}x8.pt
done



###############################
# With OPQ
###############################


for nsub in 4 2 ; do
    train_vecs="$( seq --format=$pq_data_dir/trainset_OPQ_nsub${nsub}_%g.npy 0 $((nsub-1)) )"
    SKIP python pq_qinco.py   \
      --prepare --db FB_ssnpp1M \
      --opq --OPQMatrix $pq_data_dir/opq_matrix_nsub${nsub}.npy \
      --nt 10_010_000 --nsub $nsub \
      --training_subvectors $train_vecs
done

for nsub in 4 2 ; do
    M=$((32/nsub))
    for((i=0;i<nsub;i++)); do
        run_on_4gpu TOPQ.nsub$nsub.$i.a \
        SKIP python -u train_qinco.py \
            --training_data $pq_data_dir/trainset_OPQ_nsub${nsub}_$i.npy \
            --M $M --L 2 --h 256 --lr 0.0001 --nt 10000000 --nval 10000 \
            --ngpu 4 --model $pq_data_dir/sub_model_OPQ_nsub${nsub}_${i}_M${M}_L2.pt
    done

done

# for nsub in 16 8 4 2 ; do
for nsub in 4 ; do
    M=$((32/nsub))
    models="$( seq --format=$pq_data_dir/sub_model_OPQ_nsub${nsub}_%g_M${M}_L2.pt 0 $((nsub-1)) )"
    SKIP python pq_qinco.py   \
      --recombine  --nsub $nsub \
      --opq --OPQMatrix $pq_data_dir/opq_matrix_nsub${nsub}.npy \
      --in_models $models \
      --out_model $pq_data_dir/model_OPQ_${nsub}x${M}x8.pt
done

#  run evaluation

# for nsub in 16 8 4 2; do
for nsub in 4; do
    M=$((32/nsub))
    SKIP run_on_1gpu EOPQ.nsub$nsub.d \
         python -u eval_qinco.py \
            --db FB_ssnpp1M \
            --model $pq_data_dir/model_OPQ_${nsub}x${M}x8.pt
done