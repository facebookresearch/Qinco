
PQ just duplicates the QINCo training and quantization over subvectors. 
This is driven by the `pq_qinco.py` script. 

These docs provide an example for training and evaluating the 2x16x8 PQ-QINCo model as presented in the paper.

### Training 

This is done in three steps: 

1. split training set in several subvectors 

```
python pq_qinco.py \
    --prepare --db FB_ssnpp1M \
    --nt 10_010_000 \
    --nsub 2 \
    --training_subvectors models/pq_training/trainset_{0..1}.npy
```

This will generate 2 training sets (including 10k validation vectors) for the independent PQ trainings and store them in `models/pq_training/trainset_{0..1}.npy`.

2. train the QINCo's independently. The following trains 2 16x8 QINCo models, each on the corresponding subvectors.
```
for i in {0..1}; do 
    python train_qinco.py \
        --training_data models/pq_training/trainset_$i.npy \
        --M 16 --L 2 --h 256 --lr 0.0001 \
        --nt 10_000_000 --nval 10_000 \
        --ngpu 4 --model models/pq_training/sub_model_$i.pt
done
```
Of course these 2 training steps can be done in parallel.

3. recombine the 2 QINCo models into one 2x16x8 PQ-QINCo model. 

```
python pq_qinco.py \
    --recombine \
    --nsub 2 \
    --in_models models/pq_training/sub_model_{0..1}.pt \
    --out_model models/pq_qinco_2x16x8.pt
```


The model `pq_qinco_2x16x8.pt` can be used with `codec_qinco.py` or `eval_qinco.py`, see these [docs](checkpoints.md)

### Testing 

```
python -u eval_qinco.py \
            --db FB_ssnpp1M \
            --model models/pq_qinco_2x16x8.pt
```

which outputs 
```
loading model /checkpoint/matthijs/QINCo/pq_training/model_2x16x8.pt
  database normalization factor 1
Prepared dataset dataset in dimension 256, with metric L2, size: Q 10000 B 1000000 T 10000000
Encoding database vectors of size (1000000, 256)
Encoding done in 349.60 s, MSE=42072.4
Decoding
Decoding done in 3.28 s
Performing search
1-recall@1: 0.1227
1-recall@10: 0.3204
1-recall@100: 0.6224
```
which corresponds to 2x16x8 point of Fig 6 in the paper. 

See the other combination [here](https://gist.github.com/mdouze/5b1bba81b3fb8233546c5a97a1daaa3f). 
They were obtained via [run_train_pq.bash](https://gist.github.com/mdouze/1a63f91afe20243a2f81d370686593b0) that uses slurm to do the whole process on all the settings. 

## OPQ transformation

OPQ is a rotation of input vectors that makes it more suitable for further PQ encoding by makeing subsequent sub-vectors more independent. 

It can be applied to PQ-QINCo as well with a flag to the `pq_qinco.py`: 
```
python pq_qinco.py \
    --prepare --db FB_ssnpp1M \
    --nt 10_010_000 \
    --nsub 2 \
    --opq --OPQMatrix models/pq_training/OPQ.npy \
    --training_subvectors models/pq_training/trainset_{0..1}.npy
```

Adding the `--opq` flag will also train and store the OPQ rotation matrix in the numpy file provided via `--OPQMatrix`, e.g. `--OPQ_matrix models/pq_training/OPQ.npy`.

The PQ training is the same. There is no need to provide a flag here, as the stored data has already been OPQ rotated.

The recombination needs to get the OPQ and OPQ matrix flags: 
```
python pq_qinco.py \
    --recombine \
    --nsub 2 \
    --opq --OPQMatrix models/pq_training/OPQ.npy \
    --in_models models/pq_training/sub_model_{0..1}.pt \
    --out_model models/pq_qinco_2x16x8.pt
```

The evaluation is the same. 
In the case of the FB_ssnpp1M dataset, the gain in MSE is minimal: 

| combination   | regular PQ    |  with OPQ |
| ------------- | ------------- |-----------|
| 16x2x8        | 41409.2       | 41313.0   |
| 8x4x8         | 42493.5       | 42335.3   | 
| 4x8x8         | 42805.3       | 42523.9   | 
| 2x16x8        | 42072.4       | 41912.5   |

However, OPQ would be more useful for vectors with more unbalanced components.
