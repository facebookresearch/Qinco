
Training can be done either on a Faiss dataset or on a flat array of training
vectors (using the --training_data flag).
See train_qinco.py for docs about the flags that can be used.
If you want to run QINCo-LR, i.e. with a low-rank projection in the concatenation block, add the --lowrank flag.

To run a QINCo training on 500k training vectors of BigANN, on 4 GPUs, run:
```
python -u train_qinco.py \
    --db bigann1B \
    --M 8 --L 2 --h 256 --lr 0.001 \
    --ngpu 4 --model models/test_model.pt
```
This trains an M=8 bytes QINCo model (L=2) on 4 GPUs and a default of 500k training vectors (for which a base learning rate of 1e-3 is best). 
Each iteration takes ~20s, for a total learning time of 1h. 
The output looks like [this](https://gist.github.com/mdouze/c85f69f7ac997cdc9b9096e3640e0423).

Another example: Training a model on 10M training vectors of deep1M, with settings as reported in Table 1:
```
python -u train_qinco.py \
    --db deep1M \
    --nt 10_000_000 \
    --M 8 --L 16 --h 256 --lr 0.0001 \
    --ngpu 4 --model models/deep1M_8x8_L16.pt
```

Another example for quickly running a small model on cpu: Training an 8 bytes QINCo L=2 model on only 100k training vectors of BigANN.
Running this model for 3 epochs (+-30min on cpu), already beats the MSE of 2.49e4 from RQ (beam size = 5), as reported in Table 1.

```
python -u train_qinco.py \
    --db bigann1M \
    --nt 100_000 \
    --M 8 --L 2 --h 256 --lr 0.001 \
    --ngpu 0 --model models/bigann_T100k_8x8_L2.pt
```
