
## Pretrained IVF models 

Pretrained IVF models are available at 

|    | Bigann KIVF=65k | Bigann KIVF=1M | Deep KIVF=65k | Deep KIVF=1M |
| ------ | ------------- | ------------- | ------------- | ------------- | 
| M=8 L=2 | [IVF65k_8x8_L2](https://dl.fbaipublicfiles.com/QINCo/models/bigann_IVF65k_8x8_L2.pt) | [IVF1M_8x8_L2](https://dl.fbaipublicfiles.com/QINCo/models/bigann_IVF1M_8x8_L2.pt) | [IVF65k_8x8_L2](https://dl.fbaipublicfiles.com/QINCo/models/deep_IVF65k_8x8_L2.pt) | [IVF1M_8x8_L2](https://dl.fbaipublicfiles.com/QINCo/models/deep_IVF1M_8x8_L2.pt) | 
| M=8 L=4 | [IVF65k_8x8_L4](https://dl.fbaipublicfiles.com/QINCo/models/bigann_IVF65k_8x8_L4.pt) | [IVF1M_8x8_L4](https://dl.fbaipublicfiles.com/QINCo/models/bigann_IVF1M_8x8_L4.pt) | [IVF65k_8x8_L4](https://dl.fbaipublicfiles.com/QINCo/models/deep_IVF65k_8x8_L4.pt) | [IVF1M_8x8_L4](https://dl.fbaipublicfiles.com/QINCo/models/deep_IVF1M_8x8_L4.pt) | 
| M=16 L=2 |  [IVF65k_16x8_L2](https://dl.fbaipublicfiles.com/QINCo/models/bigann_IVF65k_16x8_L2.pt) | [IVF1M_16x8_L2](https://dl.fbaipublicfiles.com/QINCo/models/bigann_IVF1M_16x8_L2.pt) | [IVF65k_16x8_L2](https://dl.fbaipublicfiles.com/QINCo/models/deep_IVF65k_16x8_L2.pt) | [IVF1M_16x8_L2](https://dl.fbaipublicfiles.com/QINCo/models/deep_IVF1M_16x8_L2.pt) | 
| M=16 L=4 | [IVF65k_16x8_L4](https://dl.fbaipublicfiles.com/QINCo/models/bigann_IVF65k_16x8_L4.pt) | [IVF1M_16x8_L4](https://dl.fbaipublicfiles.com/QINCo/models/bigann_IVF1M_16x8_L4.pt) | [IVF65k_16x8_L4](https://dl.fbaipublicfiles.com/QINCo/models/deep_IVF65k_16x8_L4.pt) | [IVF1M_16x8_L4](https://dl.fbaipublicfiles.com/QINCo/models/deep_IVF1M_16x8_L4.pt)	 |


Training an IVF model requires precomputed IVF centroids that can be obtained `search_ivf.py` with: 

```
python search_ivf.py \
   --todo train_centroids \
   --n_centroids 1048576 \
   --nt 50_000_000 \
   --IVF_centroids models/bigann_centroids_1M_repro.npy
```
This trains in about 30 min on [8 GPUs](https://gist.github.com/mdouze/4e158cd56794730ef0c1fca669c2b35e).

Precomputed centroids can also be downloaded at: 
[bigann_centroids_65k.npy](https://dl.fbaipublicfiles.com/QINCo/ivf/bigann_centroids_65k.npy)
[bigann_centroids_1M.npy](https://dl.fbaipublicfiles.com/QINCo/ivf/bigann_centroids_1M.npy)
[deep_centroids_65k.npy](https://dl.fbaipublicfiles.com/QINCo/ivf/deep_centroids_65k.npy)
[deep_centroids_1M.npy](https://dl.fbaipublicfiles.com/QINCo/ivf/deep_centroids_1M.npy).


## Small-scale example

For warmup we train a 10M index. 

### Training 

Then the IVFQINCo model is trained with: 
```
python train_qinco.py \
    --ivf --db bigann1M \
    --nt 500_000 --M 8 --L 2 --lr 0.001 \
    --IVF_centroids models/bigann_centroids_65k.npy \
    --ngpu 8  --model models/test_ivf_train.pt
```
Log [here](https://gist.github.com/mdouze/f9e61d9a042216a7f66d02710d85f415).
The reported validation loss is close to the inline table in section 5.3 of the paper. 

### Building an index 

Given a pretrained IVF model, building an IVFQINCo index can be done with: 

```
python search_ivf.py \
     --db bigann10M \
     --model models/bigann_IVF65k_16x8_L2.pt \
     --index index/bigann10M_IVF65k_16x8_L2.faissindex \
     --index_key IVF65536,RQ16x8_Nqint8
```
This trains an AQ approximation for the IVF-QINCo model and incorprates them in an index of type `IVF65536,RQ16x8_Nqint8` (which means an IVF index of size 65k with codes of 8x8 bits and a normalization of size int8). 
The whole process takes about 1 hour.
The 10M first vectors of the BigANN dataset are added to it and the index is stored in `index/bigann10M_IVF65k_16x8_L2.faissindex` (can also be downloaded [here](https://dl.fbaipublicfiles.com/QINCo/ivf/bigann10M_IVF65k_16x8_L2.faissindex))
The typical output is [this](https://gist.github.com/mdouze/6fa78d8a2de7714041cf29a9c7da7c82).

### Searching an index

The following code performs a search in the index with varying search parameters: 
```
python search_ivf.py \
      --db bigann10M \
      --model models/bigann_IVF65k_16x8_L2.pt \
      --index index/bigann10M_IVF65k_16x8_L2.faissindex \
      --todo search --nthread 32
```
Which gives [this output](https://gist.github.com/mdouze/e4b7c9dbf6a52e0f7cf100ce0096aaa8).

## Experiments from the paper 

The 1B-scale experiments from the paper (eg. the one for M=8, L=2) can be run with 

Building the index (on GPU): 
```
python ivf_search.py  \
            --todo train add \
            --db bigann1B \
            --quantizer_efConstruction 200 \
            --index_key IVF1048576_HNSW32,RQ8x8_Nqint8 \
            --nt 1000_000 
            --index index/bigann1B_IVF1M_8x8_L2.faissindex \
            --bs 512
```
The `IVF1048576_HNSW32,RQ8x8_Nqint8` means that, in addition to the options seen above, the IVF index uses an HNSW coarse quantizer: approximate but faster). 

Since this is a slow process, it is more convenient to use the `--xb_codes` option that consumes pre-assigned codes (that can be produced with `codec_qinco.py`). 
Then the process is easy to parallelize.

Searching:

```
python search_ivf.py \
      --todo search \
      --db bigann1B \
      --model simple/models/bigann_IVF1M_8x8_L2.pt \
      --index index/bigann1B_IVF1M_8x8_L2.faissindex \
      --nthread 32 \
      --nshort 10 20 50 100 200 500 1000 2000 \
      --nprobe 4 8 16 32 64 128 256 512 1024 2048 4096 \
      --quantizer_efSearch 4 8 16 32 64 128 256 512 1024 2048 4096
```



Which yields [this result](https://gist.github.com/mdouze/0187c2ca3f96f806e41567af13f80442) that reproduces figure 3 in the paper.

Pre-built 1B indexes with K_IVF=1M can be found here: 

|    | Bigann1B, L=2 | Bigann1B, L=4 | Deep1B, L=2 | Deep1B, L=4 |
| ------ | ------------- | ------------- | ------------- | ------------- | 
| M=8 | [bigann1B_IVF1M_8x8_L2.faissindex](https://dl.fbaipublicfiles.com/QINCo/ivf/bigann1B_IVF1M_8x8_L2.faissindex) | [bigann1B_IVF1M_8x8_L4.faissindex](https://dl.fbaipublicfiles.com/QINCo/ivf/bigann1B_IVF1M_8x8_L4.faissindex) | [deep1B_IVF1M_8x8_L2.faissindex](https://dl.fbaipublicfiles.com/QINCo/ivf/deep1B_IVF1M_8x8_L2.faissindex) | [deep1B_IVF1M_8x8_L4.faissindex](https://dl.fbaipublicfiles.com/QINCo/ivf/deep1B_IVF1M_8x8_L4.faissindex) |
| M=16 | [bigann1B_IVF1M_16x8_L2.faissindex](https://dl.fbaipublicfiles.com/QINCo/ivf/bigann1B_IVF1M_16x8_L2.faissindex) | [bigann1B_IVF1M_16x8_L4.faissindex](https://dl.fbaipublicfiles.com/QINCo/ivf/bigann1B_IVF1M_16x8_L4.faissindex) | [deep1B_IVF1M_16x8_L2.faissindex](https://dl.fbaipublicfiles.com/QINCo/ivf/deep1B_IVF1M_16x8_L2.faissindex) | [deep1B_IVF1M_16x8_L4.faissindex](https://dl.fbaipublicfiles.com/QINCo/ivf/deep1B_IVF1M_16x8_L4.faissindex) |
| M=32 | [bigann1B_IVF1M_32x8_L2.faissindex](https://dl.fbaipublicfiles.com/QINCo/ivf/bigann1B_IVF1M_32x8_L2.faissindex) | [bigann1B_IVF1M_32x8_L4.faissindex](https://dl.fbaipublicfiles.com/QINCo/ivf/bigann1B_IVF1M_32x8_L4.faissindex) | [deep1B_IVF1M_32x8_L2.faissindex](https://dl.fbaipublicfiles.com/QINCo/ivf/deep1B_IVF1M_32x8_L2.faissindex) | [deep1B_IVF1M_32x8_L4.faissindex](https://dl.fbaipublicfiles.com/QINCo/ivf/deep1B_IVF1M_32x8_L4.faissindex) |
