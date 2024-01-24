## Checkpoints for the experiments in the paper

Checkpoints (trained on 10M vectors) are available for various models presented in the paper (see links below). `8x8` denotes 8 bytes models and `16x8` denotes 16 bytes encoding. `L` indicates the number of residual blocks, i.e. the capapacity of the model (see Fig. 4 in the paper).
To use the checkpoints for encoding and decoding, the code assumes that they are stored in a `models/` subdirectory.

#### 8 bytes:
- bigann:	[bigann_8x8_L2](https://dl.fbaipublicfiles.com/QINCo/models/bigann_8x8_L2.pt)	[bigann_8x8_L4](https://dl.fbaipublicfiles.com/QINCo/models/bigann_8x8_L4.pt)	[bigann_8x8_L16](https://dl.fbaipublicfiles.com/QINCo/models/bigann_8x8_L16.pt)	
- deep:	[deep_8x8_L2](https://dl.fbaipublicfiles.com/QINCo/models/deep_8x8_L2.pt)	[deep_8x8_L4](https://dl.fbaipublicfiles.com/QINCo/models/deep_8x8_L4.pt)	[deep_8x8_L16](https://dl.fbaipublicfiles.com/QINCo/models/deep_8x8_L16.pt)		

- fb-ssnpp:	[FB_ssnpp_8x8_L2](https://dl.fbaipublicfiles.com/QINCo/models/FB_ssnpp_8x8_L2.pt)	[FB_ssnpp_8x8_L4](https://dl.fbaipublicfiles.com/QINCo/models/FB_ssnpp_8x8_L4.pt)		[FB_ssnpp_8x8_L16](https://dl.fbaipublicfiles.com/QINCo/models/FB_ssnpp_8x8_L16.pt)	
- Contriever :		[Contriever_8x8_L2](https://dl.fbaipublicfiles.com/QINCo/models/Contriever_8x8_L2.pt) [Contriever_8x8_L12](https://dl.fbaipublicfiles.com/QINCo/models/Contriever_8x8_L12.pt)	

#### 16 bytes:
- bigann: [bigann_16x8_L2](https://dl.fbaipublicfiles.com/QINCo/models/bigann_16x8_L2.pt)	[bigann_16x8_L4](https://dl.fbaipublicfiles.com/QINCo/models/bigann_16x8_L4.pt) [bigann_16x8_L16](https://dl.fbaipublicfiles.com/QINCo/models/bigann_16x8_L16.pt) 
- deep: [deep_16x8_L2](https://dl.fbaipublicfiles.com/QINCo/models/deep_16x8_L2.pt)	[deep_16x8_L4](https://dl.fbaipublicfiles.com/QINCo/models/deep_16x8_L4.pt) [deep_16x8_L16](https://dl.fbaipublicfiles.com/QINCo/models/deep_16x8_L16.pt) 
- fb-ssnpp: [FB_ssnpp_16x8_L2](https://dl.fbaipublicfiles.com/QINCo/models/FB_ssnpp_16x8_L2.pt)	[FB_ssnpp_16x8_L4](https://dl.fbaipublicfiles.com/QINCo/models/FB_ssnpp_16x8_L4.pt) [FB_ssnpp_16x8_L16](https://dl.fbaipublicfiles.com/QINCo/models/FB_ssnpp_16x8_L16.pt)
- Contriever: [Contriever_16x8_L12](https://dl.fbaipublicfiles.com/QINCo/models/Contriever_16x8_L12.pt)


The examples below assume there is a GPU supported by Pytorch present on the machine (see [installation](installation.md))
However, it is possible to run it on cpu as well by adding `--device cpu` to the command lines.
It also assumes that you have the dataset to be tested installed, see [downloading data][downloading_data.md].

## Encoding and decoding

To encode a set of vectors using one of the downloaded checkpoints (in this case bigann_8x8_L4.pt):

```
python codec_qinco.py \
    --model models/bigann_8x8_L4.pt --encode \
    --i data/bigann/bigann_query.bvecs \
    --o tmp/codes.npy
```

This outputs [this gist](https://gist.github.com/mdouze/7ceb9daf2053dceee01f6aae9cc763e9).

This example loads the queries of BigANN and encodes them to an array of codes.
The script can also take `.fvecs` or `.npy` files as input.

This also reports the MSE on the set of encoded vectors. 

To decode:

```
python codec_qinco.py \
    --model models/bigann_8x8_L4.pt --decode \
    --i tmp/codes.npy \
    --o tmp/decoded.npy
```

Which yields [this result](https://gist.github.com/mdouze/0124f97e0cdc6c1fde060880bc577a4b), where 
we also verify the MSE (note that we compute MSE here on the queries (for speeding up computation), rather than on the large database as done in the paper).


The compressed representation is still represented as an `int32` array so the
.npy file size is much larger than expected.
To get a file without overheads, add option `--raw` at encoding and decoding time.

For example: 
```
python codec_qinco.py \
    --model models/bigann_8x8_L4.pt --encode \
    --i data/bigann/bigann_query.bvecs \
    --o tmp/codes.raw --raw
```
produces a `codes.raw` file of exactly 80000 bytes (see [here](https://gist.github.com/mdouze/ce9f90faa824c59207b7558a8f6ecafb))

## Evaluating on one of the paper's datasets
```
python -u eval_qinco.py --db bigann1M --model models/bigann_8x8_L16.pt
```
This outputs [this text](https://gist.github.com/mdouze/f84ff34762117607d6983e33b89c7920),
which reproduces the entry for BigANN1M 8 bytes in Table 1 of the paper.
