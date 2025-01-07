# Vector Compression and Search with Improved Implicit Neural Codebooks (QINCo2)

*This repository has been updated with the code from QINCo2. To access the original QINCo1 code, see [qinco_v1 directory](qinco_v1/README.md).*

This code repository corresponds to the paper [QINCo2: Vector Compression and Search with Improved Implicit Neural Codebooks](https://arxiv.org/abs/2501.03078), introducing an improved quantization process over QINCo.
We also include code reproducing the ICML'24 paper [Residual Quantization with Implicit Neural Codebooks](https://arxiv.org/pdf/2401.14732.pdf) ([qinco_v1 directory](qinco_v1/README.md)), in which Quantization with Implicit Neural Codebooks (QINCo) was proposed. Please read both papers to learn about QINCo and QINCo2.

QINCo is a neurally-augmented algorithm for multi-codebook vector quantization, specifically residual quantization (RQ). Instead of using a fixed codebook per quantization step, QINCo uses a neural network to predict a codebook for the next quantization step, conditioned upon the quantized vector so far. In other words, the codebooks to be used depend on the Voronoi cells selected previously. This greatly enhances the capacity of the compression system, without the need to store more codebook vectors explicitly. An additional advantage of QINCo is its modularity. Thanks to training each quantization step with its own quantization error, the trained system for a certain compression rate, can also be exploited for lower compression rates, making QINCo a dynamic rate quantizer.

QINCo2 introduces several key novelties:
- A fast approximate encoding method, yielding similar MSE for a much faster training and encoding time.
- Integration of beam search to the encoding process, reaching much lower compression errors than QINCo1 for a similar encoding time when combined to approximate encoding.
- A new (optional) module to the large-scale retrieval pipeline, improving accuracy using a pairwise decoder.
- An overall upgrade of the architecture and training process.

## Citation

If you use QINCo in a research work please cite our paper:

```
@misc{vallaeys2025qinco2,
    title={Qinco2: Vector Compression and Search with Improved Implicit Neural Codebooks},
    author={Théophane Vallaeys and Matthew Muckley and Jakob Verbeek and Matthijs Douze},
    year={2025},
    eprint={2501.03078},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```


## Setup

### Installation

QINCo2 requires python3 and packages from the `requirements.txt` file. They can be installed using conda:

```bash
git clone https://github.com/facebookresearch/Qinco2
cd Qinco2
conda env create -f environment.yml
```

### Downloading the data

Download the datasets used in the paper by running the corresponding bash scripts:

#### BigANN 
```
./data/bigann/download_data.sh
```
or if you have limited storage space, you can also download only the first 1M database vectors, and the first 10M training vectors using:
```
./data/bigann/download_data.sh -small
```

#### Deep1B  
```
./data/deep1b/download_data.sh
```
or if you have limited storage space, you can also download only the first 1M database vectors, and the first 10M training vectors using:
```
./data/deep1b/download_data.sh -small
```

#### Contriever
```
./data/contriever/download_data.sh
```

#### FB-ssnpp
```
./data/fb_ssnpp/download_data.sh
```

## Pretrained checkpoints

### Base experiments checkpoints

Below are the checkpoints for the QINCo1 and QINCo2-L models, trained on all four datasets. Instructions below show how to use and evaluate them. The commands suppose that the files are stored inside the `models/` directory.

|  | **BigANN1M** | **Deep1M** | **Contriever1M** | **FB-ssnpp1M** |
|---|---|---|---|---|
| **Qinco1 (8 bytes)** | [qinco1-bigann1M-8x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/qinco1-bigann1M-8x8.pt) | [qinco1-deep1M-8x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/qinco1-deep1M-8x8.pt) | [qinco1-contriever1M-8x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/qinco1-contriever1M-8x8.pt) | [qinco1-FB_ssnpp1M-8x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/qinco1-FB_ssnpp1M-8x8.pt) |
| **Qinco1 (16 bytes)** | [qinco1-bigann1M-16x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/qinco1-bigann1M-16x8.pt) | [qinco1-deep1M-16x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/qinco1-deep1M-16x8.pt) | [qinco1-contriever1M-16x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/qinco1-contriever1M-16x8.pt) | [qinco1-FB_ssnpp1M-16x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/qinco1-FB_ssnpp1M-16x8.pt) |
| **Qinco2-L (8 bytes)** | [qinco2_L-bigann1M-8x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/qinco2_L-bigann1M-8x8.pt) | [qinco2_L-deep1M-8x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/qinco2_L-deep1M-8x8.pt) | [qinco2_L-contriever1M-8x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/qinco2_L-contriever1M-8x8.pt) | [qinco2_L-FB_ssnpp1M-8x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/qinco2_L-FB_ssnpp1M-8x8.pt) |
| **Qinco2-L (16 bytes)** | [qinco2_L-bigann1M-16x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/qinco2_L-bigann1M-16x8.pt) | [qinco2_L-deep1M-16x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/qinco2_L-deep1M-16x8.pt) | [qinco2_L-contriever1M-16x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/qinco2_L-contriever1M-16x8.pt) | [qinco2_L-FB_ssnpp1M-16x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/qinco2_L-FB_ssnpp1M-16x8.pt) |

### IVF models checkpoints

These models are trained with and additional codebook of $K_{IVF}=2^20=1048576$ codewords, corresponding to the IVF step. They can be used to evaluate large-scale search.

|  | **BigANN1B** | **Deep1B** |
|---|---|---|
| **Qinco2-S (8 bytes)** | [IVF-qinco2_S-bigann1B-8x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/IVF-qinco2_S-bigann1B-8x8.pt) | [IVF-qinco2_S-deep1B-8x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/IVF-qinco2_S-deep1B-8x8.pt) |
| **Qinco2-S (16 bytes)** | [IVF-qinco2_S-bigann1B-16x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/IVF-qinco2_S-bigann1B-16x8.pt) | [IVF-qinco2_S-deep1B-16x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/IVF-qinco2_S-deep1B-16x8.pt) |
| **Qinco2-S (32 bytes)** | [IVF-qinco2_S-bigann1B-32x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/IVF-qinco2_S-bigann1B-32x8.pt) | [IVF-qinco2_S-deep1B-32x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/IVF-qinco2_S-deep1B-32x8.pt) |
| **Qinco2-M (8 bytes)** | [IVF-qinco2_M-bigann1B-8x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/IVF-qinco2_M-bigann1B-8x8.pt) | [IVF-qinco2_M-deep1B-8x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/IVF-qinco2_M-deep1B-8x8.pt) |
| **Qinco2-M (16 bytes)** | [IVF-qinco2_M-bigann1B-16x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/IVF-qinco2_M-bigann1B-16x8.pt) | [IVF-qinco2_M-deep1B-16x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/IVF-qinco2_M-deep1B-16x8.pt) |
| **Qinco2-M (32 bytes)** | [IVF-qinco2_M-bigann1B-32x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/IVF-qinco2_M-bigann1B-32x8.pt) | [IVF-qinco2_M-deep1B-32x8.pt](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/IVF-qinco2_M-deep1B-32x8.pt) |

We also provide the IVF centroids used to create these models:

* [ivf_centroids_bigann1B_1048576.npy](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/ivf_centroids_bigann1B_1048576.npy) 
* [ivf_centroids_deep1B_1048576.npy](https://dl.fbaipublicfiles.com/QINCo/qinco2-models/ivf_centroids_deep1B_1048576.npy) 

## Usage

Every command uses the `run.py` endpoint. It can be run on multiple GPUs with [accelerate](https://huggingface.co/docs/accelerate/index) using the `./run.sh` script. If you are familiar with accelerate, you can use it directly to launch `run.py`.
Command-line arguments are parsed using the [Hydra](https://hydra.cc/) format. You can find the default configuration and all overloadable parameters inside the `config/qinco_cfg.yaml` file.

**Running on an single GPU or on CPU**: In any of the following commands, you can replace `./run.sh` by `python run.py` to use a single GPU. Set the `cpu` argument to true (`python run.py cpu=true`) to run on CPU instead.

**Datasets**: for all commands below, you can either use your own data using the `db`, `trainset`, `queries` and `queries_gt` arguments, or use one of the *default* dataset that is used within the paper. To use one of theses datasets, replace these arguments by  `db=<name of the dataset>`. The paths will be automatically populated. Be sure to download the corresponding datasets beforehand (see above).

**Available default datasets**: `db=FB_ssnpp1M`, `db=contriever1M`, `db=bigann1M`, `db=bigann1B`, `db=bigann1B`, `db=deep1B`.
The `1M` datasets are intended to be used for most tasks, while the `1B` datasets should only be used to build and search a index for large-scale search.

**Using your own data**: you will need at least a vector database (`db`) and a set of training vectors (`trainset`) to train and evaluate the MSE of the model.
Available data formats are: `bvecs`, `fvecs`, `ivecs`, `npy`, and the format should be a single matrix of dimensions `(N_samples, D)`.
During training, the last 10,000 vectors of the training set will set appart at the validation set. The database is the test set.
Additionally, for nearest-neighbour search, you need a set of queries (`queries`, matrix of dimension `(N_queries, D)`) and the id of their answers in the database (`queries_gt`, matrix of dimension `(N_queries, 1)`).

**Using a subset of the data files**: you can use only a subset of the training set and/or database, which can be usefull for testing or using limited data. Use `ds.trainset=<...>` and `ds.db=<...>` to limit their size. Similarly, you can control the number of validation samples extracted from the training set with `ds.valset=<...>`, and the number of samples from the training set used at each epoch with `ds.loop=<...>`. If not specified, the default values are `ds.valset=10_000` and `ds.loop=10_000_000`.


### Training 

```bash
./run.sh task=train  \
    output=<model_weights.pt> \
    [resume=false] \
    [model_args=<model preset>] \
    [L=<L> dh=<dh> M=<M> K=<K>] [A=<A>] [B=<B>] [de=<de>] \
    [db=<db_name>] [trainset=<trainset_path>] \
    [ivf_centroids=<path_to_ivf_centroids>] \
    [ds.loop=<epoch_size>] [ds.trainset=<trainset_max_size>] [ds.valset=<validation_set_max_size>] \
    [lr=0.0008] [batch=1024] [epochs=60] [grad_accumulate=1] [grad_clip=0.1] \
    [verbose=true] \
    [tensorboard=<path_to_directory>] \
    [model=<path_to_resume_from>]
```

This command trains a new model and accepts a set of required and optional arguments:

* The `output` argument should specify the path to the `.pt` checkpoint file that will be created during training.
    * If the `resume` argument is set to `true` (default is `false`), the training will resume from the `output` path *if it exists*, or starts from scratch otherwise.
* You need to specify the models arguments `K` (codebook size, set to 256 in the paper), `M` (number of QINCo2 steps, and bytes if $K=256$), `dh` (hidden dimension $d_h$), `de` (embedding dimension $d_e$), `L` (number of residual blocks in each step), `A` (number of fast pre-selected candidates) and `B` (size of beam search). See the paper for reference.
    * The values of `A`, `B` and `de` can be set to `0` or `null` (default value) to disable these components. `A=null` implies no beam search, `B=0` implies no candidates pre-selection, and `de` implies an embedding dimension of same size as the data dimension.
    * You can also use a **preset**, using `model_args=<model-name>`. Available models are: `qinco1`, `qinco2-S`, `qinco2-M`, `qinco2-L`, which will set all these arguments if not specified manually.
* You need to specify a path to the training data file the `trainset` argument, or using `db=<db_name>` (see "Datasets" above).
* If you want to train an IVF model (e.g. `IVF-qinco2-S`), you can optionnaly specifiy `ivf_centroids` with a path to an IVF centroid file. See below how to generate one.
* For arguments `ds.loop`, `ds.trainset`, `ds.valset`, see above ("Using a subset of the data files"). `ds.loop` will set the size of an epoch.
* You can control training by overloading the learning rate (`lr`), batch size **per** GPU (`batch`), the number of epochs (`epochs`), and also `grad_accumulate` and `grad_clip`.
* You can set `verbose` to false to print only at the beginning and end of an epoch, instead of printing a line at each batch iteration.
* The `tensorboard` argument can be specified to log training curves inside a directory. They can be displayed using tensorboard, see [https://www.tensorflow.org/tensorboard](tensorboard documentation) (usually with `tensorboard --logdir=path_to_directory`).
* You can load a previous model and resume training from it by specifying a path with `model`. It should resume from the last epoch, and it will automatically set the model arguments.

**Warning**: The MSE error shown is the validation error, and can differ significantly from the test error (see "Evaluating a model") as a smaller subset is used to compute this estimate.


<u>Usage examples:</u>

```bash
# Train a qinco2-S model on deep1M, with 16 bytes. Resume if the output file already exists.
./run.sh task=train model_args=qinco2-S db=deep1M M=16 output=runs/weights/qinco2_S-deep1M-16x8.pt resume=true

# Train a very small model (qinco2-S with some overloaded arguments) on a custom dataset, with small epochs of 100_000 samples
./run.sh task=train model_args=qinco2-S L=1 dh=32 de=64 A=4 B=4 \
    output=my_super_small_model-my_dataset-8x8.pt \
    trainset=my_dataset-trainset.fvecs \
    ds.loop=100_000 ds.valset=10_000

# Resume training from a previous model on a single CPU, with low verbosity
python run.py cpu=true task=train model=runs/weights/qinco2_S-deep1M-8x8.pt output=runs/weights/qinco2_S-deep1M-8x8-2.pt db=deep1M verbose=false

# Train a qinco1 model on deep1M, with 8 bytes. It doesn't use beam search or candidates pre-selection.
./run.sh task=train model_args=qinco1 db=deep1M output=runs/weights/qinco1-deep1M-8x8.pt
```


### Evaluating a model

#### Evaluating on the test set

```bash
./run.sh task=eval \
    model=<model_path.pt> \
    db=<database_path or db_name> \
    [A=<A>] [B=<B>] \
    [ds.db=<limit_for_data>]
```

This command evaluates the MSE on the full database. It should yield MSEs comparables to the ones from the paper (Table 3).

* `model` should specify a path to a model, either trained using the command above, or downloaded from the "Pretrained checkpoints" section.
* `db` should be a path to the dataset, or the name of a pre-defined one (see "Datasets" above).
* `A` and `B` can optionally be overloaded to change the run-time beam size and candidates pre-selection size. If not overloaded, the values from training are used.
* `ds.db` can be used to limit the amount of data used to evaluate the model.


<u>Usage examples:</u>

```bash
# Gives the MSE from the paper for the QINCo2-L model on bigann1M (8 bytes) with a larger beam
./run.sh task=eval model=models/qinco2_L-bigann1M-8x8.pt db=bigann1M A=32 B=64

# Evaluates a small custom model on 100_000 samples from a custom dataset
./run.sh task=eval model=my_super_small_model-my_dataset-8x8.pt db=my_dataset-db.fvecs ds.db=100_000
```


#### Evaluating retrieval accuracy

```bash
python run.py task=search \
    model=<model_path.pt> \
    db=<database_path or db_name> \
    [A=<A>] [B=<B>] \
    [ds.db=<limit_for_data>]
```


This command returns the retrieval accuracy (R@1 from table 3, but also R@10 and R@100) on a dataset, with full decoding of the database using QINCo2.
It **does not** evaluate large-scale search using the custom pipeline shown in Figure 3. Arguments are similar to the `eval` command.

**Single GPU process**: this command should be ran using a single process (`python run.py`), and will use a single GPU.

<u>Usage example:</u>

```bash
# Gives the R@1 from the paper for the QINCo2-L model on bigann1M (8 bytes) with a larger beam
python run.py task=search model=models/qinco2_L-bigann1M-8x8.pt db=bigann1M A=32 B=64
```


#### Evaluating on the validation set

```bash
./run.sh task=eval_valset \
    model=<model_path.pt> \
    [db=<db_name>] [trainset=<trainset_path>] \
    [A=<A>] [B=<B>] \
    [ds.valset=<>]
```

This command can be used to evaluate on the validation set (extracted from the training set) and get the MSE also obtained during training.
It works similarly to `task=eval`, but takes the trainset path (and optionally, thevalidation set size) as arguments.

### Using QINCo2 with IVF

#### Building IVF centroids

```bash
python run.py task=ivf_centroids \
    ivf_K=<IVF_codebooks_size> \
    output=<centroids_weights.npy> \
    [db=<db_name>] [trainset=<trainset_path>] \
    [ds.trainset=100_000] [ds.valset=10_000]
```

Before using the IVF centroids, you need to create them with this command, or use one of the pre-trained centroids from below.

* `ivf_K` sets the number of centroids used. In the paper, we use `ivf_K=1048576`.
* `output` should be a `.npy` path.
* `ds.trainset` can be used to train on a smaller set of vectors, if the training takes too long.

**Single GPU process**: this command should be ran using a single process (`python run.py`), and will use a single GPU.

<u>Usage examples:</u>

```bash
# Trains centroids 1048576 on deep1M, as in the paper. It should give a result similar to the `ivf_centroids_deep1B_1048576.npy` file.
python run.py task=ivf_centroids ivf_K=1048576 db=deep1M output=runs/ivf_centroids/ivf_centroids-deep1M-1048576.npy

# Trains only 700 centroids on a custom dataset
python run.py task=ivf_centroids ivf_K=700 \
    trainset=my_dataset-trainset.fvecs \
    output=runs/ivf_centroids/ivf_centroids-my_dataset-K=700.npy
```

#### Using centroids when training a model

You can train a model with an additional IVF first step, which use the IVF centroids with no beam search, following the same instructions as above for training.
You need to add the `ivf_centroids` parameter.
These models can be evaluated in the same way as other model to obtain the MSE / retrieval accuracy on the database, while using only 

<u>Usage example:</u>

```bash
# Train an IVF-qinco2-S model on deep1M, with 8 bytes.
./run.sh task=train model_args=qinco2-S db=deep1M output=runs/weights/IVF-qinco2_S-deep1M-8x8.pt ivf_centroids=models/ivf_centroids-deep1M-1048576.npy
```


#### Encode the training set and database

```bash
./run.sh task=encode \
    model=<model_path.pt> \
    output=<encoded_db_path.npz> \
    db=<database_path or db_name> \
    [encode_trainset=<false>] \
    [A=<A>] [B=<B>] \
    [ds.db=<limit_for_data>] [ds.trainset=<limit_for_trainset>]
```

This command encodes a set of vectors using the specified QINCo2 model.
You should encode both the *training set* and the *database*.
When using a predefined dataset (e.g. `db=deep1B`), add the argument `encode_trainset=true` to encode the training set instead of the database.
As this step can take a very long time on a billion-scale database, it is recommended to launch this command with multiple GPUs available.
It will do a **parallel encoding** of the database, where each GPU work on a substep of it (e.g. use 100 GPUs for a 100x acceleration of the encoding process).

<u>Usage example:</u>

```bash
# Encode the 1B vectors from deep1B dataset using an IVF-QINCo-S model
# Note that we are using `db=deep1B` here instead of `db=deep1M`, to encode all 1B vectors
./run.sh task=encode db=deep1B model=models/IVF-qinco2_S-deep1B-8x8.pt \
    output=runs/encoded_db/IVF-qinco2_S-deep1B-8x8_db.npz

# Encode the training set from deep1B
./run.sh task=encode db=deep1B encode_trainset=true model=models/IVF-qinco2_S-deep1B-8x8.pt \
    output=runs/encoded_db/IVF-qinco2_S-deep1B-8x8_trainset.npz

# Encode the training set and database for a custom dataset
./run.sh task=encode db=my_dataset-trainset.fvecs model=my-custom-IVF-model.pt output=my_encoded_trainset.npz
./run.sh task=encode db=my_dataset-db.fvecs model=my-custom-IVF-model.pt output=my_encoded_db.npz
```

#### Train a pairwise decoder

```bash
python run.py task=train_pairwise_decoder \
    ivf_centroids=<path_to_ivf_centroids> \
    output=<path_for_pairwise_decoder.pt> \
    [trainset=<trainset_path>] [db=<db_name>] \
    encoded_trainset=<path_to_encoded_trainset.npz> \
    [ds.trainset=<limit_for_trainset>] [ds.valset=<limit_for_valset>]
```

This command builds a pairwise additive decoder ("Pairwise additive decoding", section 3.3 in the QINCo2 paper) that can be used to improve performances of large-scale search within an index (see below).
It requires both the encoded as well as unencoded training set, and the IVF centroids.

**Single GPU process**: this command should be ran using a single process (`python run.py`), and will use a single GPU.

<u>Usage example:</u>

```bash
# Use the provided IVF centroids for deep1M and the previously encoded deep1M/deep1B training set to create a pairwise encoded
python run.py task=train_pairwise_decoder db=deep1B \
    ivf_centroids=models/ivf_centroids-deep1M-1048576.npy \
    output=runs/weights/qinco2s-ivf-deep1B-8x8_pairwise_decoder.pt \
    encoded_trainset=runs/encoded_db/IVF-qinco2_S-deep1M-8x8_trainset.npz
```

#### Build a search index

```bash
python run.py task=build_index \
    ivf_centroids=<path_to_ivf_centroids> \
    output=<paht_to_store_index.faissindex> \
    [trainset=<trainset_path>] [db=<db_name>] \
    encoded_trainset=<path_to_encoded_trainset.npz> \
    encoded_db=<path_to_encoded_db.npz> \
    [ds.db=<limit_for_data>] [ds.trainset=<limit_for_trainset>] [ds.valset=<limit_for_valset>]
```

This command creates a [faiss](https://github.com/facebookresearch/faiss) index to efficiently search within billion-scale databases, using the previously encoded database. The training set is used to train a set of AQ codebooks for the first fast approximative shortlist.

**Single GPU process**: this command should be ran using a single process (`python run.py`), and will use a single GPU.

<u>Usage example:</u>

```bash
python run.py task=build_index db=deep1B \
    ivf_centroids=models/ivf_centroids-deep1M-1048576.npy \
    output=runs/index/index-IVF-qinco2_S-deep1B-8x8.faissindex \
    encoded_trainset=runs/encoded_db/IVF-qinco2_S-deep1M-8x8_trainset.npz \
    encoded_db=runs/encoded_db/IVF-qinco2_S-deep1M-8x8_db.npz \
    ds.db=300_000 ds.trainset=100_000 ds.valset=10_000
```

#### Search inside an index

```bash
python run.py cpu=true task=search \
    model=<model_path.pt> \
    index=runs/index/index-qinco2s-ivf-deep1M-8x8.faissindex \
    [queries=<path_to_queries>] [queries_gt=<path_to_groundtruth>] [db=<db_name>] \
    [pairwise_decoder=<path_for_pairwise_decoder.pt>] \
    [output=<output_logs.json>] [resume=<false/true>]
```

This command search over a faiss index using the optimized search pipeline shown in Figure 3 in the paper.
It will explore different search parameters to find a pareto-optimal frontier for the speed/accuracy tradeoff shown in Figure 6 of the paper.
The model will only be used to decode elements at the end of the search pipeline.

**Single process, on CPUs**: this command should be ran using a single process, and only on (up to 32) CPUs (`python run.py cpu=true`). Our experiments in the paper used 32 CPUs for the timing.


* `pairwise_decoder`: optional argument. If specified, the search will also explore the use (or not) of the pairwise decoder within the pipeline to increase search speed.
* `output`: optional argument. If specified, all the explored search settings with their corresponding accuracies and timings will be loged into the file as a JSON object.
    * `resume` if specified witht the `output` argument, will continue exploration of search settings from a previously uncompleted `search` command.
* The queries and ground-truth answers for those should be stored as `(N_queries, D)` (floats or integers) and `(N_queries, 1)` (integers: id of the correct nearest neighbour in the database) arrays.
    * You can instead use a default database using the `db=<db_name>` argument, which will automatically give the queries and desired answers.


<u>Usage example:</u>

```bash
# Gives the search speed and accuracies within the deep1B database, for a set of search parameters.
python run.py cpu=true task=search db=deep1B \
    model=models/IVF-qinco2_S-deep1B-8x8.pt \
    index=runs/index/index-IVF-qinco2_S-deep1B-8x8.faissindex \
    output=runs/logs/search_results-IVF-qinco2_S-deep1B-8x8_v2.json resume=true \
    pairwise_decoder=runs/weights/qinco2s-ivf-deep1B-8x8_pairwise_decoder.pt
```




## Legal
Qinco2 is licenced under CC-BY-NC, please refer to the LICENSE file in the top level directory.

Copyright © Meta Platforms, Inc. See the Terms of Use and Privacy Policy for this project.
