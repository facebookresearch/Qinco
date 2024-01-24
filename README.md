## **Q**uantization with **I**mplicit **N**eural **C**odebooks (QINCo)

This code repository corresponds to the paper [Residual Quantization with Implicit Neural Codebooks](https://insert_link_here), in which **Q**uantization with **I**mplicit **N**eural **C**odebooks (QINCo) was proposed. 

QINCo is a neurally-augmented algorithm for multi-codebook vector quantization, specifically residual quantization (RQ). Instead of using a fixed codebook per quantization step, QINCo uses a neural network to predict a codebook for the next quantization step, conditioned upon the quantized vector so far. In other words, the codebooks to be used depend on the Voronoi cells selected previously. This greatly enhances the capacity of the compression system, without the need to store more codebook vectors explicitly. 

An additional advantage of QINCo is its modularity. Thanks to training each quantization step with its own quantization error, the trained system for a certain compression rate, can also be exploited for lower compression rates, making QINCo a dynamic rate quantizer.

In the paper we propose three addtional variants of QINCo:

- QINCo-LR is more suitable for large-dimensional embeddings, as its capacity scales linearly in the vector dimensions, instead of quadratically (as QINCo does).
- integration of QINCo with Product Quantization (PQ-QINCo), which is similar to the integration of normal RQ with PQ.
- to accelerate search we combine an Inverted File Index (IVF) structure with approximate decoding and re-ranking with QINCO (IVF-QINCo).


Find below the documentation on how to use QINCo and its variants:

#### [Instructions for installation of the environment](docs/installation.md)

#### [Downloading data](docs/downloading_data.md)

#### [Using pre-trained QINCo checkpoints for encoding and decoding](docs/checkpoints.md)

#### [Training QINCo(-LR) from scratch](docs/training.md)

#### [Training PQ-QINCo from scratch](docs/PQ_QINCo.md)

#### [Approximate AQ searching and QINCo re-ranking (Replicating Table 4)](docs/AQ_approximate_search.md)

#### [IVF_search](docs/IVF_search)

## License

This repository is under the CC-BY-NC license found in the root of the
repository.

## Citation

If you use QINCo in a research work please cite:

```
@article{huijben2023QINco,
  title={Residual Quantization with Implicit Neural Codebooks},
  author={Iris A.M. Huijben and Matthijs Douze and Matthew J. Muckley and Ruud J.G. van Sloun and Jakob Verbeek},
  year={2024}
}
```
