
QINCo requires:

- a recent version of pytorch (paper models were run with Pytorch 2.1.1)

- Faiss (only for training and fast search experiments)

As of 2023-11-27, both can be installed in an Anaconda environment on Linux with
```
conda create --name faiss_for_qinco  python=3.10
conda activate faiss_for_qinco
conda install -c pytorch/label/nightly -c nvidia faiss-gpu=1.7.4
conda install pytorch  pytorch-cuda=11.8 -c pytorch -c nvidia
```
This will install Pytorch 2.1.1, the nightly GPU Faiss for yesterday,

Note that IVF search requires a more
recent version of Faiss than the current official package (1.7.4), hence the
nightly package.
