# miann
Multiplexed Image Analysis with Neural Networks

## Installation

1. Create new conda environment:

`
conda create -n pelkmans-3.9 python=3.9
conda activate pelkmans-3.9
`

2. Install tensorflow:

`
pip install --upgrade pip
pip install tensorflow
`

3. Install other packages:

`
conda install matplotlib jupyterlab pandas tqdm
`

4. Install miann  

TODO: scanpy, squidpy
```
conda create -n pelkmans-3.9 python=3.9 tensorflow matplotlib jupyterlab pandas tqdm
conda install -c conda-forge leidenalg
pip install -e .
```

