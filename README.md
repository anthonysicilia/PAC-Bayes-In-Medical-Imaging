# PAC-Bayes-In-Medical-Imaging


## Introduction

The repo for the pre-print work **"PAC Bayesian Performance Guarantees for Deep(Stochastic) Networks in Medical Imaging."**
Available at: https://arxiv.org/abs/2104.05600


## Preparation



### Prerequisites

- Python 3.6
- Pytorch 1.4
- numpy
- tqdm
- pandas
- PIL

### Dataset Preparation

- Run `get_data.sh` to retrieve the ISIC2018 challenge data.
- Run `make_split.py` to generate a train test split.
- Run `python3 -m src.main **kwargs` to train models and compute bounds.

## Training

To reproduce all results run the following scripts:
- `sh scripts/run-lw.sh`
- `sh scripts/run-rn.sh`
- `sh scripts/run-un.sh`

To run individual experiments, please check the comments which identify each 
subcall with the correct experiment.
