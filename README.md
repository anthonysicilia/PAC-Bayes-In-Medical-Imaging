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



To reproduce the results showed in the fig a, b, c, and d, please run the following scripts.

### Fig a

- `sh scripts/fig_a/LW.sh`
- `sh scripts/fig_a/LW-PBB.sh`
- `sh scripts/fig_a/U-Net.sh`
- `sh scripts/fig_a/U-Net-PBB.sh`

### Fig b

- `sh scripts/fig_b/sigma_prior_0.001.sh`
- `sh scripts/fig_b/sigma_prior_0.005.sh`
- `sh scripts/fig_b/sigma_prior_0.01.sh`
- `sh scripts/fig_b/sigma_prior_0.02.sh`
- `sh scripts/fig_b/sigma_prior_0.03.sh`
- `sh scripts/fig_b/sigma_prior_0.04.sh`
- `sh scripts/fig_b/sigma_prior_0.05.sh`

### Fig c

- `sh scripts/fig_c/sigma_prior_0.001.sh`
- `sh scripts/fig_c/sigma_prior_0.005.sh`
- `sh scripts/fig_c/sigma_prior_0.01.sh`
- `sh scripts/fig_c/sigma_prior_0.02.sh`
- `sh scripts/fig_c/sigma_prior_0.03.sh`
- `sh scripts/fig_c/sigma_prior_0.04.sh`
- `sh scripts/fig_c/sigma_prior_0.05.sh`
- `sh scripts/fig_c/sigma_prior_0.1.sh`
- `sh scripts/fig_c/sigma_prior_0.2.sh`

### Fig d

- `sh scripts/fig_d/LW.sh`
- `sh scripts/fig_d/U-Net.sh`
