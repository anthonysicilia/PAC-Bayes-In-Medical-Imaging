#!/bin/bash

python3 -m src.main --model=light --init_lr=1e-2 --lr_step=30 --epochs=120 --momentum=0.95 --batch_size=8 --sigma_prior=0.2 --kl_dampening=1 --train_bound=variational --prior_max_train=30 --use_prefix --mc_samples=100 --estimator=sample --task=segment --device=1 --freeze_batchnorm