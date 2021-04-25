#!/bin/bash

python3 -m src.main --model=unet --init_lr=1e-2 --lr_step=30 --epochs=120 --momentum=0.95 --batch_size=8 --task=segment --sigma_prior=0.03 --kl_dampening=1 --prior_max_train=30 --use_prefix --estimator=sample --device=0 --freeze_batchnorm
