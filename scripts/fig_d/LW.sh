#!/bin/bash

python3 -m src.main --model=light --init_lr=1e-2 --lr_step=30 --epochs=120 --momentum=0.95 --batch_size=8 --train_bound=none --task=segment --mc_samples=100 --estimator=sample --device=1 --freeze_batchnorm
