for seed in 0
do
    python3 -m src.main \
        --task=classify \
        --model=resnet18 \
        --init_lr=5e-1 \
        --batch_size=64 \
        --baseline \
        --random_seed=$seed \
        --device=0
        # --wandb=miccai21-rerun \
        # --wandb_id=rs$seed/fig_a/RN

    python3 -m src.main \
        --task=classify \
        --model=resnet18 \
        --init_lr=5e-1 \
        --batch_size=64 \
        --use_prefix \
        --freeze_batchnorm \
        --mc_samples=1000 \
        --random_seed=$seed \
        --device=0
        # --wandb=miccai21-rerun \
        # --wandb_id=rs$seed/fig_a/RN-PBB

    python3 -m src.main \
        --task=classify \
        --model=resnet18 \
        --init_lr=5e-1 \
        --batch_size=64 \
        --use_prefix \
        --freeze_batchnorm \
        --sigma_prior=0.001 \
        --random_seed=$seed \
        --device=0
        # --wandb=miccai21-rerun \
        # --wandb_id=rs$seed/fig_d/sigma_prior_0.001

    python3 -m src.main \
        --task=classify \
        --model=resnet18 \
        --init_lr=5e-1 \
        --batch_size=64 \
        --use_prefix \
        --freeze_batchnorm \
        --sigma_prior=0.005 \
        --random_seed=$seed \
        --device=0
        # --wandb=miccai21-rerun \
        # --wandb_id=rs$seed/fig_d/sigma_prior_0.005

    python3 -m src.main \
        --task=classify \
        --model=resnet18 \
        --init_lr=5e-1 \
        --batch_size=64 \
        --use_prefix \
        --freeze_batchnorm \
        --sigma_prior=0.01 \
        --random_seed=$seed \
        --device=0
        # --wandb=miccai21-rerun \
        # --wandb_id=rs$seed/fig_d/sigma_prior_0.01

    python3 -m src.main \
        --task=classify \
        --model=resnet18 \
        --init_lr=5e-1 \
        --batch_size=64 \
        --use_prefix \
        --freeze_batchnorm \
        --sigma_prior=0.02 \
        --random_seed=$seed \
        --device=0
        # --wandb=miccai21-rerun \
        # --wandb_id=rs$seed/fig_d/sigma_prior_0.02

    python3 -m src.main \
        --task=classify \
        --model=resnet18 \
        --init_lr=5e-1 \
        --batch_size=64 \
        --use_prefix \
        --freeze_batchnorm \
        --sigma_prior=0.03 \
        --random_seed=$seed \
        --device=0
        # --wandb=miccai21-rerun \
        # --wandb_id=rs$seed/fig_d/sigma_prior_0.03

    python3 -m src.main \
        --task=classify \
        --model=resnet18 \
        --init_lr=5e-1 \
        --batch_size=64 \
        --use_prefix \
        --freeze_batchnorm \
        --sigma_prior=0.04 \
        --random_seed=$seed \
        --device=0
        # --wandb=miccai21-rerun \
        # --wandb_id=rs$seed/fig_d/sigma_prior_0.04

    python3 -m src.main \
        --task=classify \
        --model=resnet18 \
        --init_lr=5e-1 \
        --batch_size=64 \
        --use_prefix \
        --freeze_batchnorm \
        --sigma_prior=0.05 \
        --random_seed=$seed \
        --device=0
        # --wandb=miccai21-rerun \
        # --wandb_id=rs$seed/fig_d/sigma_prior_0.05
done