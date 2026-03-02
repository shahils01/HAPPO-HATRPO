#!/bin/sh
env="mujoco"
scenario="HalfCheetah-v5"
agent_conf="6x1"
agent_obsk=0
algo="happo"
exp="mlp"
running_max=20
kl_threshold=1e-4
seed=0
echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

CUDA_LAUNCH_BLOCKING=1 python train/train_mujoco.py \
    --seed ${seed} \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --scenario ${scenario} \
    --agent_conf ${agent_conf} \
    --agent_obsk ${agent_obsk} \
    --lr 5e-4 \
    --critic_lr 5e-4 \
    --clip_param 0.1 \
    --hidden_size 256 \
    --std_x_coef 1 \
    --std_y_coef 5e-1 \
    --running_id ${number} \
    --n_training_threads 32 \
    --n_rollout_threads 40 \
    --num_mini_batch 1 \
    --episode_length 100 \
    --num_env_steps 200000000 \
    --ppo_epoch 10 \
    --kl_threshold ${kl_threshold} \
    --use_value_active_masks \
    --use_eval \
    --add_center_xy \
    --use_state_agent\
    --share_policy \
    --use_wandb True \
    --wandb_name "xxx" \
    --user_name "shahil-shaik7-clemson-university"


