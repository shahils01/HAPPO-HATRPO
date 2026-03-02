#!/bin/sh
env="mujoco"
scenario="ManyAgentGoToGoalEnv-v0"
num_agents=15
agent_obsk=0
algo="happo"
exp="mlp"
running_max=5
kl_threshold=1e-4
echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for number in `seq ${running_max}`;
do
    echo "the ${number}-th running:"
    CUDA_VISIBLE_DEVICES=1 python train/train_manygotogoal.py \
        --env_name ${env} \
        --algorithm_name ${algo} \
        --experiment_name ${exp} \
        --scenario ${scenario} \
        --num_agents ${num_agents} \
        --agent_obsk ${agent_obsk} \
        --lr 4e-5 \
        --critic_lr 4e-5 \
        --clip_param 0.2 \
        --std_x_coef 1 \
        --std_y_coef 5e-1 \
        --running_id ${number} \
        --n_training_threads 32 \
        --n_rollout_threads 32 \
        --num_mini_batch 1 \
        --episode_length 200 \
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
done
