#!/bin/sh
env="football"
scenario="academy_3_vs_1_with_keeper"
# academy_pass_and_shoot_with_keeper n_agent=3
# academy_3_vs_1_with_keeper n_agent=4
# academy_counterattack_easy n_agent=11
# 11_vs_11_easy_stochastic n_agent=11
n_agent=4
algo="happo"
exp="single"
seed=0
running_max=20
kl_threshold=1e-4
echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for number in `seq ${running_max}`;
do
    echo "the ${number}-th running:"
    CUDA_VISIBLE_DEVICES=0 apptainer exec --nv /home/shahils/gfootball_apptainer/gfootball.sif \
    python3 train/train_football.py \
        --env_name ${env} \
        --algorithm_name ${algo} \
        --experiment_name ${exp} \
        --scenario ${scenario} \
        --agent_conf ${agent_conf} \
        --agent_obsk ${agent_obsk} \
        --lr 5e-5 \
        --critic_lr 5e-5 \
        --clip_param 0.2 \
        --std_x_coef 1 \
        --std_y_coef 5e-1 \
        --running_id ${number} \
        --n_training_threads 32 \
        --n_rollout_threads 40 \
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
        # --use_wandb True \
        # --wandb_name "xxx" \
        # --user_name "shahil-shaik7-clemson-university"
done
