#!/bin/sh
env="mujoco"
scenario="ManyAgentGoToGoalEnv-v0"
num_agents=15
agent_obsk=0
faulty_node=-1
eval_faulty_node=-1
algo="happo"
exp="mlp"
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python train/eval_manygotogoal.py \
 --seed ${seed} \
 --env_name ${env} \
 --algorithm_name ${algo} \
 --experiment_name ${exp} \
 --scenario ${scenario} \
 --num_agents ${num_agents} \
 --agent_obsk ${agent_obsk} \
 --faulty_node ${faulty_node} \
 --eval_faulty_node ${eval_faulty_node} \
<<<<<<< Updated upstream
 --model_dir "CHANGE_ME_MODEL_DIR" \
 --allow_partial_restore \
=======
 --model_dir "/home/shahils/Desktop/marl_ws/HAPPO-HATRPO/scripts/results/mujoco/ManyAgentGoToGoalEnv-v0/happo/mlp/7412/wandb/run-20260302_014516-uo2vf2xi/files" \
>>>>>>> Stashed changes
 --eval_loops 100 \
 --eval_episodes 100 \
 --n_training_threads 32 \
 --n_rollout_threads 1 \
 --n_eval_rollout_threads 1 \
 --episode_length 200 \
 --num_env_steps 200000000 \
 --add_center_xy \
 --use_state_agent \
 --use_eval True \
 --use_wandb True \
 --wandb_name "xxx" \
 --user_name "shahil-shaik7-clemson-university"
