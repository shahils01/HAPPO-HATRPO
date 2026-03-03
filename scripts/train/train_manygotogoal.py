#!/usr/bin/env python
import sys
import os
import wandb
sys.path.append("../")
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
import ManyAgent_GoTOGoal
import gymnasium as gym
from configs.config import get_config
from envs.ma_mujoco.multiagent_mujoco.mujoco_multi import MujocoMulti
from envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from runners.separated.mujoco_runner import MujocoRunner as Runner
"""Train script for Mujoco."""


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "mujoco":
                scenario = normalize_scenario(all_args.scenario)
                env = gym.make(scenario, disable_env_checker=True)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "mujoco":
                scenario = normalize_scenario(all_args.scenario)
                env = gym.make(scenario, disable_env_checker=True)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def normalize_scenario(scenario):
    aliases = {
        "ManyAgentGoToGoalEnv": "ManyAgentGoToGoal-v0",
        "ManyAgentGoToGoalEnv-v0": "ManyAgentGoToGoal-v0",
        "ManyAgentGoToGoal": "ManyAgentGoToGoal-v0",
    }
    return aliases.get(scenario, scenario)


def parse_args(args, parser):
    parser.add_argument('--scenario', type=str, default='ManyAgentGoToGoal-v0', help="Which mujoco task to run on")
    parser.add_argument('--num_agents', type=str, default='5')
    parser.add_argument('--agent_obsk', type=int, default=0)
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)

    # agent-specific state should be designed carefully
    parser.add_argument("--use_state_agent", action='store_true', default=False)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_true', default=False)
    parser.add_argument("--faulty_node", type=int, default=-1)
    parser.add_argument("--eval_faulty_node", type=int, nargs="+", default=None)
    parser.add_argument("--eval_loops", type=int, default=100)
    parser.add_argument("--use_single_network", action='store_true', default=False)

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    print("all config: ", all_args)
    if all_args.seed_specify:
        all_args.seed=all_args.runing_id
    else:
        all_args.seed=np.random.randint(1000,10000)
    print("seed is :",all_args.seed)
    # cuda
    if all_args.cuda: # and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.scenario / all_args.algorithm_name / all_args.experiment_name / str(all_args.seed)
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project="IROS_" + all_args.scenario,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.env_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                                str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = envs.n_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
