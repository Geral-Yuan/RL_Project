# import wandb
import swanlab
import argparse
import datetime
import random
import numpy as np

import os
import json

from train import *

def main(args):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config_path = f"config/{args.model_type}/{args.env_name}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist. Please ensure the file is present in the config directory.")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    if args.use_swanlab:
        swanlab.login(api_key=args.swanlab_key)
        swanlab.init(project="RL_Final_Project", config=config, experiment_name=f"{args.env_name}_{args.model_type}_{TIMESTAMP}")
    
    if args.model_type in ALGO_LIST["Value-Based"]:
        train_DQN(args, config, device, TIMESTAMP)
        
    elif args.model_type in ALGO_LIST["Policy-Based"]:
        if args.model_type == "PPO":
            train_PPO(args, config, device, TIMESTAMP)
        elif args.model_type == "DDPG":
            train_DDPG(args, config, device, TIMESTAMP)
        
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', required=True, type=str, choices=ENV_LIST["Atari"] + ENV_LIST["MuJoCo"], help='Environment name')
    parser.add_argument('--use_swanlab', action='store_true', help='Use SwanLab for logging and visualization')
    parser.add_argument('--swanlab_key', type=str, default=None, help='SwanLab API key')
    # parser.add_argument("--use_wandb", action="store_true", help="Use WandB for logging")
    # parser.add_argument("--wandb_key", type=str, default=None, help="WandB API key")
    parser.add_argument("--model_type", type=str, default=None, choices=["DQN", "DoubleDQN", "DuelingDQN", "PPO", "DDPG"], help="Type of model to use (default: None, which will use DQN for Atari and PPO for MuJoCo)")
    # parser.add_argument("--max_steps", type=int, default=None, help="Max steps per episode")
    parser.add_argument("--store_gif", action="store_true", help="Store GIFs of episodes")
    
    args = parser.parse_args()
    
    TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if args.model_type is None:
        if args.env_name in ENV_LIST["Atari"]:
            args.model_type = "DQN"
        elif args.env_name in ENV_LIST["MuJoCo"]:
            args.model_type = "PPO"
    
    if not check_env_algo(args.env_name, args.model_type):
        raise ValueError(f"Environment {args.env_name} is not compatible with model {args.model_type}. Please choose a value-based model for Atari or a policy-based model for MuJoCo.")
    
    # if args.use_wandb:
    #     wandb.login(key=args.wandb_key)
    #     wandb.init(project="RL Final Project", config=args, name=f"{args.env_name}_{args.model_type}_{TIMESTAMP}")
    
    main(args)