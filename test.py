import torch
import os
import imageio.v2 as imageio

from utils import *

def eval_agent(agent, env_name, gif_path=None):
    eval_env = make_env(env_name, eval=True)
    # eval_env.metadata['render_fps'] = 30
    eval_state, _ = eval_env.reset()
    frames, done = [], False
    
    episode_return = 0
    with torch.no_grad():
        while not done:
            frames.append(eval_env.render())
            action = agent.select_action(eval_state)
            eval_state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_return += reward
    eval_env.close()
    if gif_path is not None:
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        imageio.mimsave(gif_path, frames, fps=30)
    return episode_return

def test(args, config, device):
    env = make_env(args.env_name, eval=False)
    
    if args.model_type in ALGO_LIST["Value-Based"]:       
        dqn_params = config["dqn_params"]
        dqn_params["in_channels"] = env.observation_space.shape[0]
        dqn_params["action_dim"] = env.action_space.n
        dqn_params["epsilon"] = config.get("initial_epsilon", 0.5)
        dqn_params["device"] = config.get("device", device)
        
        from model.DQN import DQN
        agent = DQN(**dqn_params)
        agent.load_ckpt(args.load_ckpt_path)
        print(f"Loaded DQN model from {args.load_ckpt_path}")
        agent.epsilon = 0.0
        
    elif args.model_type in ALGO_LIST["Policy-Based"]:
        if args.model_type == "PPO":
            ppo_params = config["ppo_params"]
            ppo_params["state_dim"] = env.observation_space.shape[0]
            ppo_params["action_dim"] = env.action_space.shape[0]
            ppo_params["device"] = config.get("device", device)
            
            from model.PPO import PPO
            agent = PPO(**ppo_params)
            agent.load_ckpt(args.load_ckpt_path)
            print(f"Loaded PPO model from {args.load_ckpt_path}")
            
        elif args.model_type == "DDPG":
            ddpg_params = config["ddpg_params"]
            ddpg_params["state_dim"] = env.observation_space.shape[0]
            ddpg_params["action_dim"] = env.action_space.shape[0]
            ddpg_params["device"] = config.get("device", device)
            
            from model.DDPG import DDPG
            agent = DDPG(**ddpg_params)
            agent.load_ckpt(args.load_ckpt_path)
            print(f"Loaded DDPG model from {args.load_ckpt_path}")
            
    gif_path = f'gif/test/{args.env_name}/{args.model_type}_{args.timestamp}.gif'
    episode_return = eval_agent(agent, args.env_name, gif_path)
    
    print(f"Tested {args.model_type} on {args.env_name}, Episode Return: {episode_return}")
            
