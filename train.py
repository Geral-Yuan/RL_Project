import numpy as np
from tqdm import tqdm
from time import time
import swanlab

from utils import *
from test import eval_agent

def train_off_policy_agent(args, config, env, agent):
    if args.store_gif:
        video_dir, vis_interval = setup_visualization(args, config)
    return_list = []
    replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])
    pbar = tqdm(range(config["num_episodes"]), desc=f"Training {args.model_type} on {args.env_name}", unit="episodes")
    start_time = time()
    for i in pbar:
        state, _ = env.reset()
        done = False
        episode_return = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            loss = None
            if replay_buffer.size() > config["training_start_size"]:
                loss = agent.update(replay_buffer)

        return_list.append(episode_return)
        avg_return = np.mean(return_list[-50:])
        pbar.set_postfix({'return': '%.3f' % episode_return})
        # if args.use_wandb:
        #     wandb.log({"epsilon": agent.epsilon, "return": episode_return, "avg_return": avg_return})
        if args.use_swanlab:
            if "epsilon_decay" in config:
                swanlab.log({"epsilon": agent.epsilon, "return": episode_return, "avg_return": avg_return, "loss": loss})
            else:
                swanlab.log({"return": episode_return, "avg_return": avg_return})
            # elif "tau_decay" in config:
            #     swanlab.log({"tau": agent.tau, "return": episode_return, "avg_return": avg_return})
            

        if "epsilon_decay" in config:
            agent.epsilon = max(config["min_epsilon"], agent.epsilon * config["epsilon_decay"])
        # elif "tau_decay" in config:
        #     agent.tau = max(config["min_tau"], agent.tau * config["tau_decay"],)

        if args.store_gif and (i + 1) % vis_interval == 0:
            gif_path = video_dir / f"ep{i+1:04d}.gif"
            eval_agent(agent, args.env_name, gif_path=gif_path)
            
    end_time = time()
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")
    
    last_100 = np.array(return_list[-100:])

    max_return = np.max(last_100)
    min_return = np.min(last_100)
    mean = np.mean(last_100)
    std = np.std(last_100)

    print("Last 100 episodes statistics:")
    print("Max Return:", max_return)
    print("Min Return:", min_return)
    print("Mean:", mean)
    print("Standard Deviation:", std)
    
    if args.save_ckpt:
        if args.save_ckpt_path is None:
            args.save_ckpt_path = f"ckpt/{args.env_name}/{args.model_type}.pt"
        agent.save_ckpt(args.save_ckpt_path)
        print(f"Checkpoint of {args.model_type} on {args.env_name} saved to {args.save_ckpt_path}")
        
    return return_list

def train_DQN(args, config, device):
    env = make_env(args.env_name, eval=False)        
    dqn_params = config["dqn_params"]
    dqn_params["in_channels"] = env.observation_space.shape[0]
    dqn_params["action_dim"] = env.action_space.n
    dqn_params["epsilon"] = config.get("initial_epsilon", 0.5)
    dqn_params["device"] = config.get("device", device)
    
    from model.DQN import DQN
    agent = DQN(**dqn_params)
    
    return_list = train_off_policy_agent(args, config, env, agent)
    
    env.close()
    
    return return_list
    

def train_PPO(args, config, device):
    env = make_env(args.env_name, eval=False)
    
    ppo_params = config["ppo_params"]
    ppo_params["state_dim"] = env.observation_space.shape[0]
    ppo_params["action_dim"] = env.action_space.shape[0]
    ppo_params["device"] = config.get("device", device)
    
    from model.PPO import PPO
    agent = PPO(**ppo_params)
    
    if args.store_gif:
        video_dir, vis_interval = setup_visualization(args, config)
    
    return_list = []
    pbar = tqdm(range(config["num_episodes"]), desc=f"Training {args.model_type} on {args.env_name}", unit="episode")
    start_time = time()
    for i in pbar:
        episode_return = 0
        states, actions, rewards, next_states, dones = [], [], [], [], []
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            state = next_state
            episode_return += reward
        actor_loss, critic_loss = agent.update(states, actions, rewards, next_states, dones)
        
        return_list.append(episode_return)
        avg_return = np.mean(return_list[-50:])
        pbar.set_postfix({'return': '%.3f' % episode_return})
        # if args.use_wandb:
        #     wandb.log({"return": episode_return, "avg_return": avg_return})
        if args.use_swanlab:
            swanlab.log({"return": episode_return, "avg_return": avg_return, "actor_loss": actor_loss, "critic_loss": critic_loss})
        
        if args.store_gif and (i + 1) % vis_interval == 0:
            gif_path = video_dir / f"ep{i+1:04d}.gif"
            eval_agent(agent, args.env_name, gif_path=gif_path)
        
    end_time = time()
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")
    
    last_100 = np.array(return_list[-100:])

    max_return = np.max(last_100)
    min_return = np.min(last_100)
    mean = np.mean(last_100)
    std = np.std(last_100)

    print("Last 100 episodes statistics:")
    print("Max Return:", max_return)
    print("Min Return:", min_return)
    print("Mean:", mean)
    print("Standard Deviation:", std)
    
    if args.save_ckpt:
        if args.save_ckpt_path is None:
            args.save_ckpt_path = f"ckpt/{args.env_name}/{args.model_type}.pt"
        agent.save_ckpt(args.save_ckpt_path)
    
    env.close()
    
    return return_list
    
def train_DDPG(args, config, device):
    env = make_env(args.env_name, eval=False)
    
    ddpg_params = config["ddpg_params"]
    ddpg_params["state_dim"] = env.observation_space.shape[0]
    ddpg_params["action_dim"] = env.action_space.shape[0]
    ddpg_params["device"] = config.get("device", device)
    
    from model.DDPG import DDPG
    agent = DDPG(**ddpg_params)
    return_list = train_off_policy_agent(args, config, env, agent)
    
    env.close()
    
    return return_list