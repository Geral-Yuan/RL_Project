import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_scale):
        super(Actor, self).__init__()
        self.action_scale = action_scale
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) * self.action_scale
    
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class DDPG:
    def __init__(self, state_dim, action_dim, hidden_dim=64, actor_lr=1e-4, critic_lr=1e-3, action_scale=1.0, sigma=0.01, tau=0.01, gamma=0.98, batch_size=32, delay_policy_update=2, delay_target_update=20, policy_noise=0.2, noise_clip=0.5, device='cpu'):
        self.actor = Actor(state_dim, hidden_dim, action_dim, action_scale).to(device)
        self.critic = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, hidden_dim, action_dim, action_scale).to(device)
        self.target_critic = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.sigma = sigma
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.delay_policy_update = delay_policy_update
        self.delay_target_update = delay_target_update
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.device = device
        self.update_count = 0
        
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = self.actor(state).cpu().detach().numpy()
        action += np.random.randn(self.action_dim) * self.sigma
        return np.clip(action, -self.actor.action_scale, self.actor.action_scale)
    
    def soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
    def update(self, replay_buffer):
        self.update_count += 1
        states, actions, rewards, next_states, dones = replay_buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1).to(self.device)

        with torch.no_grad():
            next_action = self.target_actor(next_states)
            
            # Add clipped Gaussian noise
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-self.action_scale, self.action_scale)

            # Target Q value using smoothed action
            next_Q = self.target_critic(next_states, next_action)
            Q_targets = rewards + (1 - dones) * self.gamma * next_Q

        # Update critic
        critic_loss = F.mse_loss(self.critic(states, actions), Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        if self.update_count % self.delay_policy_update == 0:
            # Update actor
            actor_loss = -self.critic(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        
        if self.update_count % self.delay_target_update == 0:
            self.soft_update(self.actor, self.target_actor)
            self.soft_update(self.critic, self.target_critic)

    def save_ckpt(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)
        
    def load_ckpt(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])