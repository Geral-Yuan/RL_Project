import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_scale):
        super(Actor, self).__init__()
        self.action_scale = action_scale
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_scale
    
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
class DDPG:
    def __init__(self, state_dim, action_dim, hidden_dim=64, actor_lr=1e-4, critic_lr=1e-3, action_scale=1.0, sigma=0.01, tau=0.005, gamma=0.98, batch_size=32, device='cpu'):
        self.actor = Actor(state_dim, hidden_dim, action_dim, action_scale).to(device)
        self.critic = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, hidden_dim, action_dim, action_scale).to(device)
        self.target_critic = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.action_dim = action_dim
        self.sigma = sigma
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        
    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = self.actor(state).cpu().detach().numpy()
        action += np.random.randn(self.action_dim) * self.sigma
        return np.clip(action, -self.actor.action_scale, self.actor.action_scale)
    
    def soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
    def update(self, replay_buffer):
        states, actions, rewards, next_states, dones = replay_buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1).to(self.device)

        next_Q = self.target_critic(next_states, self.target_actor(next_states))
        Q_targets = rewards + (1 - dones) * self.gamma * next_Q
        critic_loss = F.mse_loss(self.critic(states, actions), Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)
