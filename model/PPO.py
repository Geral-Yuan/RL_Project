import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Policy Network for PPO algorithm.
    To handle continuous action space, the network outputs mean and std of a Gaussian distribution.
    """
    def __init__(self, state_dim, hidden_dim, action_dim, action_scale):
        super(Actor, self).__init__()
        self.action_scale = action_scale
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.mean = nn.Linear(hidden_dim*2, action_dim) # predict mean
        self.std = nn.Linear(hidden_dim*2, action_dim) # predict std
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale
        std = F.softplus(self.std(x)) + 1e-3    # std > 0
        return mean, std
    

class Critic(nn.Module):
    """
    Value Network for PPO algorithm.
    """
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1) # predict value
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        value = self.fc2(x)
        return value


class PPO:
    def __init__(self, state_dim, action_dim, hidden_dim=64, num_iter=10, actor_lr=1e-4, critic_lr=1e-3, action_scale=1.0, gamma=0.9, epsilon=0.2, device='cpu'):
        self.actor = Actor(state_dim, hidden_dim, action_dim, action_scale).to(device)
        self.critic = Critic(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.num_iter = num_iter
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device
        
    def select_action(self, state):
        if isinstance(state, tuple):
            state = state[0]
            
        state = torch.FloatTensor(state).to(self.device)
        mean, std = self.actor(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action.cpu().detach().numpy()
    
    
    def gae(self, states, rewards, next_states, dones):
        values = self.critic(states).squeeze(-1)
        next_values = self.critic(next_states).squeeze(-1)

        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        gae = 0

        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values[t] * mask - values[t]
            gae = delta + self.gamma * self.lamda * mask * gae
            advantages[t] = gae

        returns = advantages + values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns.detach(), advantages.detach()
        
    
    def update(self, states, actions, rewards, next_states, dones):
        # print(states)
        states = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(self.device)
        actions = torch.from_numpy(np.asarray(actions, dtype=np.float32)).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.from_numpy(np.asarray(next_states, dtype=np.float32)).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1).to(self.device)
        
        # Compute advantages and returns
        returns = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        advantages = (returns - self.critic(states)).detach()
        # returns, advantages = self.gae(states, rewards, next_states, dones)
        
        with torch.no_grad():
            mean, std = self.actor(states)
            dist = torch.distributions.Normal(mean, std)
            old_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            
        actor_loss_list = []
        critic_loss_list = []
        
        for _ in range(self.num_iter):
            mean, std = self.actor(states)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            
            ratio = (new_log_probs - old_log_probs).exp()
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            actor_loss_list.append(actor_loss.item())
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            critic_loss = nn.MSELoss()(self.critic(states), returns.detach())
            critic_loss_list.append(critic_loss.item())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        
        return np.mean(actor_loss_list), np.mean(critic_loss_list)
        
