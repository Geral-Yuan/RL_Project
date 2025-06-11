import random
import numpy as np
import torch
import torch.nn as nn
    
class QNet(nn.Module):
    def __init__(self, in_channels, action_dim, is_cnn):
        super(QNet, self).__init__()
        self.is_cnn = is_cnn
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),  # -> (batch, 32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),                           # -> (batch, 64, 9, 9)
            nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1),                           # -> (batch, 64, 7, 7)
            # nn.ReLU()
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(in_features=64 * 7 * 7, out_features=512),
        #     nn.ReLU(),
        #     nn.Linear(in_features=512, out_features=action_dim)
        # )
        self.fc = nn.Linear(in_features=64 * 9 * 9, out_features=action_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=action_dim)
        )
        
    def forward(self, x):
        if self.is_cnn:
            x = x / 255.0
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
        return self.mlp(x)
    
class DQN:
    def __init__(self, dqn_type, is_cnn, in_channels, action_dim, epsilon, gamma, learning_rate, batch_size, update_cycle, device):
        self.dqn_type = dqn_type
        self.action_dim = action_dim
        self.QNet = QNet(in_channels, action_dim, is_cnn=is_cnn).to(device)
        self.target_QNet = QNet(in_channels, action_dim, is_cnn=is_cnn).to(device)
        self.target_QNet.load_state_dict(self.QNet.state_dict())
        self.optimizer = torch.optim.Adam(self.QNet.parameters(), lr=learning_rate)
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_cycle = update_cycle
        self.step = 0
        self.device = device
        
    def take_action(self, state):
        if isinstance(state, tuple):
            state = state[0]
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = np.array(state, dtype=np.float32).reshape(1, *state.shape)
            return self.QNet(torch.tensor(state).to(self.device)).argmax().item()
    
    def update(self, replay_buffer):
        states, actions, rewards, next_states, dones = replay_buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions).view(-1, 1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1).to(self.device)
        
        Q_value = self.QNet(states).gather(1, actions)
        if self.dqn_type == "DoubleDQN":
            next_actions = self.QNet(next_states).max(1)[1].view(-1, 1)
            max_next_Q_value = self.target_QNet(next_states).gather(1, next_actions)
        else:
            max_next_Q_value = self.target_QNet(next_states).max(1)[0].view(-1, 1)
        Q_target = rewards + self.gamma * max_next_Q_value * (1 - dones)
        loss = nn.MSELoss()(Q_value, Q_target)
        ret_loss = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.step % self.update_cycle == 0:
            self.target_QNet.load_state_dict(self.QNet.state_dict())
        
        self.step += 1
        
        return ret_loss

    