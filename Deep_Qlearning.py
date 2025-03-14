import grid2op
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

#import grid2op.plotting.plotly_graph as gp

# 1. Create environment Grid2Op

env = grid2op.make("rte_case14_redisp", test=True)
state_size = env.observation_space.shape[0]  # state set
action_size = env.action_space.dim  # action set


#from grid2op.Parameters import Parameters

#gp.plot_obs(env, env.get_obs()).show()

# 2. Neuronal network for DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 3. Agent DQN
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Factor de descuento
        self.epsilon = 1.0  # Tasa de exploración inicial
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_values = self.model(state_tensor)
        return torch.argmax(action_values).item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(torch.FloatTensor(next_state).unsqueeze(0))).item()
            
            output = self.model(torch.FloatTensor(state).unsqueeze(0))[0, action]
            loss = self.criterion(output, torch.tensor(target))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 4. Training the agent DQN
agent = DQNAgent(state_size, action_size)
n_episodes = 500
batch_size = 32

for episode in range(n_episodes):
    state = env.reset()
    state = state.to_vect()  # Convertir a vector numérico
    total_reward = 0
    done = False
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(env.action_space.get_action(action))
        next_state = next_state.to_vect()
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    
    agent.replay(batch_size)
    print(f"Episode {episode+1}/{n_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")
    
#gp.plot_obs(env, env.get_obs()).show()
env.close()
