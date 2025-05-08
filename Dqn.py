import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# 定义优先经验回放
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # 优先级的幂，确定采样概率
        self.beta = beta    # 重要性采样的幂
        self.beta_increment = beta_increment  # beta随时间增加
        self.buffer = deque(maxlen=capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done):
        """添加新经验"""
        max_priority = np.max(self.priorities) if self.size > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """基于优先级采样"""
        if self.size < batch_size:
            indices = range(self.size)
        else:
            priorities = self.priorities[:self.size]
            probabilities = priorities ** self.alpha
            probabilities /= np.sum(probabilities)
            indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # 计算重要性采样权重
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= np.max(weights)
        self.beta = min(1.0, self.beta + self.beta_increment)  # 增加beta
        
        states = np.zeros((batch_size, *self.buffer[0][0].shape))
        next_states = np.zeros((batch_size, *self.buffer[0][3].shape))
        actions, rewards, dones = [], [], []
        
        for i, idx in enumerate(indices):
            state, action, reward, next_state, done = self.buffer[idx]
            states[i] = state
            next_states[i] = next_state
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, errors):
        """更新优先级"""
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error + 1e-5  # 添加小值防止优先级为0

# 定义 Q 网络模型
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 修改DQN智能体
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 超参数调整
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.05  # 最小探索率
        self.epsilon_decay = 0.997  # 衰减速度
        self.learning_rate = 0.0005  # 学习率
        self.update_target_freq = 50  # 更频繁地更新目标网络
        
        # 使用优先经验回放
        self.memory = PrioritizedReplayBuffer(capacity=10000)
        
        # 创建更深/更宽的Q网络
        self.q_network = self._build_model().to(self.device)
        self.target_network = self._build_model().to(self.device)
        self.update_target_network()
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss(reduction='none')  # 使用无缩减损失函数计算单独样本的损失
        
        # 训练步数计数器
        self.train_step_counter = 0
    
    def _build_model(self):
        """构建更复杂的Q网络"""
        model = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        return model
    
    def update_target_network(self):
        """更新目标网络权重"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.memory.add(state, action, reward, next_state, done)
    
    def act(self, state, valid_actions, epsilon=None):
        """选择动作"""
        if epsilon is None:
            epsilon = self.epsilon  # 使用默认值
        
        if np.random.rand() <= epsilon:
            return np.random.choice(valid_actions)
        
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state).cpu().numpy()[0]
            # 仅考虑有效动作
            valid_q_values = {a: q_values[a] for a in valid_actions}
            return max(valid_q_values, key=valid_q_values.get)
    
    def replay(self, batch_size):
        """从经验回放中采样并训练网络"""
        # 采样批次及重要性权重
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(batch_size)
        
        # 转换为张量
        states_tensor = torch.FloatTensor(states).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        
        # 计算当前Q值
        q_values = self.q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值 (使用Double DQN)
        with torch.no_grad():
            # 从在线网络选择动作
            best_actions = self.q_network(next_states_tensor).argmax(dim=1)
            # 使用目标网络计算Q值
            next_q_values = self.target_network(next_states_tensor).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            targets = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values
        
        # 计算TD误差用于更新优先级
        td_errors = torch.abs(q_values - targets).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)
        
        # 计算加权损失
        losses = self.criterion(q_values, targets)
        weighted_loss = (losses * weights_tensor).mean()
        
        # 更新网络
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 定期更新目标网络
        self.train_step_counter += 1
        if self.train_step_counter % self.update_target_freq == 0:
            self.update_target_network()
        
        return weighted_loss.item()