import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)

class Critic(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

class A2CAgent:
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_size, action_size).to(self.device)
        self.critic = Critic(state_size).to(self.device)
        self.optimizerA = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizerC = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state, valid_actions):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.actor(state).detach().cpu().numpy()[0]
        mask = np.zeros_like(probs)
        mask[valid_actions] = 1
        masked_probs = probs * mask
        if masked_probs.sum() == 0:
            masked_probs[valid_actions] = 1.0
        masked_probs /= masked_probs.sum()
        action = np.random.choice(len(masked_probs), p=masked_probs)
        return action, masked_probs[action]

    def update(self, trajectory):
        total_actor_loss = 0
        total_critic_loss = 0
        # trajectory: [(state, action, reward, next_state, done, valid_actions), ...]
        for i, (state, action, reward, next_state, done, valid_actions) in enumerate(trajectory):
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            value = self.critic(state_t)
            next_value = self.critic(next_state_t)
            next_value = next_value.squeeze()  # 修正：变成 shape [1]
            target = torch.tensor([reward], dtype=torch.float32, device=self.device)
            if not done:
                target += self.gamma * next_value
            advantage = target - value.squeeze()

            # 更新Actor
            probs = self.actor(state_t)
            log_prob = torch.log(probs[0, action] + 1e-8)
            actor_loss = -log_prob * advantage.detach()
            self.optimizerA.zero_grad()
            actor_loss.backward()
            self.optimizerA.step()

            # 更新Critic
            critic_loss = advantage.pow(2)
            self.optimizerC.zero_grad()
            critic_loss.backward()
            self.optimizerC.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
        avg_loss = (total_actor_loss + total_critic_loss) / (2 * len(trajectory))
        return avg_loss