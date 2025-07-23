import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .base_agent import BaseAgent

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        
        # Actor head
        self.actor = nn.Linear(128, action_dim)
        
        # Critic head
        self.critic = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

class PPOAgent(BaseAgent):
    """Proximal Policy Optimization (PPO) agent implementation."""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2, entropy_coef=0.01):
        super().__init__(state_dim, action_dim)
        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.memory = []
        
    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.model(state)
            dist = torch.distributions.Categorical(action_probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            
            # Map index to action parameters
            action_type = action_idx.item()
            action_params = []
            
            # Define action parameters based on action type
            if action_type == 0:  # No-op
                pass
            elif 1 <= action_type <= 4:  # Add gear
                action_params = [np.random.uniform(0, 1) for _ in range(4)]  # x, y, radius, teeth
            elif 5 <= action_type <= 6:  # Remove gear
                action_params = [np.random.randint(0, 10)]  # gear index
            elif 7 <= action_type <= 9:  # Adjust position
                action_params = [np.random.randint(0, 10), np.random.uniform(0, 1), np.random.uniform(0, 1)]  # index, x, y
            elif 10 <= action_type <= 12:  # Change size
                action_params = [np.random.randint(0, 10), np.random.uniform(0, 1), np.random.randint(8, 32)]  # index, radius, teeth
            
            # Combine action type and parameters
            action = [action_type] + action_params
            
        return action, log_prob.item()
    
    def update(self, states, actions, rewards, next_states, dones):
        # Convert to tensors - use single numpy array conversion for efficiency
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)
        
        # Calculate advantages
        with torch.no_grad():
            _, current_values = self.model(states)
            _, next_values = self.model(next_states)
            td_target = rewards + (1 - dones) * self.gamma * next_values
            advantages = td_target - current_values
        
        # Get current policy
        action_probs, values = self.model(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        
        # PPO loss calculation
        ratios = torch.exp(log_probs - log_probs.detach())
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Critic loss - reshape td_target to match values shape
        critic_loss = F.mse_loss(values, td_target.detach().view(-1, 1))
        
        # Entropy bonus
        entropy = dist.entropy().mean()
        
        # Total loss
        loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
