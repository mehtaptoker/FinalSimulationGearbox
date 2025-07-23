import abc
import torch
import torch.nn as nn

class BaseAgent(abc.ABC):
    """Abstract base class for reinforcement learning agents."""
    
    def __init__(self, state_dim, action_dim):
        """
        Initialize the agent.
        
        Args:
            state_dim (int): Dimensionality of the state space
            action_dim (int): Dimensionality of the action space
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @abc.abstractmethod
    def act(self, state):
        """
        Select an action given the current state.
        
        Args:
            state (torch.Tensor): Current state of the environment
            
        Returns:
            action (torch.Tensor): Selected action
        """
        pass
    
    @abc.abstractmethod
    def update(self, states, actions, rewards, next_states, dones):
        """
        Update the agent's parameters based on collected experiences.
        
        Args:
            states (list): List of states encountered
            actions (list): List of actions taken
            rewards (list): List of rewards received
            next_states (list): List of next states
            dones (list): List of termination flags
        """
        pass
    
    @abc.abstractmethod
    def save(self, path):
        """Save the agent's model to the specified path."""
        pass
    
    @abc.abstractmethod
    def load(self, path):
        """Load the agent's model from the specified path."""
        pass
