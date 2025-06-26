import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DynamicsNet(nn.Module):
    """
    Neural network for learning cancer dynamics - exact copy from original code
    """
    
    def __init__(self, input_size=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
        
    def forward(self, x):
        return self.network(x)


class DynamicsDataset(Dataset):
    """
    Dataset class for cancer dynamics data - exact copy from original code
    """
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AdaptiveDynamicsNet(nn.Module):
    """
    Advanced neural network with adaptive architecture for fractional dynamics
    """
    
    def __init__(self, input_size=4, hidden_sizes=[64, 32], output_size=4, 
                 dropout_rate=0.1, activation='relu'):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()
            
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self.activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
            
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x):
        return self.network(x)
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class FractionalDynamicsNet(nn.Module):
    """
    Specialized neural network for fractional dynamics with memory mechanism
    """
    
    def __init__(self, input_size=4, hidden_sizes=[64, 32], output_size=4, 
                 memory_size=10, alpha=1.0):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.memory_size = memory_size
        self.alpha = alpha
        
        # Main dynamics network
        self.dynamics_net = AdaptiveDynamicsNet(
            input_size, hidden_sizes, output_size
        )
        
        # Memory processing network
        self.memory_net = nn.Sequential(
            nn.Linear(input_size * memory_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )
        
        # Fractional weighting network
        self.weight_net = nn.Sequential(
            nn.Linear(1, 16),  # Input: alpha value
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # Weight between 0 and 1
        )
        
    def forward(self, x, memory_states=None, alpha_tensor=None):
        """
        Forward pass with optional memory and alpha weighting
        
        Args:
            x: Current state [batch_size, input_size]
            memory_states: Historical states [batch_size, memory_size, input_size]
            alpha_tensor: Alpha values [batch_size, 1]
        """
        # Standard dynamics prediction
        dynamics_output = self.dynamics_net(x)
        
        if memory_states is not None and alpha_tensor is not None:
            # Process memory
            batch_size = x.size(0)
            memory_flat = memory_states.view(batch_size, -1)
            memory_output = self.memory_net(memory_flat)
            
            # Get fractional weight
            frac_weight = self.weight_net(alpha_tensor)
            
            # Combine dynamics and memory
            output = (1 - frac_weight) * dynamics_output + frac_weight * memory_output
        else:
            output = dynamics_output
            
        return output


class EnsembleDynamicsNet(nn.Module):
    """
    Ensemble of neural networks for robust predictions
    """
    
    def __init__(self, input_size=4, num_networks=3, hidden_sizes=[64, 32], 
                 output_size=4):
        super().__init__()
        
        self.num_networks = num_networks
        
        # Create ensemble of networks
        self.networks = nn.ModuleList([
            AdaptiveDynamicsNet(input_size, hidden_sizes, output_size)
            for _ in range(num_networks)
        ])
        
        # Weighting network for ensemble combination
        self.weight_net = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, num_networks),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        """Forward pass through ensemble"""
        # Get predictions from all networks
        predictions = []
        for net in self.networks:
            pred = net(x)
            predictions.append(pred.unsqueeze(2))
            
        # Stack predictions [batch_size, output_size, num_networks]
        stacked_preds = torch.cat(predictions, dim=2)
        
        # Get ensemble weights [batch_size, num_networks]
        weights = self.weight_net(x)
        
        # Weighted combination [batch_size, output_size]
        output = torch.sum(stacked_preds * weights.unsqueeze(1), dim=2)
        
        return output, weights


def create_network(network_type='standard', **kwargs):
    """
    Factory function to create different types of neural networks
    
    Args:
        network_type (str): Type of network ('standard', 'adaptive', 'fractional', 'ensemble')
        **kwargs: Additional arguments for network creation
        
    Returns:
        Neural network instance
    """
    if network_type == 'standard':
        return DynamicsNet(kwargs.get('input_size', 4))
    elif network_type == 'adaptive':
        return AdaptiveDynamicsNet(**kwargs)
    elif network_type == 'fractional':
        return FractionalDynamicsNet(**kwargs)
    elif network_type == 'ensemble':
        return EnsembleDynamicsNet(**kwargs)
    else:
        raise ValueError(f"Unknown network type: {network_type}")


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model):
    """Get detailed information about a model"""
    info = {
        'type': model.__class__.__name__,
        'parameters': count_parameters(model),
        'architecture': str(model)
    }
    return info
    