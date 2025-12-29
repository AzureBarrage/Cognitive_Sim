import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
from src.config import NetworkConfig

class CognitiveNetwork(nn.Module):
    """
    Neural network component representing cognitive processing.
    Integrates with memory layer for context-aware processing.
    """
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.output_size = config.output_size
        
        # Core processing layers
        self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        
        # Stability tracking weights (plasticity)
        self.plasticity = nn.Parameter(torch.ones(self.hidden_size))
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor, memory_context: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with optional memory context injection.
        """
        # Initial processing
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        
        # Memory integration if context provided
        if memory_context is not None:
            # Simple attention-like mechanism: modulate activations based on memory
            if memory_context.size() == x.size():
                x = x * (1.0 + torch.tanh(memory_context))
        
        # Apply plasticity modulation
        x = x * self.plasticity
        
        # Deep processing
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        
        # Output generation
        output = self.output_layer(x)
        return output

    def get_layer_weights(self) -> List[Any]:
        """Extract weights for entropy calculation."""
        return [
            self.layer1.weight.detach().numpy(),
            self.layer2.weight.detach().numpy(),
            self.output_layer.weight.detach().numpy()
        ]
    
    def update_plasticity(self, error_signal: float):
        """
        Update neural plasticity based on performance/error.
        Higher error -> higher plasticity (learning rate) needed.
        """
        with torch.no_grad():
            adjustment = torch.sigmoid(torch.tensor(error_signal)) * 0.1
            self.plasticity.add_(adjustment)
            # Clip to reasonable range
            self.plasticity.clamp_(0.5, 2.0)
