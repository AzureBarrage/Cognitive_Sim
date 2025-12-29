import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from src.config import NetworkConfig
from src.core.entropy_calculator import EntropyCalculator

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
        self.entropy_threshold = config.entropy_threshold
        
        # Core processing layers
        self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        
        # Stability tracking weights (plasticity)
        self.plasticity = nn.Parameter(torch.ones(self.hidden_size))
        
        self.dropout = nn.Dropout(0.2)
        
    def calculate_uncertainty(self) -> float:
        """
        Calculate current network uncertainty based on weight entropy.
        
        Returns:
            float: Uncertainty score (higher means more uncertain)
        """
        weights = self.get_layer_weights()
        # Flatten and concatenate weights for a holistic view
        flat_weights = []
        for w in weights:
            flat_weights.extend(w.flatten())
            
        return EntropyCalculator.calculate_weight_entropy(flat_weights)

    def forward(self, x: torch.Tensor, memory_context: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with optional memory context injection.
        Returns output and meta-cognitive state.
        """
        # Meta-cognitive check
        uncertainty = self.calculate_uncertainty()
        meta_state = {
            "uncertainty": uncertainty,
            "high_uncertainty": uncertainty > self.entropy_threshold
        }

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
        
        # If highly uncertain, dampen output confidence (simulating hesitation)
        if meta_state["high_uncertainty"]:
            output = output * 0.8
            
        return output, meta_state

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

    def save_weights(self, path: str):
        """Save network weights to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str):
        """Load network weights from disk."""
        path = Path(path)
        if path.exists():
            self.load_state_dict(torch.load(path))

