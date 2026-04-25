import time
import numpy as np
from typing import Union, List
from src.config import MemoryConfig

class EntropyCalculator:
    """
    Calculates entropy and uncertainty metrics for the cognitive simulation.
    Based on Shannon Entropy: H(X) = -Σ P(x) log P(x)
    """
    
    @staticmethod
    def calculate_weight_entropy(weights: Union[np.ndarray, List[float]]) -> float:
        """
        Calculate Shannon entropy of weight distribution.
        
        Args:
            weights: Array of weights/probabilities
            
        Returns:
            float: Calculated entropy value
        """
        w = np.array(weights)
        # Normalize weights to get probability distribution
        if np.sum(w) == 0:
            return 0.0
        
        # Ensure non-negative
        w = np.abs(w)
        probs = w / np.sum(w)
        
        # Avoid log(0)
        mask = probs > 0
        entropy = -np.sum(probs[mask] * np.log2(probs[mask]))
        
        return float(entropy)
    
    @staticmethod
    def calculate_memory_uncertainty(stability: float, decay_rate: float, time_elapsed: float) -> float:
        """
        Calculate uncertainty in memory retention based on Ebbinghaus forgetting curve.
        
        R = e^(-t/S) where R is retention, t is time, S is stability
        Uncertainty = 1 - R
        
        Args:
            stability: Memory stability factor
            decay_rate: Base decay rate
            time_elapsed: Time since last reinforcement
            
        Returns:
            float: Uncertainty score (0-1)
        """
        if stability <= 0:
            return 1.0
            
        # Modified Ebbinghaus: R = e^(-decay * t / stability)
        retention = np.exp(-(decay_rate * time_elapsed) / stability)
        uncertainty = 1.0 - retention
        
        return float(np.clip(uncertainty, 0.0, 1.0))

    @staticmethod
    def calculate_system_entropy(layer_states: List[np.ndarray]) -> float:
        """
        Calculate total system entropy across multiple layers.
        """
        total_entropy = 0.0
        for layer in layer_states:
            total_entropy += EntropyCalculator.calculate_weight_entropy(layer)
        return total_entropy / len(layer_states) if layer_states else 0.0
