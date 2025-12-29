import torch
import torch.optim as optim
from typing import Dict, Optional
import numpy as np

class CognitiveOptimizer:
    """
    Custom optimizer wrapper that adapts learning rates based on 
    memory stability and system entropy.
    Implements Spaced Repetition Optimization principles.
    """
    def __init__(self, model_parameters, base_lr: float = 0.01):
        self.base_lr = base_lr
        self.optimizer = optim.Adam(model_parameters, lr=base_lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

    def step(self, system_entropy: float = 0.0, memory_stability: float = 1.0):
        """
        Perform optimization step with adaptive learning rate.
        
        Args:
            system_entropy: Current entropy of the system (uncertainty)
            memory_stability: Average stability of active memories
        """
        # Adapt learning rate based on cognitive state
        # High entropy (uncertainty) -> Higher learning rate needed (exploration)
        # High stability -> Lower learning rate needed (exploitation/refinement)
        
        adaptive_factor = (1.0 + system_entropy) / (1.0 + memory_stability)
        current_lr = self.base_lr * adaptive_factor
        
        # Clamp LR to prevent instability
        current_lr = np.clip(current_lr, 1e-5, 0.1)
        
        # Update optimizer params
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
            
        self.optimizer.step()

    def zero_grad(self):
        """Clear gradients."""
        self.optimizer.zero_grad()

    def update_scheduler(self, metric: float):
        """Update internal scheduler based on performance metric (e.g. loss)."""
        self.scheduler.step(metric)

    def get_current_lr(self) -> float:
        """Return current learning rate."""
        return self.optimizer.param_groups[0]['lr']
