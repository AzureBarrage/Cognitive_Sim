from typing import Dict, List, Any
import time
import numpy as np

class MetricsTracker:
    """
    Tracks and stores metrics for the simulation.
    """
    def __init__(self):
        self.metrics: Dict[str, List[Dict[str, Any]]] = {
            "training": [],
            "validation": [],
            "system": []
        }
        self.start_time = time.time()

    def log_training_step(self, step: int, loss: float, entropy: float, plasticity: float):
        """Logs metrics for a single training step."""
        self.metrics["training"].append({
            "step": step,
            "loss": loss,
            "entropy": entropy,
            "plasticity": plasticity,
            "timestamp": time.time() - self.start_time
        })

    def log_system_health(self, step: int, memory_usage: float, cpu_usage: float):
        """Logs system health metrics."""
        self.metrics["system"].append({
            "step": step,
            "memory_usage": memory_usage,
            "cpu_usage": cpu_usage,
            "timestamp": time.time() - self.start_time
        })

    def get_latest(self, category: str) -> Dict[str, Any]:
        """Returns the most recent metric for a category."""
        if category in self.metrics and self.metrics[category]:
            return self.metrics[category][-1]
        return {}

    def get_history(self, category: str) -> List[Dict[str, Any]]:
        """Returns the full history for a category."""
        return self.metrics.get(category, [])
    
    def calculate_average_loss(self, window: int = 100) -> float:
        """Calculates the moving average of training loss."""
        if not self.metrics["training"]:
            return 0.0
        
        losses = [m["loss"] for m in self.metrics["training"][-window:]]
        return float(np.mean(losses))
