from typing import List, Dict
import numpy as np

class StabilityTracker:
    """
    Monitors system-wide memory stability and entropy to guide scheduling and optimization.
    """
    def __init__(self, entropy_threshold: float = 0.7):
        self.entropy_threshold = entropy_threshold
        self.stability_history: List[float] = []
        self.entropy_history: List[float] = []
        
    def log_state(self, stability: float, entropy: float):
        """Record current state metrics."""
        self.stability_history.append(stability)
        self.entropy_history.append(entropy)
        
        # Maintain sliding window of history
        if len(self.stability_history) > 1000:
            self.stability_history.pop(0)
            self.entropy_history.pop(0)

    def is_unstable(self) -> bool:
        """
        Check if system is in an unstable state requiring intervention.
        Returns True if entropy exceeds threshold or stability drops suddenly.
        """
        if not self.entropy_history:
            return False
            
        current_entropy = self.entropy_history[-1]
        
        # Direct threshold check
        if current_entropy > self.entropy_threshold:
            return True
            
        # Trend check: Sudden spike in entropy
        if len(self.entropy_history) >= 5:
            avg_prev = np.mean(self.entropy_history[-5:-1])
            if current_entropy > avg_prev * 1.5:  # 50% spike
                return True
                
        return False

    def get_system_health(self) -> Dict[str, float]:
        """Return aggregate health metrics."""
        if not self.stability_history:
            return {'avg_stability': 1.0, 'avg_entropy': 0.0}
            
        return {
            'avg_stability': float(np.mean(self.stability_history)),
            'avg_entropy': float(np.mean(self.entropy_history)),
            'trend': float(self.stability_history[-1] - self.stability_history[0]) if len(self.stability_history) > 1 else 0.0
        }
