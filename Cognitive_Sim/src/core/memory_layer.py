import time
import numpy as np
from typing import Dict, Any, Optional
from src.config import MemoryConfig
from src.core.entropy_calculator import EntropyCalculator

class MemoryLayer:
    """
    Implements memory retention mechanics using Ebbinghaus forgetting curve
    and entropy-based stability tracking.
    """
    def __init__(self, config: MemoryConfig):
        self.initial_retention = config.initial_retention
        self.decay_rate = config.decay_rate
        self.stability_threshold = config.stability_threshold
        
        # Memory store: {memory_id: {'data': ..., 'created_at': ..., 'stability': ..., 'last_access': ...}}
        self.memories: Dict[str, Dict[str, Any]] = {}
        self.entropy_calc = EntropyCalculator()

    def add_memory(self, memory_id: str, data: Any, initial_stability: float = 1.0):
        """Store a new memory trace."""
        current_time = time.time()
        self.memories[memory_id] = {
            'data': data,
            'created_at': current_time,
            'last_access': current_time,
            'stability': initial_stability,
            'access_count': 1
        }

    def retrieve_memory(self, memory_id: str) -> Optional[Any]:
        """
        Retrieve memory if it hasn't decayed below threshold.
        Updates stability on successful retrieval.
        """
        if memory_id not in self.memories:
            return None

        memory = self.memories[memory_id]
        retention = self._calculate_retention(memory)
        
        # If retention is too low, memory might be "forgotten" or corrupted
        if retention < 0.1:  # Hard threshold for complete forgetting
            return None

        # Update access stats
        self._reinforce_memory(memory_id)
        return memory['data']

    def _calculate_retention(self, memory: Dict[str, Any]) -> float:
        """
        Calculate current retention strength based on Ebbinghaus curve.
        R = e^(-decay * t / stability)
        """
        elapsed = time.time() - memory['last_access']
        # Convert to relevant time unit (e.g., simulated hours/days) - simplifying to seconds for now
        return np.exp(-(self.decay_rate * elapsed) / memory['stability'])

    def _reinforce_memory(self, memory_id: str):
        """
        Strengthen memory stability upon access (Spaced Repetition principle).
        """
        memory = self.memories[memory_id]
        current_time = time.time()
        elapsed = current_time - memory['last_access']
        
        # Stability increase depends on difficulty of retrieval (time passed)
        # S_new = S_old * (1 + factor)
        stability_gain = 0.1 * elapsed  # Simplified gain model
        memory['stability'] += stability_gain
        memory['last_access'] = current_time
        memory['access_count'] += 1

    def get_at_risk_memories(self, threshold: float = 0.3) -> list[str]:
        """
        Identify memories that are below the stability threshold.
        """
        at_risk = []
        for mem_id, memory in self.memories.items():
            if self._calculate_retention(memory) < threshold:
                at_risk.append(mem_id)
        return at_risk

    def get_memory_health(self, memory_id: str) -> Dict[str, float]:
        """Get diagnostics for a specific memory."""
        if memory_id not in self.memories:
            return {}
            
        memory = self.memories[memory_id]
        retention = self._calculate_retention(memory)
        elapsed = time.time() - memory['last_access']
        
        uncertainty = self.entropy_calc.calculate_memory_uncertainty(
            memory['stability'], self.decay_rate, elapsed
        )
        
        return {
            'retention': float(retention),
            'stability': float(memory['stability']),
            'uncertainty': uncertainty,
            'age': time.time() - memory['created_at']
        }
