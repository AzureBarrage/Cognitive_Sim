import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from src.config import MemoryConfig
from src.core.entropy_calculator import EntropyCalculator
from src.utils.logger import logger

class MemoryLayer:
    """
    A 'Smart Database' that implements the Ebbinghaus Forgetting Curve.
    Acts as a filter between the Agent and its Knowledge Base.
    """
    def __init__(self, config: MemoryConfig):
        self.initial_retention = config.initial_retention
        self.decay_rate = config.decay_rate
        self.stability_threshold = config.stability_threshold
        
        # Memory store: {memory_id: {'data': ..., 'created_at': ..., 'stability': ..., 'last_access': ...}}
        self.memories: Dict[str, Dict[str, Any]] = {}
        self.entropy_calc = EntropyCalculator()
        
        # Persistence settings
        self.persistence_path = Path("data/memory_store.json")

    def add_memory(self, memory_id: str, data: Any, initial_stability: float = 1.0):
        """Store a new memory trace."""
        current_time = time.time()
        
        # Convert tensors to list for JSON serialization if needed, 
        # or rely on a separate serializer. For now, we assume data is serializable or handled elsewhere.
        # Ideally, large data blobs (images) stay on disk, we just store the metadata here.
        
        self.memories[memory_id] = {
            'data': data, # In a real app, this might just be a pointer/filepath
            'created_at': current_time,
            'last_access': current_time,
            'stability': initial_stability,
            'access_count': 1
        }
        logger.debug(f"Memory added: {memory_id}")

    def retrieve_memory(self, memory_id: str) -> Optional[Any]:
        """
        Retrieve memory if it hasn't decayed below threshold.
        Updates stability on successful retrieval.
        """
        if memory_id not in self.memories:
            return None

        memory = self.memories[memory_id]
        retention = self._calculate_retention(memory)
        
        # BIOLOGICAL CONSTRAINT:
        # If retention is too low, the synapse has failed to fire.
        # The memory is effectively "gone" for this moment.
        if retention < 0.15: # Threshold for "Recall Failure"
            logger.info(f"Recall Failed for {memory_id} (R={retention:.2f})")
            return None

        # Update access stats (Spaced Repetition)
        self._reinforce_memory(memory_id)
        return memory['data']

    def _calculate_retention(self, memory: Dict[str, Any]) -> float:
        """
        Calculate current retention strength based on Ebbinghaus curve.
        R = e^(-decay * t / stability)
        """
        elapsed = time.time() - memory['last_access']
        # We use a scalar to make seconds relevant for the simulation speed
        # In real-time apps, this might be days.
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
        stability_gain = 0.1 * elapsed  
        
        memory['stability'] += stability_gain
        memory['last_access'] = current_time
        memory['access_count'] += 1

    def get_at_risk_memories(self, threshold: float = 0.3) -> List[str]:
        """Identify memories that are fading."""
        at_risk = []
        for mem_id, memory in self.memories.items():
            if self._calculate_retention(memory) < threshold:
                at_risk.append(mem_id)
        return at_risk

    def save_state(self):
        """Persist memory metadata to disk."""
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Deep copy to avoid mutating runtime state during serialization
        # Convert any numpy types to python native types
        serializable_memories = {}
        for k, v in self.memories.items():
            mem_copy = v.copy()
            # Handle potentially non-serializable data fields
            if 'input' in mem_copy['data'] and hasattr(mem_copy['data']['input'], 'tolist'):
                 # It's likely a Tensor/Numpy array
                 # For the 'Plug and Play' template, we skip saving heavy data in JSON
                 # We assume 'data' is lightweight or we just save metadata
                 pass
            
            serializable_memories[k] = mem_copy

        with open(self.persistence_path, 'w') as f:
            # We use a custom encoder or just exclude complex data objects for now
            # For this MVP, we assume the 'data' field is JSON safe or we sanitize it
            # To be safe, we only save metadata for the template
            clean_dump = {
                k: {
                    'created_at': v['created_at'],
                    'last_access': v['last_access'],
                    'stability': v['stability'],
                    'access_count': v['access_count'],
                    # 'data': v['data'] # Uncomment if data is text/json-safe
                } 
                for k, v in serializable_memories.items()
            }
            json.dump(clean_dump, f, indent=2)
        
        logger.info(f"Memory State saved to {self.persistence_path}")

    def load_state(self):
        """Load memory metadata from disk."""
        if not self.persistence_path.exists():
            return

        with open(self.persistence_path, 'r') as f:
            data = json.load(f)
        
        # When loading, we need to handle the 'Offline Time'
        # The system was off. Time has passed.
        # We don't change timestamps (they are absolute), 
        # but the next _calculate_retention() call will naturally see a HUGE elapsed time.
        
        self.memories = data
        logger.info(f"Memory State loaded: {len(self.memories)} items.")
