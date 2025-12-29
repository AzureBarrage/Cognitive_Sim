from typing import Dict, Any, Tuple, List, Optional
import torch
import numpy as np
from src.core.memory_layer import MemoryLayer
from src.core.network import CognitiveNetwork
from src.core.optimizer import CognitiveOptimizer
from src.utils.logger import logger

class CognitiveAgent:
    """
    The Agentic Model that interacts with the environment.
    Manages energy, compute cycles, and makes decisions on learning vs reviewing.
    """
    def __init__(self, 
                 memory: MemoryLayer, 
                 network: CognitiveNetwork, 
                 optimizer: CognitiveOptimizer,
                 initial_energy: float = 100.0):
        self.memory = memory
        self.network = network
        self.optimizer = optimizer
        
        # Agent State
        self.energy = initial_energy
        self.compute_cycles_spent = 0.0
        self.total_rewards = 0.0
        
        # Configuration (could be moved to config file)
        self.review_threshold = 0.4  # Stability threshold to trigger review
        self.energy_cost_learn = 5.0
        self.energy_cost_review = 2.0
        self.energy_reward_correct = 10.0
        self.compute_cost_relearn = 15.0 # High cost if forgot

    def decide_strategy(self) -> str:
        """
        Decide whether to 'learn_new' or 'review'.
        Strategy: If there are memories at risk of decaying, prioritize review.
        """
        # Check for decaying memories
        # We need a way to scan memory health without retrieving (and altering) everything.
        # For now, we assume the memory layer can provide a list of 'at-risk' IDs.
        at_risk_memories = self.memory.get_at_risk_memories(threshold=self.review_threshold)
        
        if len(at_risk_memories) > 0:
            logger.info(f"Agent decided to REVIEW ({len(at_risk_memories)} items at risk)")
            return "review"
        
        if self.energy < self.energy_cost_learn:
             logger.info("Agent too tired to learn new, forced to rest/review (or potentially fail)")
             # In a real survival sim, this might be death or forced rest. 
             # For now, we'll review if possible or do nothing.
             return "review"

        logger.info("Agent decided to LEARN NEW")
        return "learn_new"

    def learn_new(self, input_data: torch.Tensor, target: torch.Tensor) -> Dict[str, Any]:
        """
        Process new information (Flashcard).
        Cost: Energy.
        Reward: Potential future retrieval.
        """
        if self.energy < self.energy_cost_learn:
            return {"status": "failed", "reason": "low_energy"}

        self.energy -= self.energy_cost_learn
        
        # 1. Forward Pass & Train
        self.optimizer.zero_grad()
        output, meta_state = self.network(input_data)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        
        # Use entropy/stability for optimization step (simplified here)
        self.optimizer.step(system_entropy=meta_state['uncertainty'], stability_metric=1.0) # Assume stable for new

        # 2. Store in Memory
        # Generate a unique ID for this 'flashcard' (concept)
        memory_id = f"mem_{int(loss.item() * 10000)}_{np.random.randint(0, 1000)}"
        self.memory.add_memory(memory_id, {
            "input": input_data,
            "target": target,
            "learned_loss": loss.item()
        })
        
        return {
            "status": "learned", 
            "loss": loss.item(),
            "energy_remaining": self.energy,
            "memory_id": memory_id
        }

    def review(self) -> Dict[str, Any]:
        """
        Review an existing memory.
        If retrieval successful: Gain Energy (Reward).
        If retrieval failed (forgotten): Pay Compute Cost (Re-learn).
        """
        # 1. Select memory to review
        at_risk = self.memory.get_at_risk_memories(threshold=self.review_threshold)
        if not at_risk:
            # If nothing strictly at risk, pick a random one or oldest
            # For now, simplistic fallback
            return {"status": "skipped", "reason": "nothing_to_review"}
            
        memory_id = at_risk[0] # Pick most critical
        
        # 2. Attempt Retrieval
        memory_item = self.memory.retrieve_memory(memory_id)
        
        if memory_item:
            # SUCCESS: Memory retrieved (not decayed too much)
            # Reward: Gain energy
            self.energy += self.energy_reward_correct
            self.total_rewards += self.energy_reward_correct
            self.energy -= self.energy_cost_review # Cost of the action itself
            
            # Reinforce (Training on reviewed item)
            input_data = memory_item['input']
            target = memory_item['target']
            
            self.optimizer.zero_grad()
            output, _ = self.network(input_data)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            self.optimizer.step(0.5, 1.0) # Simplified update
            
            return {
                "status": "reviewed_success",
                "energy_gained": self.energy_reward_correct,
                "current_energy": self.energy
            }
        
        else:
            # FAILURE: Memory forgotten
            # Cost: Compute cycles to re-learn (simulated penalty)
            self.compute_cycles_spent += self.compute_cost_relearn
            
            # In a real system, we'd need to fetch the data again from the 'Environment' (Book)
            # Since we can't retrieve it from memory, we treat it as 'lost' until re-encountered.
            # But for the simulation, we might assume the agent 'looks it up'
            
            return {
                "status": "reviewed_failed",
                "penalty": "compute_cost_incurred",
                "compute_spent": self.compute_cycles_spent
            }
