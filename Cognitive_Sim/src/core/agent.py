from typing import Dict, Any, Tuple, List, Optional
import torch
import numpy as np
from src.core.memory_layer import MemoryLayer
from src.core.network import CognitiveNetwork
from src.core.optimizer import CognitiveOptimizer
from src.config import AgentConfig
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
                 config: Optional[AgentConfig] = None):
        self.memory = memory
        self.network = network
        self.optimizer = optimizer

        self.config = config or AgentConfig()
        
        # Agent State
        self.energy = float(self.config.initial_energy)
        self.compute_cycles_spent = 0.0
        self.total_rewards = 0.0

        # Success/failure counters for metrics
        self.review_successes = 0
        self.review_failures = 0
        self.learn_successes = 0
        self.learn_failures = 0
        
        # Cached config fields
        self.review_threshold = float(self.memory.review_threshold)
        self.energy_cost_learn = float(self.config.energy_cost_learn)
        self.energy_cost_review = float(self.config.energy_cost_review)
        self.energy_reward_correct = float(self.config.energy_reward_correct)
        self.compute_cost_relearn = float(self.config.compute_cost_relearn)
        self.energy_cost_sleep = float(self.config.energy_cost_sleep)
        self.energy_gain_sleep = float(self.config.energy_gain_sleep)
        self.sleep_when_energy_below = float(self.config.sleep_when_energy_below)
        self.consolidation_boost = float(self.config.consolidation_boost)

    def decide_strategy(self) -> str:
        """
        Decide whether to 'learn_new' or 'review'.
        Strategy: If there are memories at risk of decaying, prioritize review.
        """
        # If energy is critically low, sleep/consolidate
        if self.energy <= self.sleep_when_energy_below:
            logger.info("Agent decided to SLEEP (low energy)")
            return "sleep"

        # Fast due-check using memory scheduler
        if self.memory.has_at_risk_memory(threshold=self.review_threshold):
            logger.info("Agent decided to REVIEW (items due)")
            return "review"
        
        if self.energy < self.energy_cost_learn:
             logger.info("Agent too tired to learn new, forced to rest/review (or potentially fail)")
             # In a real survival sim, this might be death or forced rest. 
             # For now, we'll review if possible or do nothing.
             return "sleep"

        logger.info("Agent decided to LEARN NEW")
        return "learn_new"

    def learn_new(self, input_data: torch.Tensor, target: torch.Tensor) -> Dict[str, Any]:
        """
        Process new information (Flashcard).
        Cost: Energy.
        Reward: Potential future retrieval.
        """
        if self.energy < self.energy_cost_learn:
            self.learn_failures += 1
            return {"status": "failed", "reason": "low_energy"}

        self.energy -= self.energy_cost_learn
        
        # 1. Forward Pass & Train
        self.optimizer.zero_grad()
        output, meta_state = self.network(input_data)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        
        # Use entropy/stability for optimization step (simplified here)
        self.optimizer.step(system_entropy=meta_state['uncertainty'], memory_stability=1.0)  # Assume stable for new

        # 2. Store in Memory
        # Generate a unique ID for this 'flashcard' (concept)
        memory_id = f"mem_{int(loss.item() * 10000)}_{np.random.randint(0, 1000)}"
        self.memory.add_memory(memory_id, {
            "input": input_data,
            "target": target,
            "learned_loss": loss.item()
        })

        self.learn_successes += 1
        
        return {
            "status": "learned", 
            "loss": loss.item(),
            "energy_remaining": self.energy,
            "memory_id": memory_id
        }

    def sleep(self) -> Dict[str, Any]:
        """Consolidate memories and recover energy.

        Simple model:
        - Pay a small energy cost
        - Recover a chunk of energy
        - Boost stability of currently due memories
        """
        self.energy = max(0.0, self.energy - self.energy_cost_sleep)
        self.energy += self.energy_gain_sleep

        boosted = 0
        due_ids = self.memory.get_at_risk_memories(limit=250)
        for mem_id in due_ids:
            mem = self.memory.memories.get(mem_id)
            if not mem:
                continue
            mem['stability'] = float(mem.get('stability', 1.0)) * (1.0 + self.consolidation_boost)
            self.memory._schedule_review(mem_id)
            boosted += 1

        return {
            "status": "slept",
            "energy": self.energy,
            "boosted_memories": boosted
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

            self.review_successes += 1
            
            # Reinforce (Training on reviewed item)
            input_data = memory_item['input']
            target = memory_item['target']
            
            self.optimizer.zero_grad()
            output, _ = self.network(input_data)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            self.optimizer.step(system_entropy=0.5, memory_stability=1.0)  # Simplified update
            
            return {
                "status": "reviewed_success",
                "energy_gained": self.energy_reward_correct,
                "current_energy": self.energy
            }
        
        else:
            # FAILURE: Memory forgotten
            # Cost: Compute cycles to re-learn (simulated penalty)
            self.compute_cycles_spent += self.compute_cost_relearn
            self.review_failures += 1
            
            # In a real system, we'd need to fetch the data again from the 'Environment' (Book)
            # Since we can't retrieve it from memory, we treat it as 'lost' until re-encountered.
            # But for the simulation, we might assume the agent 'looks it up'
            
            return {
                "status": "reviewed_failed",
                "penalty": "compute_cost_incurred",
                "compute_spent": self.compute_cycles_spent
            }
