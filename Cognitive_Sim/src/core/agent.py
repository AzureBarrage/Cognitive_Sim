from typing import Any, Dict, Optional

import numpy as np
import torch

from src.config import AgentConfig
from src.core.interfaces import Policy, PolicyContext
from src.core.memory_layer import MemoryLayer
from src.core.network import CognitiveNetwork
from src.core.optimizer import CognitiveOptimizer
from src.core.policy import build_policy


class CognitiveAgent:
    """Agent orchestration: policy -> memory/network actions -> reward update."""

    def __init__(
        self,
        memory: MemoryLayer,
        network: CognitiveNetwork,
        optimizer: CognitiveOptimizer,
        config: Optional[AgentConfig] = None,
        policy: Optional[Policy] = None,
    ):
        self.memory = memory
        self.network = network
        self.optimizer = optimizer
        self.config = config or AgentConfig()
        self.policy = policy or build_policy(self.config)

        self.energy = float(self.config.initial_energy)
        self.max_energy = float(self.config.max_energy)
        self.compute_cycles_spent = 0.0
        self.total_rewards = 0.0

        self.review_successes = 0
        self.review_failures = 0
        self.learn_successes = 0
        self.learn_failures = 0
        self.sleep_count = 0

        self.last_action: Optional[str] = None
        self.last_reward: float = 0.0

    def _build_policy_context(self) -> PolicyContext:
        return PolicyContext(
            energy=float(self.energy),
            max_energy=float(self.max_energy),
            due_count=int(self.memory.get_due_review_count(limit=self.config.review_due_limit)),
            entropy=float(self.network.calculate_uncertainty()),
            memory_count=len(self.memory.memories),
        )

    def decide_strategy(self) -> str:
        context = self._build_policy_context()
        action = self.policy.select_action(context)
        self.last_action = action
        return action

    def _clamp_energy(self) -> None:
        self.energy = float(max(0.0, min(self.max_energy, self.energy)))

    def _apply_reward(self, context: PolicyContext, action: str, reward: float) -> None:
        self.last_reward = float(reward)
        self.total_rewards += float(reward)
        self.policy.update(context, action, reward)

    def learn_new(self, input_data: torch.Tensor, target: torch.Tensor, memory_id: Optional[str] = None) -> Dict[str, Any]:
        context = self._build_policy_context()
        if self.energy < float(self.config.energy_cost_learn):
            self.learn_failures += 1
            self._apply_reward(context, "learn_new", -1.0)
            return {"status": "failed", "reason": "low_energy"}

        self.energy -= float(self.config.energy_cost_learn)
        train_result = self.network.train_step(input_data, target, self.optimizer.optimizer)

        if memory_id is None:
            loss_value = int(train_result["loss"] * 10000)
            memory_id = "mem_" + str(loss_value) + "_" + str(np.random.randint(0, 100000))

        embedding = input_data.detach().flatten().cpu().tolist()
        self.memory.add_memory(
            memory_id,
            data={"input": input_data.detach().cpu(), "target": target.detach().cpu(), "loss": train_result["loss"]},
            initial_stability=1.0,
            embedding=embedding,
        )

        self.learn_successes += 1
        self._clamp_energy()

        reward = -float(self.config.energy_cost_learn) + max(0.0, 1.0 - float(train_result["loss"]))
        self._apply_reward(context, "learn_new", reward)
        return {
            "status": "learned",
            "loss": train_result["loss"],
            "mae": train_result["mae"],
            "uncertainty": train_result["uncertainty"],
            "energy_remaining": self.energy,
            "memory_id": memory_id,
        }

    def review(self) -> Dict[str, Any]:
        context = self._build_policy_context()
        due_ids = self.memory.get_at_risk_memories(limit=self.config.review_due_limit)
        if not due_ids:
            self._apply_reward(context, "review", -0.1)
            return {"status": "skipped", "reason": "nothing_to_review"}

        memory_id = due_ids[0]
        payload = self.memory.retrieve_memory(memory_id, reinforce=False)
        self.energy -= float(self.config.energy_cost_review)

        if payload is None:
            self.memory.review_memory(memory_id, success=False)
            self.review_failures += 1
            self.compute_cycles_spent += float(self.config.compute_cost_relearn)
            self._clamp_energy()
            reward = -float(self.config.energy_cost_review) - 1.0
            self._apply_reward(context, "review", reward)
            return {
                "status": "reviewed_failed",
                "memory_id": memory_id,
                "compute_spent": self.compute_cycles_spent,
            }

        input_data = payload["input"]
        target = payload["target"]
        train_result = self.network.train_step(input_data, target, self.optimizer.optimizer)
        self.memory.review_memory(memory_id, success=True)

        self.review_successes += 1
        self.energy += float(self.config.energy_reward_correct)
        self._clamp_energy()

        reward = float(self.config.energy_reward_correct - self.config.energy_cost_review) + max(0.0, 1.0 - train_result["loss"])
        self._apply_reward(context, "review", reward)
        return {
            "status": "reviewed_success",
            "memory_id": memory_id,
            "loss": train_result["loss"],
            "uncertainty": train_result["uncertainty"],
            "current_energy": self.energy,
        }

    def sleep(self) -> Dict[str, Any]:
        context = self._build_policy_context()
        self.energy = max(0.0, self.energy - float(self.config.energy_cost_sleep))
        self.energy += float(self.config.energy_gain_sleep)
        boosted = self.memory.consolidate_due_memories(limit=250, boost=float(self.config.consolidation_boost))
        self.sleep_count += 1
        self._clamp_energy()

        reward = -float(self.config.energy_cost_sleep) + (0.1 * boosted)
        self._apply_reward(context, "sleep", reward)
        return {
            "status": "slept",
            "energy": self.energy,
            "boosted_memories": boosted,
        }

    def act(self, action: str, input_data: Optional[torch.Tensor] = None, target: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        if action == "learn_new":
            if input_data is None or target is None:
                raise ValueError("learn_new action requires input_data and target")
            return self.learn_new(input_data=input_data, target=target)
        if action == "review":
            return self.review()
        if action == "sleep":
            return self.sleep()
        raise ValueError("Unsupported action: " + action)
