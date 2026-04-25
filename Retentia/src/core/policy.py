import random
from dataclasses import dataclass, field
from typing import Dict

from src.config import AgentConfig
from src.core.interfaces import ActionType, Policy, PolicyContext


@dataclass
class HeuristicPolicy(Policy):
    entropy_review_threshold: float
    sleep_when_energy_below: float
    review_due_minimum: int

    def select_action(self, context: PolicyContext) -> ActionType:
        energy_ratio = context.energy / max(context.max_energy, 1e-6)
        if context.energy <= self.sleep_when_energy_below:
            return "sleep"

        # Guardrails: avoid degenerate review-only behavior.
        if context.memory_count <= 0:
            return "learn_new"
        if context.due_count <= 0:
            if energy_ratio < 0.1:
                return "sleep"
            return "learn_new"

        if context.due_count >= self.review_due_minimum:
            return "review"
        if context.entropy >= self.entropy_review_threshold and energy_ratio > 0.2:
            return "review"
        if energy_ratio < 0.1:
            return "sleep"
        return "learn_new"

    def update(self, context: PolicyContext, action: ActionType, reward: float) -> None:
        return


@dataclass
class ContextualBanditPolicy(Policy):
    epsilon: float
    alpha: float
    q_values: Dict[str, float] = field(default_factory=lambda: {"learn_new": 0.0, "review": 0.0, "sleep": 0.0})

    def _features(self, context: PolicyContext) -> float:
        return (
            (context.energy / max(context.max_energy, 1e-6)) * 0.5
            + min(1.0, context.due_count / 25.0) * 0.3
            + min(1.0, context.entropy) * 0.2
        )

    def select_action(self, context: PolicyContext) -> ActionType:
        if random.random() < self.epsilon:
            return random.choice(["learn_new", "review", "sleep"])
        score = self._features(context)
        best_action = "learn_new"
        best_value = float("-inf")
        for action, value in self.q_values.items():
            adjusted = value + (0.05 * score if action == "review" else 0.0)
            if adjusted > best_value:
                best_value = adjusted
                best_action = action
        return best_action

    def update(self, context: PolicyContext, action: ActionType, reward: float) -> None:
        old = self.q_values.get(action, 0.0)
        self.q_values[action] = old + self.alpha * (reward - old)


def build_policy(config: AgentConfig) -> Policy:
    policy_type = str(config.policy_type).lower()
    if policy_type == "bandit":
        return ContextualBanditPolicy(
            epsilon=float(config.bandit_epsilon),
            alpha=float(config.bandit_learning_rate),
        )
    return HeuristicPolicy(
        entropy_review_threshold=float(config.entropy_review_threshold),
        sleep_when_energy_below=float(config.sleep_when_energy_below),
        review_due_minimum=max(1, int(config.review_due_limit // 2)),
    )
