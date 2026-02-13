from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import torch


ActionType = str


@dataclass
class PolicyContext:
    energy: float
    max_energy: float
    due_count: int
    entropy: float
    memory_count: int


class MemoryStore(Protocol):
    def add_memory(self, memory_id: str, data: Any, initial_stability: float = 1.0) -> None:
        ...

    def retrieve_memory(self, memory_id: str) -> Optional[Any]:
        ...

    def get_at_risk_memories(self, threshold: Optional[float] = None, limit: int = 1000) -> List[str]:
        ...

    def has_at_risk_memory(self, threshold: Optional[float] = None) -> bool:
        ...

    def get_due_review_count(self, limit: int = 1000) -> int:
        ...


class Policy(Protocol):
    def select_action(self, context: PolicyContext) -> ActionType:
        ...

    def update(self, context: PolicyContext, action: ActionType, reward: float) -> None:
        ...


class Environment(Protocol):
    def reset(self, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def get_next_flashcard(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def step(self) -> Dict[str, Any]:
        ...


class BrainModel(Protocol):
    def forward(self, x: torch.Tensor, memory_context: Optional[torch.Tensor] = None):
        ...

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        memory_context: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        ...

    def eval_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        memory_context: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        ...
