from typing import Optional

import numpy as np
import torch
import torch.optim as optim


class CognitiveOptimizer:
    """Optimizer wrapper with adaptive LR and scheduler support."""

    def __init__(
        self,
        model_parameters,
        base_lr: float = 0.01,
        weight_decay: float = 0.0,
    ):
        self.base_lr = float(base_lr)
        self.optimizer = optim.Adam(model_parameters, lr=self.base_lr, weight_decay=float(weight_decay))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

    def step(self, system_entropy: float = 0.0, memory_stability: float = 1.0) -> None:
        adaptive_factor = (1.0 + float(system_entropy)) / (1.0 + max(0.0, float(memory_stability)))
        current_lr = float(np.clip(self.base_lr * adaptive_factor, 1e-5, 0.1))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def update_scheduler(self, metric: float) -> None:
        self.scheduler.step(float(metric))

    def get_current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "base_lr": self.base_lr,
        }

    def load_state_dict(self, state: Optional[dict]) -> None:
        if not state:
            return
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
        if "base_lr" in state:
            self.base_lr = float(state["base_lr"])
