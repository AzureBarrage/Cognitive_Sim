from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import NetworkConfig
from src.core.entropy_calculator import EntropyCalculator


class CognitiveNetwork(nn.Module):
    """Neural core with explicit train/eval steps and uncertainty tracking."""

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        self.input_size = int(config.input_size)
        self.hidden_size = int(config.hidden_size)
        self.output_size = int(config.output_size)
        self.entropy_threshold = float(config.entropy_threshold)
        self.uncertainty_update_interval = int(config.uncertainty_update_interval)
        self.loss_type = str(config.loss_type).lower()
        self.gradient_clip_norm = float(config.gradient_clip_norm)

        self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(0.2)
        self.plasticity = nn.Parameter(torch.ones(self.hidden_size))

        self._uncertainty_cache: float = 0.0
        self._uncertainty_dirty: bool = True
        self._uncertainty_updates_since_refresh: int = self.uncertainty_update_interval

    @staticmethod
    def resolve_device(device_name: str) -> torch.device:
        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_name)

    def get_layer_weights(self) -> List[np.ndarray]:
        return [
            self.layer1.weight.detach().cpu().numpy(),
            self.layer2.weight.detach().cpu().numpy(),
            self.output_layer.weight.detach().cpu().numpy(),
        ]

    def calculate_uncertainty(self) -> float:
        flat_weights = np.concatenate([w.reshape(-1) for w in self.get_layer_weights()], axis=0)
        uncertainty = float(EntropyCalculator.calculate_weight_entropy(flat_weights))
        self._uncertainty_cache = uncertainty
        self._uncertainty_dirty = False
        self._uncertainty_updates_since_refresh = 0
        return uncertainty

    def mark_uncertainty_dirty(self) -> None:
        self._uncertainty_dirty = True
        self._uncertainty_updates_since_refresh += 1

    def get_uncertainty(self, force: bool = False) -> float:
        should_refresh = force or (
            self._uncertainty_dirty
            and self._uncertainty_updates_since_refresh >= self.uncertainty_update_interval
        )
        if should_refresh:
            return self.calculate_uncertainty()
        return float(self._uncertainty_cache)

    @staticmethod
    def _step_optimizer(optimizer: Any, system_entropy: float, memory_stability: float) -> None:
        try:
            optimizer.step(system_entropy=system_entropy, memory_stability=memory_stability)
        except TypeError:
            optimizer.step()

    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            return F.mse_loss(output, target)
        if self.loss_type == "bce":
            return F.binary_cross_entropy_with_logits(output, target)
        raise ValueError("Unsupported loss_type: " + self.loss_type)

    def forward(
        self,
        x: torch.Tensor,
        memory_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        uncertainty = self.get_uncertainty()
        meta_state = {
            "uncertainty": uncertainty,
            "high_uncertainty": uncertainty > self.entropy_threshold,
        }

        x = F.relu(self.layer1(x))
        x = self.dropout(x)

        if memory_context is not None and memory_context.shape == x.shape:
            x = x * (1.0 + torch.tanh(memory_context))

        x = x * self.plasticity
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        output = self.output_layer(x)

        if meta_state["high_uncertainty"]:
            output = output * 0.8
        return output, meta_state

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: Any,
        memory_context: Optional[torch.Tensor] = None,
        memory_stability: float = 1.0,
    ) -> Dict[str, float]:
        self.train()
        optimizer.zero_grad()
        output, meta = self.forward(inputs, memory_context=memory_context)
        loss = self._compute_loss(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.gradient_clip_norm)
        self._step_optimizer(
            optimizer,
            system_entropy=float(meta["uncertainty"]),
            memory_stability=float(memory_stability),
        )
        self.mark_uncertainty_dirty()
        with torch.no_grad():
            mae = torch.mean(torch.abs(output - targets)).item()
            accuracy = max(0.0, 1.0 - mae)
        return {
            "loss": float(loss.item()),
            "uncertainty": float(self.get_uncertainty()),
            "mae": float(mae),
            "accuracy": float(accuracy),
        }

    def eval_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        memory_context: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        self.eval()
        with torch.no_grad():
            output, meta = self.forward(inputs, memory_context=memory_context)
            loss = self._compute_loss(output, targets)
            mae = torch.mean(torch.abs(output - targets)).item()
            accuracy = max(0.0, 1.0 - mae)
        return {
            "loss": float(loss.item()),
            "uncertainty": float(meta["uncertainty"]),
            "mae": float(mae),
            "accuracy": float(accuracy),
        }

    def update_plasticity(self, error_signal: float) -> None:
        with torch.no_grad():
            adjustment = torch.sigmoid(torch.tensor(error_signal)) * 0.1
            self.plasticity.add_(adjustment)
            self.plasticity.clamp_(0.5, 2.0)

    def save_checkpoint(self, path: str, optimizer_state: Optional[Dict[str, Any]] = None) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model_state": self.state_dict()}
        if optimizer_state is not None:
            payload["optimizer_state"] = optimizer_state
        torch.save(payload, checkpoint_path)

    def load_checkpoint(self, path: str) -> Optional[Dict[str, Any]]:
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            return None
        payload = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(payload, dict) and "model_state" in payload:
            self.load_state_dict(payload["model_state"])
            self.mark_uncertainty_dirty()
            self._uncertainty_updates_since_refresh = self.uncertainty_update_interval
            return payload
        self.load_state_dict(payload)
        self.mark_uncertainty_dirty()
        self._uncertainty_updates_since_refresh = self.uncertainty_update_interval
        return {"model_state": payload}

    def save_weights(self, path: str) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), checkpoint_path)

    def load_weights(self, path: str) -> None:
        checkpoint_path = Path(path)
        if checkpoint_path.exists():
            self.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            self.mark_uncertainty_dirty()
            self._uncertainty_updates_since_refresh = self.uncertainty_update_interval
