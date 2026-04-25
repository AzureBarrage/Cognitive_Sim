import torch

from src.config import NetworkConfig
from src.core.network import CognitiveNetwork
from src.core.optimizer import CognitiveOptimizer


class SpyOptimizer(CognitiveOptimizer):
    def __init__(self, model_parameters, base_lr: float = 0.01):
        super().__init__(model_parameters, base_lr=base_lr)
        self.calls = []

    def step(self, system_entropy: float = 0.0, memory_stability: float = 1.0) -> None:
        self.calls.append((float(system_entropy), float(memory_stability)))
        super().step(system_entropy=system_entropy, memory_stability=memory_stability)


def test_train_step_returns_expected_metrics() -> None:
    config = NetworkConfig(input_size=10, hidden_size=16, output_size=5, device="cpu", uncertainty_update_interval=1)
    network = CognitiveNetwork(config)
    optimizer = SpyOptimizer(network.parameters(), base_lr=0.01)

    x = torch.randn(4, 10)
    y = torch.randn(4, 5)
    result = network.train_step(x, y, optimizer, memory_stability=0.5)

    assert "loss" in result
    assert "uncertainty" in result
    assert "mae" in result
    assert "accuracy" in result
    assert result["loss"] >= 0.0
    assert optimizer.calls
    assert abs(optimizer.calls[0][1] - 0.5) < 1e-9


def test_eval_step_does_not_crash() -> None:
    config = NetworkConfig(input_size=10, hidden_size=16, output_size=5, device="cpu")
    network = CognitiveNetwork(config)
    x = torch.randn(2, 10)
    y = torch.randn(2, 5)
    result = network.eval_step(x, y)
    assert result["loss"] >= 0.0
    assert "accuracy" in result


def test_get_uncertainty_uses_cache_until_marked_dirty(monkeypatch) -> None:
    network = CognitiveNetwork(NetworkConfig(input_size=10, hidden_size=16, output_size=5, device="cpu"))
    call_count = {"count": 0}
    original_calculate_uncertainty = network.calculate_uncertainty

    def counted_calculate_uncertainty() -> float:
        call_count["count"] += 1
        return original_calculate_uncertainty()

    monkeypatch.setattr(network, "calculate_uncertainty", counted_calculate_uncertainty)

    first = network.get_uncertainty()
    second = network.get_uncertainty()
    network.mark_uncertainty_dirty()
    third = network.get_uncertainty()

    assert first == second
    assert third == second
    assert call_count["count"] == 1
