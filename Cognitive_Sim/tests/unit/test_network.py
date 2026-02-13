import torch

from src.config import NetworkConfig
from src.core.network import CognitiveNetwork


def test_train_step_returns_expected_metrics() -> None:
    config = NetworkConfig(input_size=10, hidden_size=16, output_size=5, device="cpu")
    network = CognitiveNetwork(config)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

    x = torch.randn(4, 10)
    y = torch.randn(4, 5)
    result = network.train_step(x, y, optimizer)

    assert "loss" in result
    assert "uncertainty" in result
    assert "mae" in result
    assert result["loss"] >= 0.0


def test_eval_step_does_not_crash() -> None:
    config = NetworkConfig(input_size=10, hidden_size=16, output_size=5, device="cpu")
    network = CognitiveNetwork(config)
    x = torch.randn(2, 10)
    y = torch.randn(2, 5)
    result = network.eval_step(x, y)
    assert result["loss"] >= 0.0
