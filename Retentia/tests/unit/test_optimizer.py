import torch

from src.core.optimizer import CognitiveOptimizer


def test_optimizer_lr_adapts_with_entropy_and_stability() -> None:
    layer = torch.nn.Linear(2, 1)
    optimizer = CognitiveOptimizer(layer.parameters(), base_lr=0.01)
    optimizer.step(system_entropy=1.0, memory_stability=0.5)
    lr = optimizer.get_current_lr()
    assert 1e-5 <= lr <= 0.1


def test_optimizer_state_roundtrip() -> None:
    layer = torch.nn.Linear(2, 1)
    opt1 = CognitiveOptimizer(layer.parameters(), base_lr=0.02)
    state = opt1.state_dict()

    opt2 = CognitiveOptimizer(layer.parameters(), base_lr=0.01)
    opt2.load_state_dict(state)
    assert abs(opt2.base_lr - 0.02) < 1e-9
