import torch

from src.config import AgentConfig, MemoryConfig, NetworkConfig
from src.core.agent import CognitiveAgent
from src.core.memory_layer import MemoryLayer
from src.core.network import CognitiveNetwork
from src.core.optimizer import CognitiveOptimizer


class RecordingOptimizer(CognitiveOptimizer):
    def __init__(self, model_parameters, base_lr: float = 0.01):
        super().__init__(model_parameters, base_lr=base_lr)
        self.calls = []

    def step(self, system_entropy: float = 0.0, memory_stability: float = 1.0) -> None:
        self.calls.append((float(system_entropy), float(memory_stability)))
        super().step(system_entropy=system_entropy, memory_stability=memory_stability)


def test_agent_review_boundary_conditions(tmp_path) -> None:
    memory = MemoryLayer(
        MemoryConfig(
            store_dir=str(tmp_path / "store"),
            index_path=str(tmp_path / "index.json"),
            initial_interval_seconds=1.0,
            recall_failure_retention=0.9,
        )
    )
    network = CognitiveNetwork(NetworkConfig(input_size=10, hidden_size=16, output_size=5, device="cpu"))
    optimizer = CognitiveOptimizer(network.parameters(), base_lr=0.01)
    agent = CognitiveAgent(memory, network, optimizer, config=AgentConfig(initial_energy=20.0))

    memory.add_memory("m1", {"input": torch.randn(1, 10), "target": torch.randn(1, 5)})
    memory.memories["m1"].last_reviewed -= 10000.0

    result = agent.review()
    assert result["status"] in {"reviewed_failed", "reviewed_success"}


def test_agent_learn_updates_memory() -> None:
    memory = MemoryLayer(MemoryConfig())
    network = CognitiveNetwork(NetworkConfig(input_size=10, hidden_size=16, output_size=5, device="cpu"))
    optimizer = CognitiveOptimizer(network.parameters(), base_lr=0.01)
    agent = CognitiveAgent(memory, network, optimizer, config=AgentConfig(initial_energy=100.0))

    x = torch.randn(2, 10)
    y = torch.randn(2, 5)
    before = len(memory.memories)
    result = agent.learn_new(x, y)
    after = len(memory.memories)
    assert result["status"] == "learned"
    assert after == before + 1


def test_agent_uses_cached_uncertainty_and_optimizer_wrapper(tmp_path, monkeypatch) -> None:
    memory = MemoryLayer(
        MemoryConfig(
            store_dir=str(tmp_path / "store"),
            index_path=str(tmp_path / "index.json"),
            initial_interval_seconds=1.0,
        )
    )
    network = CognitiveNetwork(NetworkConfig(input_size=10, hidden_size=16, output_size=5, device="cpu", uncertainty_update_interval=1))
    optimizer = RecordingOptimizer(network.parameters(), base_lr=0.01)
    agent = CognitiveAgent(memory, network, optimizer, config=AgentConfig(initial_energy=100.0))

    calculate_calls = {"count": 0}
    original_calculate = network.calculate_uncertainty

    def counted_calculate() -> float:
        calculate_calls["count"] += 1
        return original_calculate()

    monkeypatch.setattr(network, "calculate_uncertainty", counted_calculate)

    x = torch.randn(2, 10)
    y = torch.randn(2, 5)
    result = agent.learn_new(x, y)
    context = agent._build_policy_context()

    assert result["status"] == "learned"
    assert optimizer.calls
    assert optimizer.calls[0][1] >= 1.0
    assert calculate_calls["count"] >= 1
    assert context.entropy == network.get_uncertainty()
