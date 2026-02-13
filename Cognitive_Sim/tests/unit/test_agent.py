import torch

from src.config import AgentConfig, MemoryConfig, NetworkConfig
from src.core.agent import CognitiveAgent
from src.core.memory_layer import MemoryLayer
from src.core.network import CognitiveNetwork
from src.core.optimizer import CognitiveOptimizer


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
