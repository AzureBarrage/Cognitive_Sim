import torch

from src.config import MemoryConfig
from src.core.memory_layer import MemoryLayer


def test_due_memory_scheduler_returns_oldest_due_item(tmp_path) -> None:
    memory = MemoryLayer(
        MemoryConfig(
            store_dir=str(tmp_path / "store"),
            index_path=str(tmp_path / "index.json"),
            initial_interval_seconds=1.0,
        )
    )

    memory.add_memory("old", {"input": torch.randn(1, 10), "target": torch.randn(1, 5)})
    memory.add_memory("new", {"input": torch.randn(1, 10), "target": torch.randn(1, 5)})

    memory.memories["old"].last_reviewed -= 1000.0
    memory.memories["new"].last_reviewed -= 10.0
    memory._schedule_review("old")
    memory._schedule_review("new")

    due = memory.get_at_risk_memories(limit=2)
    assert "old" in due
