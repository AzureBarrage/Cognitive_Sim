import torch

from src.config import MemoryConfig
from src.core.memory_layer import MemoryLayer


def test_decay_curve_monotonic(tmp_path) -> None:
    config = MemoryConfig(
        decay_rate=0.5,
        store_dir=str(tmp_path / "store"),
        index_path=str(tmp_path / "index.json"),
    )
    memory = MemoryLayer(config)
    memory.add_memory("m1", {"input": torch.zeros(1, 10), "target": torch.zeros(1, 5)})
    record = memory.memories["m1"]

    retention_now = memory._calculate_retention(record, now=record.last_reviewed)
    retention_later = memory._calculate_retention(record, now=record.last_reviewed + 100.0)
    assert retention_now >= retention_later
    assert retention_now <= 1.0
    assert retention_later >= 0.0


def test_spaced_repetition_scheduler_progresses_interval(tmp_path) -> None:
    config = MemoryConfig(
        initial_interval_seconds=1.0,
        store_dir=str(tmp_path / "store"),
        index_path=str(tmp_path / "index.json"),
    )
    memory = MemoryLayer(config)
    payload = {"input": torch.randn(1, 10), "target": torch.randn(1, 5)}
    memory.add_memory("m2", payload)

    first_interval = memory.memories["m2"].interval_seconds
    memory.review_memory("m2", success=True)
    second_interval = memory.memories["m2"].interval_seconds
    memory.review_memory("m2", success=True)
    third_interval = memory.memories["m2"].interval_seconds

    assert second_interval >= first_interval
    assert third_interval >= second_interval


def test_persistence_roundtrip(tmp_path) -> None:
    config = MemoryConfig(
        store_dir=str(tmp_path / "store"),
        index_path=str(tmp_path / "index.json"),
    )
    memory = MemoryLayer(config)
    payload = {"input": torch.randn(1, 10), "target": torch.randn(1, 5)}
    memory.add_memory("persist_me", payload)
    memory.save_state()

    clone = MemoryLayer(config)
    clone.load_state()
    restored = clone.retrieve_memory("persist_me", reinforce=False)
    assert restored is not None
    assert "input" in restored
    assert "target" in restored
