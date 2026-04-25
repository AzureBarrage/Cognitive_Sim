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


def test_advance_time_makes_memory_due(tmp_path) -> None:
    config = MemoryConfig(
        initial_interval_seconds=30.0,
        store_dir=str(tmp_path / "store"),
        index_path=str(tmp_path / "index.json"),
    )
    memory = MemoryLayer(config)
    memory.add_memory("advance_due", {"input": torch.randn(1, 10), "target": torch.randn(1, 5)})

    before = memory.get_due_review_count()
    memory.advance_time(120.0)
    after = memory.get_due_review_count()

    assert before == 0
    assert after >= 1


def test_due_review_helpers_use_heap_for_default_threshold(tmp_path, monkeypatch) -> None:
    config = MemoryConfig(
        initial_interval_seconds=30.0,
        store_dir=str(tmp_path / "store"),
        index_path=str(tmp_path / "index.json"),
    )
    memory = MemoryLayer(config)
    memory.add_memory("heap_due", {"input": torch.randn(1, 10), "target": torch.randn(1, 5)})
    memory.advance_time(120.0)

    def fail_retention_scan(*args, **kwargs):
        raise AssertionError("default due helpers should not recalculate retention")

    monkeypatch.setattr(memory, "_calculate_retention", fail_retention_scan)

    assert memory.get_due_review_count() == 1
    assert memory.has_at_risk_memory() is True
    assert memory.get_due_memory_ids(limit=1) == ["heap_due"]


def test_save_state_skips_clean_payload_rewrites(tmp_path, monkeypatch) -> None:
    config = MemoryConfig(
        store_dir=str(tmp_path / "store"),
        index_path=str(tmp_path / "index.json"),
    )
    memory = MemoryLayer(config)
    memory.add_memory("clean_save", {"input": torch.randn(1, 10), "target": torch.randn(1, 5)})

    saved_paths = []
    original_save = torch.save

    def counting_save(*args, **kwargs):
        saved_paths.append(str(args[1]))
        return original_save(*args, **kwargs)

    monkeypatch.setattr("src.core.memory_layer.torch.save", counting_save)

    memory.save_state()
    first_save_count = len(saved_paths)
    memory.save_state()

    assert first_save_count >= 1
    assert len(saved_paths) == first_save_count
