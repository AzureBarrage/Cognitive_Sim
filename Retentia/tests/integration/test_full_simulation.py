from src.config import SimulationConfig
from src.main import CognitiveSimulation


def test_short_deterministic_simulation(monkeypatch, tmp_path) -> None:
    cfg = SimulationConfig()
    cfg.seed = 123
    cfg.network.device = "cpu"
    cfg.data.environment = "deterministic"
    cfg.data.size = 64
    cfg.data.batch_size = 8
    cfg.runtime.checkpoint_path = str(tmp_path / "checkpoint.pt")
    cfg.runtime.artifact_dir = str(tmp_path / "artifacts")
    cfg.memory.store_dir = str(tmp_path / "memory_store")
    cfg.memory.index_path = str(tmp_path / "memory_index.json")

    monkeypatch.setattr("src.main.load_config", lambda env: cfg)

    sim = CognitiveSimulation(config_env="testing", fresh=True, seed=123)
    summary = sim.run_training_loop(steps=20, sleep_seconds=0.0)

    assert summary["duration_seconds"] >= 0.0
    assert summary["memory"]["count"] >= 0.0


def test_simulated_time_advances_due_reviews_and_counters(monkeypatch, tmp_path) -> None:
    cfg = SimulationConfig()
    cfg.seed = 123
    cfg.network.device = "cpu"
    cfg.data.environment = "deterministic"
    cfg.data.size = 64
    cfg.data.batch_size = 8
    cfg.runtime.checkpoint_path = str(tmp_path / "checkpoint.pt")
    cfg.runtime.artifact_dir = str(tmp_path / "artifacts")
    cfg.memory.store_dir = str(tmp_path / "memory_store")
    cfg.memory.index_path = str(tmp_path / "memory_index.json")
    cfg.memory.initial_interval_seconds = 10.0
    cfg.runtime.simulated_step_seconds = 60.0
    cfg.agent.policy_type = "heuristic"
    cfg.agent.review_due_limit = 2

    monkeypatch.setattr("src.main.load_config", lambda env: cfg)

    sim = CognitiveSimulation(config_env="testing", fresh=True, seed=123)
    summary = sim.run_training_loop(steps=12, sleep_seconds=0.0)

    events = summary["events"]
    assert events["learn_events"] >= 1
    assert events["review_selected"] >= 1
    assert events["review_attempted"] >= 1
