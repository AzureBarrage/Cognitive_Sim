import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SimulationConfig


@pytest.fixture()
def temp_simulation_config(tmp_path) -> SimulationConfig:
    config = SimulationConfig()
    config.network.device = "cpu"
    config.runtime.checkpoint_path = str(tmp_path / "checkpoint.pt")
    config.runtime.artifact_dir = str(tmp_path / "artifacts")
    config.memory.store_dir = str(tmp_path / "memory_store")
    config.memory.index_path = str(tmp_path / "memory_index.json")
    config.memory.eager_load = False
    config.data.environment = "deterministic"
    config.data.size = 128
    config.data.batch_size = 8
    config.seed = 123
    return config
