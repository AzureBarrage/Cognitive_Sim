from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field

class MemoryConfig(BaseModel):
    initial_retention: float = Field(default=1.0, ge=0.0, le=1.0)
    decay_rate: float = Field(default=0.1, gt=0.0)
    stability_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    # Policy / scheduling
    review_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    recall_failure_retention: float = Field(default=0.15, ge=0.0, le=1.0)
    min_stability: float = Field(default=0.1, gt=0.0)
    default_difficulty: float = Field(default=2.5, ge=1.3, le=3.0)
    initial_interval_seconds: float = Field(default=30.0, gt=0.0)
    relearn_penalty_factor: float = Field(default=1.5, ge=1.0)

    # Persistence
    store_dir: str = Field(default="data/memory_store")
    index_path: str = Field(default="data/memory_index.json")
    eager_load: bool = Field(default=False)

class NetworkConfig(BaseModel):
    input_size: int = Field(default=10, gt=0)
    hidden_size: int = Field(default=20, gt=0)
    output_size: int = Field(default=5, gt=0)
    learning_rate: float = Field(default=0.01, gt=0.0)
    entropy_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    loss_type: str = Field(default="mse")
    gradient_clip_norm: float = Field(default=1.0, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    device: str = Field(default="auto")

    # Performance/monitoring
    uncertainty_update_interval: int = Field(default=10, ge=1)


class AgentConfig(BaseModel):
    # Energy economy
    initial_energy: float = Field(default=100.0, ge=0.0)
    energy_cost_learn: float = Field(default=5.0, ge=0.0)
    energy_cost_review: float = Field(default=2.0, ge=0.0)
    energy_reward_correct: float = Field(default=10.0, ge=0.0)

    # Sleep/consolidation
    energy_cost_sleep: float = Field(default=1.0, ge=0.0)
    energy_gain_sleep: float = Field(default=25.0, ge=0.0)
    sleep_when_energy_below: float = Field(default=10.0, ge=0.0)
    consolidation_boost: float = Field(default=0.25, ge=0.0)

    # Penalties
    compute_cost_relearn: float = Field(default=15.0, ge=0.0)

    # Policy
    policy_type: str = Field(default="heuristic")
    entropy_review_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    review_due_limit: int = Field(default=25, ge=1)
    bandit_epsilon: float = Field(default=0.1, ge=0.0, le=1.0)
    bandit_learning_rate: float = Field(default=0.05, gt=0.0)
    max_energy: float = Field(default=200.0, gt=0.0)


class DataConfig(BaseModel):
    environment: str = Field(default="deterministic")
    size: int = Field(default=256, ge=16)
    input_dim: int = Field(default=10, ge=1)
    output_dim: int = Field(default=5, ge=1)
    noise_std: float = Field(default=0.1, ge=0.0)
    batch_size: int = Field(default=16, ge=1)


class RuntimeConfig(BaseModel):
    checkpoint_path: str = Field(default="data/network_checkpoint.pt")
    artifact_dir: str = Field(default="logs/runs")
    dev_reset_enabled: bool = Field(default=True)
    tenant_db_path: str = Field(default="data/tenant_memory.db")
    analytics_window_days: int = Field(default=30, ge=1)
    pilot_min_retained_mastery_lift: float = Field(default=0.05)
    pilot_min_forgetting_velocity_reduction: float = Field(default=0.10)
    pilot_min_review_efficiency_lift: float = Field(default=0.05)
    pilot_confidence_z_threshold: float = Field(default=1.96, gt=0.0)
    pilot_min_sample_size: int = Field(default=100, ge=10)
    pilot_max_onboarding_hours: float = Field(default=40.0, ge=0.0)

class SimulationConfig(BaseModel):
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    seed: int = Field(default=42)
    entropy_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    log_level: str = "INFO"

    @classmethod
    def load_from_yaml(cls, path: Union[str, Path]) -> "SimulationConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        return cls(**config_data)

def load_config(env: str = "development") -> SimulationConfig:
    """Load configuration for the specified environment."""
    config_dir = Path(__file__).parent.parent / "configs"
    config_path = config_dir / f"{env}.yaml"

    if config_path.exists():
        return SimulationConfig.load_from_yaml(config_path)
    return SimulationConfig()
