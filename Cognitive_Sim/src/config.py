from pydantic import BaseModel, Field
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class MemoryConfig(BaseModel):
    initial_retention: float = Field(default=1.0, ge=0.0, le=1.0)
    decay_rate: float = Field(default=0.1, gt=0.0)
    stability_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    # Policy / scheduling
    review_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    recall_failure_retention: float = Field(default=0.15, ge=0.0, le=1.0)

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

class SimulationConfig(BaseModel):
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    entropy_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    log_level: str = "INFO"

    @classmethod
    def load_from_yaml(cls, path: str | Path) -> "SimulationConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return cls(**config_data)

def load_config(env: str = "development") -> SimulationConfig:
    """Load configuration for the specified environment."""
    config_dir = Path(__file__).parent.parent / "configs"
    config_path = config_dir / f"{env}.yaml"
    
    if config_path.exists():
        return SimulationConfig.load_from_yaml(config_path)
    return SimulationConfig()
