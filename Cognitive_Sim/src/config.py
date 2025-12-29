from pydantic import BaseModel, Field
import yaml
from pathlib import Path
from typing import Dict, Any

class MemoryConfig(BaseModel):
    initial_retention: float = Field(default=1.0, ge=0.0, le=1.0)
    decay_rate: float = Field(default=0.1, gt=0.0)
    stability_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

class NetworkConfig(BaseModel):
    input_size: int = Field(default=10, gt=0)
    hidden_size: int = Field(default=20, gt=0)
    output_size: int = Field(default=5, gt=0)
    learning_rate: float = Field(default=0.01, gt=0.0)
    entropy_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

class SimulationConfig(BaseModel):
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
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
