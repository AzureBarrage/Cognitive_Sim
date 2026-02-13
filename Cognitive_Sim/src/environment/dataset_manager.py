from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class DeterministicToyDataset(Dataset):
    """Deterministic linear-ish mapping dataset."""

    def __init__(self, size: int = 256, input_dim: int = 10, output_dim: int = 5):
        self.size = int(size)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        base = torch.arange(self.size * self.input_dim, dtype=torch.float32).reshape(self.size, self.input_dim)
        self.data = ((base % 17) / 17.0) * 2.0 - 1.0
        weight = torch.linspace(-0.8, 0.8, steps=self.input_dim * self.output_dim, dtype=torch.float32).reshape(
            self.input_dim, self.output_dim
        )
        self.targets = torch.tanh(torch.matmul(self.data, weight))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


class StochasticToyDataset(Dataset):
    """Stochastic non-linear dataset with reproducible generator support."""

    def __init__(self, size: int = 256, input_dim: int = 10, output_dim: int = 5, noise_std: float = 0.1, seed: int = 42):
        self.size = int(size)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        gen = torch.Generator().manual_seed(int(seed))
        self.data = torch.randn(self.size, self.input_dim, generator=gen)
        base = torch.sin(self.data) + torch.cos(self.data * 0.5)
        projection = torch.randn(self.input_dim, self.output_dim, generator=gen)
        self.targets = torch.matmul(base, projection) / float(self.input_dim)
        self.targets = self.targets + torch.randn(self.size, self.output_dim, generator=gen) * float(noise_std)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


class DatasetManager:
    """Dataset factory and loader manager for simulation environments."""

    def __init__(self, config: Dict):
        self.config = config
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def load_data(self, seed: int = 42) -> None:
        data_cfg = self.config.get("data", {})
        env = str(data_cfg.get("environment", "deterministic")).lower()
        size = int(data_cfg.get("size", 256))
        input_dim = int(data_cfg.get("input_dim", self.config.get("network", {}).get("input_size", 10)))
        output_dim = int(data_cfg.get("output_dim", self.config.get("network", {}).get("output_size", 5)))
        noise_std = float(data_cfg.get("noise_std", 0.1))

        if env == "deterministic":
            self.train_dataset = DeterministicToyDataset(size=size, input_dim=input_dim, output_dim=output_dim)
            self.val_dataset = DeterministicToyDataset(size=max(32, size // 4), input_dim=input_dim, output_dim=output_dim)
            return

        if env == "stochastic":
            self.train_dataset = StochasticToyDataset(
                size=size,
                input_dim=input_dim,
                output_dim=output_dim,
                noise_std=noise_std,
                seed=seed,
            )
            self.val_dataset = StochasticToyDataset(
                size=max(32, size // 4),
                input_dim=input_dim,
                output_dim=output_dim,
                noise_std=noise_std,
                seed=seed + 1,
            )
            return

        raise ValueError("Unknown environment type: " + env)

    def get_train_loader(self, batch_size: int = 32) -> DataLoader:
        if self.train_dataset is None:
            self.load_data()
        return DataLoader(self.train_dataset, batch_size=int(batch_size), shuffle=True)

    def get_val_loader(self, batch_size: int = 32) -> DataLoader:
        if self.val_dataset is None:
            self.load_data()
        return DataLoader(self.val_dataset, batch_size=int(batch_size), shuffle=False)
