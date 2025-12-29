import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional

class SyntheticDataset(Dataset):
    """
    A simple synthetic dataset for testing the cognitive simulation.
    Generates random input data and target labels.
    """
    def __init__(self, size: int = 1000, input_dim: int = 10, output_dim: int = 1):
        self.size = size
        self.data = torch.randn(size, input_dim)
        # Create a simple non-linear relationship for the target
        self.targets = torch.sum(self.data ** 2, dim=1, keepdim=True) + torch.randn(size, output_dim) * 0.1

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]

class DatasetManager:
    """
    Manages data loading and processing for the simulation.
    """
    def __init__(self, config: dict):
        self.config = config
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def load_data(self):
        """
        Loads the dataset based on configuration. 
        Currently supports 'synthetic' data for simulation testing.
        """
        data_type = self.config.get("data", {}).get("type", "synthetic")
        
        if data_type == "synthetic":
            input_dim = self.config.get("network", {}).get("input_size", 10)
            output_dim = self.config.get("network", {}).get("output_size", 1)
            size = self.config.get("data", {}).get("size", 1000)
            
            self.train_dataset = SyntheticDataset(size=size, input_dim=input_dim, output_dim=output_dim)
            self.val_dataset = SyntheticDataset(size=size // 5, input_dim=input_dim, output_dim=output_dim)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def get_train_loader(self, batch_size: int = 32) -> DataLoader:
        """Returns the training data loader."""
        if self.train_dataset is None:
            self.load_data()
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

    def get_val_loader(self, batch_size: int = 32) -> DataLoader:
        """Returns the validation data loader."""
        if self.val_dataset is None:
            self.load_data()
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
