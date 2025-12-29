import torch
from torch.utils.data import DataLoader
from typing import Iterator, Tuple

class SimulationEnvironment:
    """
    Simulates the environment that feeds 'Flashcards' (data points) to the agent.
    Acts as the source of new knowledge.
    """
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = self._get_iterator()
        self.current_epoch = 0

    def _get_iterator(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        return iter(self.dataloader)

    def get_next_flashcard(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the next data point from the environment.
        Resets iterator if end of dataset is reached (cyclic environment).
        """
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.current_epoch += 1
            self.iterator = self._get_iterator()
            batch = next(self.iterator)
        
        # We process one item at a time for the agentic loop
        # Batch size in dataloader should ideally be 1, but we handle larger batches by slicing
        return batch[0][0].unsqueeze(0), batch[1][0].unsqueeze(0)
