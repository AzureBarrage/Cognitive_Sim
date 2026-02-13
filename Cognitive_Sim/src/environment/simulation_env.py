from typing import Any, Dict, Iterator, Optional, Tuple

import torch
from torch.utils.data import DataLoader


class SimulationEnvironment:
    """Environment interface wrapper around dataloaders.

    Provides explicit reset/step API while still supporting flashcard pulls.
    """

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = self._get_iterator()
        self.current_epoch = 0
        self.current_step = 0

    def _get_iterator(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        return iter(self.dataloader)

    def reset(self, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seed is not None:
            torch.manual_seed(int(seed))
        self.iterator = self._get_iterator()
        self.current_epoch = 0
        self.current_step = 0
        return self.get_next_flashcard()

    def get_next_flashcard(self) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.current_epoch += 1
            self.iterator = self._get_iterator()
            batch = next(self.iterator)

        self.current_step += 1
        features, target = batch
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if target.ndim == 1:
            target = target.unsqueeze(0)
        return features, target

    def step(self) -> Dict[str, Any]:
        x, y = self.get_next_flashcard()
        return {
            "observation": x,
            "target": y,
            "reward": 0.0,
            "done": False,
            "info": {"epoch": self.current_epoch, "step": self.current_step},
        }
