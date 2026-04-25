import numpy as np

from src.core.entropy_calculator import EntropyCalculator


def test_weight_entropy_uniform_distribution() -> None:
    values = np.array([1.0, 1.0, 1.0, 1.0])
    entropy = EntropyCalculator.calculate_weight_entropy(values)
    assert entropy == 2.0


def test_memory_uncertainty_bounds() -> None:
    uncertainty = EntropyCalculator.calculate_memory_uncertainty(stability=1.0, decay_rate=0.2, time_elapsed=100.0)
    assert 0.0 <= uncertainty <= 1.0
    assert uncertainty > 0.9
