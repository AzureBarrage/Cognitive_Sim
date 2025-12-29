import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
from pathlib import Path

class Visualizer:
    """
    Visualizes simulation data including loss curves, entropy, and memory stability.
    """
    def __init__(self, save_dir: str = "logs/plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_history(self, history: List[Dict[str, Any]], filename: str = "training_history.png"):
        """Plots loss, entropy, and plasticity over time."""
        if not history:
            return

        steps = [h["step"] for h in history]
        losses = [h["loss"] for h in history]
        entropies = [h["entropy"] for h in history]
        plasticities = [h["plasticity"] for h in history]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Loss Plot
        ax1.plot(steps, losses, label="Loss", color="blue")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Metrics Over Time")
        ax1.legend()
        ax1.grid(True)

        # Entropy Plot
        ax2.plot(steps, entropies, label="System Entropy", color="orange")
        ax2.set_ylabel("Entropy")
        ax2.legend()
        ax2.grid(True)

        # Plasticity Plot
        ax3.plot(steps, plasticities, label="Plasticity", color="green")
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Plasticity")
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig(self.save_dir / filename)
        plt.close()

    def plot_memory_distribution(self, memory_stability: np.ndarray, filename: str = "memory_stability.png"):
        """Plots the distribution of memory stability scores."""
        plt.figure(figsize=(10, 6))
        plt.hist(memory_stability, bins=20, color='purple', alpha=0.7)
        plt.title("Memory Stability Distribution")
        plt.xlabel("Stability Score")
        plt.ylabel("Count")
        plt.grid(True)
        plt.savefig(self.save_dir / filename)
        plt.close()
