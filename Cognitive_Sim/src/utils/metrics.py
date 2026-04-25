import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class MetricsTracker:
    """Tracks counters/time-series and exports JSONL artifacts."""

    def __init__(self, artifact_dir: str = "logs/runs"):
        self.start_time = time.time()
        self.metrics: Dict[str, List[Dict[str, Any]]] = {
            "training": [],
            "system": [],
            "agent": [],
        }
        self.counters: Dict[str, int] = {
            "learn_events": 0,
            "review_events": 0,
            "review_selected": 0,
            "review_attempted": 0,
            "review_skipped": 0,
            "review_success": 0,
            "review_failure": 0,
            "forgetting_events": 0,
            "sleep_events": 0,
        }

        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        run_id = str(int(self.start_time))
        self.jsonl_path = self.artifact_dir / ("run_" + run_id + ".jsonl")

    def _timestamp(self) -> float:
        return float(time.time() - self.start_time)

    def _append(self, category: str, payload: Dict[str, Any]) -> None:
        row = {"category": category, "timestamp": self._timestamp(), **payload}
        self.metrics.setdefault(category, []).append(row)
        with open(self.jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def log_training_step(self, step: int, loss: float, entropy: float, accuracy: float, energy: float) -> None:
        self._append(
            "training",
            {
                "step": int(step),
                "loss": float(loss),
                "entropy": float(entropy),
                "accuracy": float(accuracy),
                "energy": float(energy),
            },
        )

    def log_system_health(self, step: int, memory_usage: float, cpu_usage: float) -> None:
        self._append(
            "system",
            {
                "step": int(step),
                "memory_usage": float(memory_usage),
                "cpu_usage": float(cpu_usage),
            },
        )

    def log_agent_state(
        self,
        step: int,
        action: str,
        energy: float,
        memory_count: int,
        entropy: float,
        review_success_rate: float,
    ) -> None:
        self._append(
            "agent",
            {
                "step": int(step),
                "action": action,
                "energy": float(energy),
                "memory_count": int(memory_count),
                "entropy": float(entropy),
                "review_success_rate": float(review_success_rate),
            },
        )

    def inc(self, counter_name: str, delta: int = 1) -> None:
        self.counters[counter_name] = int(self.counters.get(counter_name, 0)) + int(delta)

    def get_history(self, category: str) -> List[Dict[str, Any]]:
        return self.metrics.get(category, [])

    def get_latest(self, category: str) -> Dict[str, Any]:
        history = self.metrics.get(category, [])
        return history[-1] if history else {}

    def summarize(self) -> Dict[str, Any]:
        training = self.metrics.get("training", [])
        losses = [row["loss"] for row in training if "loss" in row]
        accuracy = [row["accuracy"] for row in training if "accuracy" in row]
        entropy = [row["entropy"] for row in training if "entropy" in row]
        energy = [row["energy"] for row in training if "energy" in row]

        summary = {
            "duration_seconds": self._timestamp(),
            "events": {k: int(v) for k, v in self.counters.items()},
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
            "avg_accuracy": float(np.mean(accuracy)) if accuracy else 0.0,
            "avg_entropy": float(np.mean(entropy)) if entropy else 0.0,
            "avg_energy": float(np.mean(energy)) if energy else 0.0,
        }
        summary_path = self.artifact_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary

    def trends(self, window: int = 20) -> Dict[str, float]:
        training = self.metrics.get("training", [])
        if not training:
            return {"loss": 0.0, "accuracy": 0.0, "entropy": 0.0, "energy": 0.0}

        sliced = training[-max(1, int(window)) :]
        return {
            "loss": float(np.mean([row.get("loss", 0.0) for row in sliced])),
            "accuracy": float(np.mean([row.get("accuracy", 0.0) for row in sliced])),
            "entropy": float(np.mean([row.get("entropy", 0.0) for row in sliced])),
            "energy": float(np.mean([row.get("energy", 0.0) for row in sliced])),
        }

    def calculate_average_loss(self, window: int = 100) -> float:
        history = self.metrics.get("training", [])
        if not history:
            return 0.0
        losses = [row.get("loss", 0.0) for row in history[-max(1, int(window)) :]]
        return float(np.mean(losses))

    def extract_series(self, category: str, field: str, default: Optional[float] = None) -> List[float]:
        values: List[float] = []
        for row in self.metrics.get(category, []):
            if field in row:
                values.append(float(row[field]))
            elif default is not None:
                values.append(float(default))
        return values
