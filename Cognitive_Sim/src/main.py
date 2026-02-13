import json
import time
from pathlib import Path
from typing import Any, Dict

import click
import torch

from src.config import SimulationConfig, load_config
from src.core.agent import CognitiveAgent
from src.core.memory_layer import MemoryLayer
from src.core.network import CognitiveNetwork
from src.core.optimizer import CognitiveOptimizer
from src.environment.dataset_manager import DatasetManager
from src.environment.simulation_env import SimulationEnvironment
from src.performance_monitor import PerformanceMonitor
from src.utils.metrics import MetricsTracker
from src.utils.seed import set_global_seed


class CognitiveSimulation:
    def __init__(self, config_env: str = "development", fresh: bool = False, seed: int = 42):
        self.config: SimulationConfig = load_config(config_env)
        self.seed = int(seed if seed is not None else self.config.seed)
        set_global_seed(self.seed)

        self.device = CognitiveNetwork.resolve_device(self.config.network.device)

        self.memory = MemoryLayer(self.config.memory)
        self.network = CognitiveNetwork(self.config.network).to(self.device)
        self.optimizer = CognitiveOptimizer(
            self.network.parameters(),
            base_lr=self.config.network.learning_rate,
            weight_decay=self.config.network.weight_decay,
        )
        self.agent = CognitiveAgent(self.memory, self.network, self.optimizer, config=self.config.agent)

        self.data_manager = DatasetManager(self.config.model_dump())
        self.data_manager.load_data(seed=self.seed)
        self.env = SimulationEnvironment(self.data_manager.get_train_loader(batch_size=self.config.data.batch_size))

        self.monitor = PerformanceMonitor()
        self.metrics = MetricsTracker(artifact_dir=self.config.runtime.artifact_dir)
        self.checkpoint_path = self.config.runtime.checkpoint_path

        if not fresh:
            self.memory.load_state()
            payload = self.network.load_checkpoint(self.checkpoint_path)
            if payload and "optimizer_state" in payload:
                self.optimizer.load_state_dict(payload["optimizer_state"])

    @staticmethod
    def _batch_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
        diff = torch.mean(torch.abs(outputs - targets)).item()
        return float(max(0.0, 1.0 - diff))

    def _log_step(self, step: int, action: str, action_result: Dict[str, Any], outputs: torch.Tensor, targets: torch.Tensor) -> None:
        entropy = float(self.network.calculate_uncertainty())
        accuracy = self._batch_accuracy(outputs, targets)
        self.metrics.log_training_step(
            step=step,
            loss=float(action_result.get("loss", 0.0)),
            entropy=entropy,
            accuracy=accuracy,
            energy=float(self.agent.energy),
        )

        total_reviews = self.agent.review_successes + self.agent.review_failures
        success_rate = (self.agent.review_successes / total_reviews) if total_reviews else 0.0
        self.metrics.log_agent_state(
            step=step,
            action=action,
            energy=float(self.agent.energy),
            memory_count=len(self.memory.memories),
            entropy=entropy,
            review_success_rate=float(success_rate),
        )

        usage = self.monitor.get_resource_usage()
        self.metrics.log_system_health(
            step=step,
            memory_usage=float(usage.get("memory_rss_mb", 0.0)),
            cpu_usage=float(usage.get("cpu_percent", 0.0)),
        )

    def run_training_loop(self, steps: int = 100, sleep_seconds: float = 0.0) -> Dict[str, Any]:
        click.echo(f"Starting simulation: steps={steps}, seed={self.seed}, device={self.device}")
        for step in range(int(steps)):
            action = self.agent.decide_strategy()
            batch_inputs, batch_targets = self.env.get_next_flashcard()
            batch_inputs = batch_inputs.to(self.device)
            batch_targets = batch_targets.to(self.device)

            if action == "learn_new":
                result = self.agent.learn_new(batch_inputs, batch_targets)
                self.metrics.inc("learn_events")
            elif action == "review":
                result = self.agent.review()
                self.metrics.inc("review_events")
                if result.get("status") == "reviewed_success":
                    self.metrics.inc("review_success")
                if result.get("status") == "reviewed_failed":
                    self.metrics.inc("review_failure")
                    self.metrics.inc("forgetting_events")
            else:
                result = self.agent.sleep()
                self.metrics.inc("sleep_events")

            with torch.no_grad():
                outputs, _ = self.network(batch_inputs)
            self._log_step(step=step, action=action, action_result=result, outputs=outputs, targets=batch_targets)

            if sleep_seconds > 0:
                time.sleep(float(sleep_seconds))

            if (step + 1) % max(1, int(steps // 10)) == 0:
                click.echo(
                    f"step={step+1}/{steps} action={action} energy={self.agent.energy:.2f} "
                    f"memories={len(self.memory.memories)} due={self.memory.get_due_review_count()}"
                )

        self.persist_state()
        summary = self.metrics.summarize()
        summary["memory"] = self.memory.memory_stats()
        return summary

    def persist_state(self) -> None:
        self.memory.save_state()
        self.network.save_checkpoint(self.checkpoint_path, optimizer_state=self.optimizer.state_dict())

    def verify(self) -> Dict[str, Any]:
        issues = []
        if self.config.network.input_size <= 0 or self.config.network.output_size <= 0:
            issues.append("Network dimensions must be positive")
        if self.config.agent.energy_cost_learn <= 0:
            issues.append("energy_cost_learn must be > 0")
        if self.config.memory.decay_rate <= 0:
            issues.append("memory.decay_rate must be > 0")
        if self.config.agent.energy_reward_correct <= self.config.agent.energy_cost_review:
            issues.append("energy_reward_correct should exceed review cost to incentivize successful review")

        status = "ok" if not issues else "failed"
        return {"status": status, "issues": issues, "seed": self.seed, "device": str(self.device)}


@click.group()
def cli() -> None:
    """Cognitive Simulation CLI."""


@cli.command()
@click.option("--env", default="development", show_default=True, help="Environment configuration")
@click.option("--steps", default=50, show_default=True, help="Simulation steps")
@click.option("--fresh", is_flag=True, help="Do not load persisted memory/checkpoint")
@click.option("--seed", default=42, show_default=True, type=int, help="Deterministic seed")
@click.option("--sleep-seconds", default=0.0, show_default=True, type=float, help="Delay between steps")
def run(env: str, steps: int, fresh: bool, seed: int, sleep_seconds: float) -> None:
    """Run simulation loop."""
    sim = CognitiveSimulation(config_env=env, fresh=fresh, seed=seed)
    summary = sim.run_training_loop(steps=steps, sleep_seconds=sleep_seconds)
    click.echo(json.dumps(summary, indent=2))


@cli.command()
@click.option("--env", default="development", show_default=True, help="Environment configuration")
@click.option("--seed", default=42, show_default=True, type=int, help="Deterministic seed")
def verify(env: str, seed: int) -> None:
    """Validate configuration and key invariants."""
    sim = CognitiveSimulation(config_env=env, fresh=True, seed=seed)
    result = sim.verify()
    if result["status"] != "ok":
        click.echo(json.dumps(result, indent=2))
        raise SystemExit(1)
    click.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    cli()
