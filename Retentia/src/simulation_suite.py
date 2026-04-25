import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import SimulationConfig, load_config
from src.main import CognitiveSimulation


def _prepare_run_config(
    config_env: str,
    output_dir: str,
    run_name: str,
    policy_type: Optional[str] = None,
    force_cpu: bool = True,
    deterministic_data: bool = False,
    simulated_step_seconds: Optional[float] = None,
) -> SimulationConfig:
    config = load_config(config_env).model_copy(deep=True)
    run_root = Path(output_dir) / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    config.runtime.artifact_dir = str(run_root / "artifacts")
    config.runtime.checkpoint_path = str(run_root / "network_checkpoint.pt")
    config.runtime.tenant_db_path = str(run_root / "tenant_memory.db")
    config.memory.store_dir = str(run_root / "memory_store")
    config.memory.index_path = str(run_root / "memory_index.json")

    if policy_type:
        config.agent.policy_type = str(policy_type)
    if force_cpu:
        config.network.device = "cpu"
    if deterministic_data:
        config.data.environment = "deterministic"
        config.data.noise_std = 0.0
    if simulated_step_seconds is not None:
        config.runtime.simulated_step_seconds = float(simulated_step_seconds)

    return config


def _run_with_config(config: SimulationConfig, steps: int, seed: int, fresh: bool) -> Dict[str, Any]:
    sim = CognitiveSimulation(config_env="custom", config=config, fresh=fresh, seed=seed)
    summary = sim.run_training_loop(steps=int(steps), sleep_seconds=0.0)
    return {
        "summary": summary,
        "artifact_dir": str(config.runtime.artifact_dir),
        "checkpoint_path": str(config.runtime.checkpoint_path),
        "memory_index_path": str(config.memory.index_path),
        "policy_type": str(config.agent.policy_type),
        "seed": int(seed),
    }


def _extract_key_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
    memory = summary.get("memory", {}) if isinstance(summary, dict) else {}
    events = summary.get("events", {}) if isinstance(summary, dict) else {}
    return {
        "avg_loss": float(summary.get("avg_loss", 0.0)),
        "avg_accuracy": float(summary.get("avg_accuracy", 0.0)),
        "avg_entropy": float(summary.get("avg_entropy", 0.0)),
        "avg_energy": float(summary.get("avg_energy", 0.0)),
        "memory_count": float(memory.get("count", 0.0)),
        "memory_avg_retention": float(memory.get("avg_retention", 0.0)),
        "review_success": float(events.get("review_success", 0.0)),
        "review_failure": float(events.get("review_failure", 0.0)),
        "forgetting_events": float(events.get("forgetting_events", 0.0)),
    }


def run_policy_ab_comparison(
    config_env: str = "development",
    steps: int = 80,
    seed: int = 42,
    output_dir: str = "logs/experiments",
    simulated_step_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    heuristic_cfg = _prepare_run_config(
        config_env=config_env,
        output_dir=output_dir,
        run_name="policy_heuristic",
        policy_type="heuristic",
        simulated_step_seconds=simulated_step_seconds,
    )
    bandit_cfg = _prepare_run_config(
        config_env=config_env,
        output_dir=output_dir,
        run_name="policy_bandit",
        policy_type="bandit",
        simulated_step_seconds=simulated_step_seconds,
    )

    heuristic = _run_with_config(heuristic_cfg, steps=steps, seed=seed, fresh=True)
    bandit = _run_with_config(bandit_cfg, steps=steps, seed=seed, fresh=True)

    h_metrics = _extract_key_metrics(heuristic["summary"])
    b_metrics = _extract_key_metrics(bandit["summary"])
    metric_delta = {name: float(b_metrics[name] - h_metrics[name]) for name in h_metrics.keys()}

    return {
        "experiment": "policy_ab_comparison",
        "config_env": config_env,
        "steps": int(steps),
        "seed": int(seed),
        "runs": {
            "heuristic": heuristic,
            "bandit": bandit,
        },
        "comparison": {
            "heuristic_metrics": h_metrics,
            "bandit_metrics": b_metrics,
            "bandit_minus_heuristic": metric_delta,
        },
    }


def run_reproducibility_check(
    config_env: str = "development",
    steps: int = 50,
    seed: int = 42,
    output_dir: str = "logs/experiments",
    tolerance: float = 1e-9,
    simulated_step_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    first_cfg = _prepare_run_config(
        config_env=config_env,
        output_dir=output_dir,
        run_name="repro_run_1",
        deterministic_data=True,
        simulated_step_seconds=simulated_step_seconds,
    )
    second_cfg = _prepare_run_config(
        config_env=config_env,
        output_dir=output_dir,
        run_name="repro_run_2",
        deterministic_data=True,
        simulated_step_seconds=simulated_step_seconds,
    )

    first = _run_with_config(first_cfg, steps=steps, seed=seed, fresh=True)
    second = _run_with_config(second_cfg, steps=steps, seed=seed, fresh=True)

    first_metrics = _extract_key_metrics(first["summary"])
    second_metrics = _extract_key_metrics(second["summary"])
    deltas = {name: abs(float(first_metrics[name] - second_metrics[name])) for name in first_metrics.keys()}
    passed = all(delta <= float(tolerance) for delta in deltas.values())

    return {
        "experiment": "reproducibility_check",
        "config_env": config_env,
        "steps": int(steps),
        "seed": int(seed),
        "tolerance": float(tolerance),
        "passed": bool(passed),
        "first_run": first,
        "second_run": second,
        "deltas": deltas,
    }


def run_persistence_restart_check(
    config_env: str = "development",
    steps_before_restart: int = 40,
    steps_after_restart: int = 20,
    seed: int = 42,
    output_dir: str = "logs/experiments",
    simulated_step_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    config = _prepare_run_config(
        config_env=config_env,
        output_dir=output_dir,
        run_name="persistence_restart",
        simulated_step_seconds=simulated_step_seconds,
    )

    initial_run = _run_with_config(config, steps=steps_before_restart, seed=seed, fresh=True)
    initial_memory_count = float(initial_run["summary"].get("memory", {}).get("count", 0.0))

    resumed_sim = CognitiveSimulation(config_env="custom", config=config, fresh=False, seed=seed)
    loaded_memory_count = float(len(resumed_sim.memory.memories))
    resumed_summary = resumed_sim.run_training_loop(steps=int(steps_after_restart), sleep_seconds=0.0)

    checkpoint_exists = Path(config.runtime.checkpoint_path).exists()
    memory_index_exists = Path(config.memory.index_path).exists()
    passed = bool(
        loaded_memory_count >= initial_memory_count
        and checkpoint_exists
        and memory_index_exists
    )

    return {
        "experiment": "persistence_restart_check",
        "config_env": config_env,
        "seed": int(seed),
        "steps_before_restart": int(steps_before_restart),
        "steps_after_restart": int(steps_after_restart),
        "passed": passed,
        "initial_run": initial_run,
        "loaded_memory_count_before_resume": loaded_memory_count,
        "resumed_summary": resumed_summary,
        "checkpoint_exists": bool(checkpoint_exists),
        "memory_index_exists": bool(memory_index_exists),
    }


def run_standard_suite(
    config_env: str = "development",
    steps: int = 60,
    seed: int = 42,
    output_dir: str = "logs/experiments",
    simulated_step_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    policy_ab = run_policy_ab_comparison(
        config_env=config_env,
        steps=steps,
        seed=seed,
        output_dir=output_dir,
        simulated_step_seconds=simulated_step_seconds,
    )
    reproducibility = run_reproducibility_check(
        config_env=config_env,
        steps=max(10, steps // 2),
        seed=seed,
        output_dir=output_dir,
        simulated_step_seconds=simulated_step_seconds,
    )
    persistence = run_persistence_restart_check(
        config_env=config_env,
        steps_before_restart=steps,
        steps_after_restart=max(5, steps // 3),
        seed=seed,
        output_dir=output_dir,
        simulated_step_seconds=simulated_step_seconds,
    )

    report = {
        "suite": "standard",
        "config_env": config_env,
        "seed": int(seed),
        "steps": int(steps),
        "results": {
            "policy_ab": policy_ab,
            "reproducibility": reproducibility,
            "persistence_restart": persistence,
        },
    }

    report_path = root / "suite_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    report["report_path"] = str(report_path)
    return report


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Retentia experiment and validation suite")
    parser.add_argument(
        "scenario",
        choices=["ab-policy", "repro", "persistence", "all"],
        help="Which simulation scenario to run",
    )
    parser.add_argument("--env", default="development", help="Configuration environment")
    parser.add_argument("--steps", type=int, default=60, help="Primary step count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="logs/experiments", help="Output directory for suite artifacts")
    parser.add_argument(
        "--sim-step-seconds",
        type=float,
        default=None,
        help="Optional simulated seconds to advance memory-time per training step",
    )
    args = parser.parse_args(argv)

    if args.scenario == "ab-policy":
        result = run_policy_ab_comparison(
            config_env=args.env,
            steps=args.steps,
            seed=args.seed,
            output_dir=args.output_dir,
            simulated_step_seconds=args.sim_step_seconds,
        )
    elif args.scenario == "repro":
        result = run_reproducibility_check(
            config_env=args.env,
            steps=args.steps,
            seed=args.seed,
            output_dir=args.output_dir,
            simulated_step_seconds=args.sim_step_seconds,
        )
    elif args.scenario == "persistence":
        result = run_persistence_restart_check(
            config_env=args.env,
            steps_before_restart=args.steps,
            steps_after_restart=max(5, args.steps // 3),
            seed=args.seed,
            output_dir=args.output_dir,
            simulated_step_seconds=args.sim_step_seconds,
        )
    else:
        result = run_standard_suite(
            config_env=args.env,
            steps=args.steps,
            seed=args.seed,
            output_dir=args.output_dir,
            simulated_step_seconds=args.sim_step_seconds,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

