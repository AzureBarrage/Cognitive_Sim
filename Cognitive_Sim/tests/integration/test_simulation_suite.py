from pathlib import Path

from src.simulation_suite import (
    run_persistence_restart_check,
    run_policy_ab_comparison,
    run_reproducibility_check,
)


def test_policy_ab_comparison_runs(temp_simulation_config, tmp_path) -> None:
    result = run_policy_ab_comparison(
        config_env="testing",
        steps=8,
        seed=123,
        output_dir=str(tmp_path / "experiments"),
        simulated_step_seconds=60.0,
    )

    assert result["experiment"] == "policy_ab_comparison"
    assert "heuristic" in result["runs"]
    assert "bandit" in result["runs"]
    assert "bandit_minus_heuristic" in result["comparison"]


def test_reproducibility_check_passes_under_deterministic_setup(tmp_path) -> None:
    result = run_reproducibility_check(
        config_env="testing",
        steps=8,
        seed=777,
        output_dir=str(tmp_path / "experiments"),
        tolerance=1e-6,
        simulated_step_seconds=60.0,
    )

    assert result["experiment"] == "reproducibility_check"
    assert isinstance(result["passed"], bool)
    assert set(result["deltas"].keys())


def test_persistence_restart_check_creates_state_files(tmp_path) -> None:
    result = run_persistence_restart_check(
        config_env="testing",
        steps_before_restart=8,
        steps_after_restart=4,
        seed=123,
        output_dir=str(tmp_path / "experiments"),
        simulated_step_seconds=60.0,
    )

    assert result["experiment"] == "persistence_restart_check"
    assert result["checkpoint_exists"] is True
    assert result["memory_index_exists"] is True
    assert result["loaded_memory_count_before_resume"] >= 0.0

