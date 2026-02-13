from src.config import AgentConfig
from src.core.interfaces import PolicyContext
from src.core.policy import ContextualBanditPolicy, HeuristicPolicy, build_policy


def test_heuristic_policy_prefers_sleep_when_low_energy() -> None:
    policy = HeuristicPolicy(entropy_review_threshold=0.7, sleep_when_energy_below=10.0, review_due_minimum=2)
    context = PolicyContext(energy=5.0, max_energy=100.0, due_count=0, entropy=0.1, memory_count=5)
    assert policy.select_action(context) == "sleep"


def test_heuristic_policy_prefers_review_when_due_items_exist() -> None:
    policy = HeuristicPolicy(entropy_review_threshold=0.7, sleep_when_energy_below=10.0, review_due_minimum=2)
    context = PolicyContext(energy=80.0, max_energy=100.0, due_count=3, entropy=0.1, memory_count=5)
    assert policy.select_action(context) == "review"


def test_bandit_policy_update_changes_q_values() -> None:
    policy = ContextualBanditPolicy(epsilon=0.0, alpha=0.5)
    context = PolicyContext(energy=90.0, max_energy=100.0, due_count=1, entropy=0.2, memory_count=10)
    before = policy.q_values["learn_new"]
    policy.update(context, "learn_new", reward=2.0)
    assert policy.q_values["learn_new"] > before


def test_build_policy_returns_expected_implementation() -> None:
    cfg = AgentConfig(policy_type="bandit")
    policy = build_policy(cfg)
    assert isinstance(policy, ContextualBanditPolicy)
