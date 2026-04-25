import pytest

from src.tenant.store import TenantMemoryStore


def test_tenant_store_record_and_queue(tmp_path) -> None:
    store = TenantMemoryStore(str(tmp_path / "tenant.db"))
    org = store.create_org(name="Acme", org_id="org_test")
    user = store.create_user(org_id=org["org_id"], email="u1@example.com", user_id="usr_test")
    store.upsert_concept(
        org_id=org["org_id"],
        concept_id="c1",
        title="Concept One",
        prompt="What is concept one?",
        answer="A test concept.",
        tags=["demo", "unit"],
    )

    store.record_attempt(user_id=user["user_id"], concept_id="c1", correct=True)
    store.record_attempt(user_id=user["user_id"], concept_id="c1", correct=False)
    queue = store.get_review_queue(user_id=user["user_id"], limit=10)

    assert len(queue) >= 1
    assert queue[0]["concept_id"] == "c1"
    assert queue[0]["concept"]["title"] == "Concept One"
    assert "reason_code" in queue[0]
    store.close()


def test_tenant_store_concepts_audit_and_roi(tmp_path) -> None:
    store = TenantMemoryStore(str(tmp_path / "tenant.db"))
    store.create_org(name="Acme", org_id="org_platform")
    store.create_user(org_id="org_platform", email="learner@example.com", user_id="usr_platform")

    concept = store.upsert_concept(
        org_id="org_platform",
        concept_id="billing_escalation",
        title="Billing escalation",
        prompt="What should support do during billing escalation?",
        answer="Follow the escalation playbook.",
        explanation="Ensures consistent support outcomes.",
        tags=["support", "billing"],
        difficulty=1.5,
        source="playbook",
    )
    assert concept["version"] == 1
    assert store.get_concept("org_platform", "billing_escalation")["tags"] == ["support", "billing"]

    updated = store.upsert_concept(
        org_id="org_platform",
        concept_id="billing_escalation",
        title="Billing escalation v2",
        prompt="What should support do during billing escalation?",
        answer="Follow the escalation playbook and document the result.",
    )
    assert updated["version"] == 2
    assert len(store.list_concepts("org_platform")) == 1

    store.record_attempt(user_id="usr_platform", concept_id="billing_escalation", correct=True)
    report = store.build_roi_report(
        org_id="org_platform",
        learners=1000,
        training_hours_saved_per_learner=2.0,
        cost_per_training_hour=50.0,
        annual_contract_value=25000.0,
    )
    assert report["gross_savings"] == 100000.0
    assert report["roi_multiple"] == 4.0

    audit_actions = {event["action"] for event in store.list_audit_events(org_id="org_platform", limit=20)}
    assert "concept_upserted" in audit_actions
    assert "attempt_recorded" in audit_actions
    assert "roi_report_generated" in audit_actions
    store.close()


def test_tenant_store_analytics(tmp_path) -> None:
    store = TenantMemoryStore(str(tmp_path / "tenant.db"))
    store.create_org(name="Beta", org_id="org_beta")
    store.create_user(org_id="org_beta", email="u2@example.com", user_id="usr_beta")

    store.record_attempt(user_id="usr_beta", concept_id="math", correct=True)
    store.record_attempt(user_id="usr_beta", concept_id="math", correct=True)
    store.record_attempt(user_id="usr_beta", concept_id="history", correct=False)

    user_metrics = store.compute_user_analytics(user_id="usr_beta", window_days=7)
    org_metrics = store.compute_org_analytics(org_id="org_beta", window_days=7)

    assert user_metrics["attempts"] == 3
    assert 0.0 <= user_metrics["retention_percentage"] <= 1.0
    assert org_metrics["users"] == 1
    store.close()


def test_tenant_store_pilot_setup_and_baseline(tmp_path) -> None:
    store = TenantMemoryStore(str(tmp_path / "tenant.db"))
    store.create_org(name="Pilot Org", org_id="org_pilot")
    users = [
        store.create_user(org_id="org_pilot", email=f"u{idx}@example.com", user_id=f"usr_{idx}")
        for idx in range(4)
    ]

    for idx, user in enumerate(users):
        store.record_attempt(
            user_id=user["user_id"],
            concept_id=f"concept_{idx}",
            correct=(idx % 2 == 0),
        )

    setup = store.setup_pilot(org_id="org_pilot", name="Support Team Pilot", treatment_ratio=0.5, random_seed=7)
    assert setup["control_count"] == 2
    assert setup["treatment_count"] == 2

    snapshot = store.get_pilot_run(setup["pilot_id"])
    assert snapshot["pilot_id"] == setup["pilot_id"]
    assert snapshot["latest_baseline"] is None

    baseline = store.capture_pilot_baseline(pilot_id=setup["pilot_id"], window_days=30)
    assert baseline["pilot_id"] == setup["pilot_id"]
    assert baseline["control_metrics"]["users"] == 2
    assert baseline["treatment_metrics"]["users"] == 2

    refreshed_snapshot = store.get_pilot_run(setup["pilot_id"])
    assert refreshed_snapshot["latest_baseline"] is not None
    assert refreshed_snapshot["latest_baseline"]["baseline_id"] == baseline["baseline_id"]
    store.close()


def test_create_user_requires_existing_org(tmp_path) -> None:
    store = TenantMemoryStore(str(tmp_path / "tenant.db"))
    with pytest.raises(ValueError, match="organization_not_found"):
        store.create_user(org_id="org_missing", email="missing@example.com", user_id="usr_missing")
    store.close()


def test_record_attempt_requires_existing_user(tmp_path) -> None:
    store = TenantMemoryStore(str(tmp_path / "tenant.db"))
    store.create_org(name="Acme", org_id="org_test")
    with pytest.raises(ValueError, match="user_not_found"):
        store.record_attempt(user_id="usr_missing", concept_id="c1", correct=True)
    store.close()


def test_duplicate_org_and_user_conflicts_raise_clear_errors(tmp_path) -> None:
    store = TenantMemoryStore(str(tmp_path / "tenant.db"))
    store.create_org(name="Acme", org_id="org_dup")
    with pytest.raises(ValueError, match="organization_already_exists"):
        store.create_org(name="Acme Duplicate", org_id="org_dup")

    store.create_user(org_id="org_dup", email="user@example.com", user_id="usr_dup")
    with pytest.raises(ValueError, match="user_already_exists"):
        store.create_user(org_id="org_dup", email="user2@example.com", user_id="usr_dup")
    with pytest.raises(ValueError, match="user_email_already_exists"):
        store.create_user(org_id="org_dup", email="user@example.com", user_id="usr_other")
    store.close()


def test_capture_pilot_baseline_computes_each_user_once(tmp_path, monkeypatch) -> None:
    store = TenantMemoryStore(str(tmp_path / "tenant.db"))
    store.create_org(name="Pilot Org", org_id="org_once")
    users = [
        store.create_user(org_id="org_once", email=f"u{idx}@example.com", user_id=f"usr_once_{idx}")
        for idx in range(4)
    ]

    for idx, user in enumerate(users):
        store.record_attempt(
            user_id=user["user_id"],
            concept_id=f"concept_{idx}",
            correct=(idx % 2 == 0),
        )

    setup = store.setup_pilot(org_id="org_once", name="Efficient Baseline", treatment_ratio=0.5, random_seed=11)

    call_count = {"count": 0}
    original_compute = store._compute_user_analytics_at

    def counted_compute(*args, **kwargs):
        call_count["count"] += 1
        return original_compute(*args, **kwargs)

    monkeypatch.setattr(store, "_compute_user_analytics_at", counted_compute)

    baseline = store.capture_pilot_baseline(pilot_id=setup["pilot_id"], window_days=30)

    assert baseline["control_metrics"]["users"] == 2
    assert baseline["treatment_metrics"]["users"] == 2
    assert call_count["count"] == len(users)
    store.close()
