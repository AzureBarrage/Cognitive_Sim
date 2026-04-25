from src.tenant.store import TenantMemoryStore


def test_tenant_store_record_and_queue(tmp_path) -> None:
    store = TenantMemoryStore(str(tmp_path / "tenant.db"))
    org = store.create_org(name="Acme", org_id="org_test")
    user = store.create_user(org_id=org["org_id"], email="u1@example.com", user_id="usr_test")

    store.record_attempt(user_id=user["user_id"], concept_id="c1", correct=True)
    store.record_attempt(user_id=user["user_id"], concept_id="c1", correct=False)
    queue = store.get_review_queue(user_id=user["user_id"], limit=10)

    assert len(queue) >= 1
    assert queue[0]["concept_id"] == "c1"
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
