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
