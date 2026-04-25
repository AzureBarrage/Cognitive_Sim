import httpx
import pytest

from src.frontend_app import (
    COMPLIANCE_DEMO_TOPICS,
    ask,
    build_uniform_vector_csv,
    compliance_risk_distribution,
    compliance_status,
    create_organization,
    create_user,
    enrich_queue_for_compliance,
    evaluate_pilot,
    get_analytics,
    get_metrics,
    get_pilot_run,
    list_memories,
    pilot_history,
    capture_pilot_baseline,
    get_review_queue,
    get_status,
    normalize_base_url,
    parse_float_list,
    record_attempt,
    risk_level,
    seed_demo_data,
    sleep,
    setup_pilot,
    teach,
)


def _mock_transport(handler):
    return httpx.MockTransport(handler)


def test_normalize_base_url_defaults_and_trims() -> None:
    assert normalize_base_url("") == "http://localhost:8000"
    assert normalize_base_url(" http://localhost:9000/ ") == "http://localhost:9000"


def test_parse_float_list_validation() -> None:
    assert parse_float_list("1,2,3", expected_len=3) == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        parse_float_list("1,2", expected_len=3)


def test_build_uniform_vector_csv() -> None:
    assert build_uniform_vector_csv(3, value=0.5) == "0.5,0.5,0.5"


def test_compliance_demo_topics_and_risk_helpers() -> None:
    topic_ids = {topic["concept_id"] for topic in COMPLIANCE_DEMO_TOPICS}
    assert {"hipaa_001", "hipaa_002", "sec_001", "sec_002"}.issubset(topic_ids)
    assert risk_level(0.8) == "🔴 High Risk"
    assert risk_level(0.5) == "🟡 Medium Risk"
    assert risk_level(0.1) == "🟢 Low Risk"
    assert compliance_status(0.9) == "✅ Stable"
    assert compliance_status(0.7) == "⚠️ At Risk"
    assert compliance_status(0.9, high_risk_count=1) == "🔴 Critical"

    queue = enrich_queue_for_compliance(
        [
            {
                "concept_id": "hipaa_001",
                "risk_score": 0.75,
                "retention": 0.25,
                "concept": {"title": "Protected Health Information (PHI)"},
                "reason": "Retention has fallen below target.",
            }
        ]
    )
    assert queue[0]["compliance_topic"] == "Protected Health Information (PHI)"
    assert queue[0]["risk_level"] == "🔴 High Risk"
    assert compliance_risk_distribution(queue) == {"high": 1, "medium": 0, "low": 0}


def test_get_status_and_metrics_requests() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/status":
            return httpx.Response(200, json={"energy": 100.0})
        if request.url.path == "/metrics":
            return httpx.Response(200, json={"trends": {}, "counters": {}})
        return httpx.Response(404, json={"error": "not found"})

    transport = _mock_transport(handler)
    assert get_status("http://localhost:8000", transport=transport)["energy"] == 100.0
    assert "trends" in get_metrics("http://localhost:8000", transport=transport)


def test_learning_and_sleep_actions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/teach":
            payload = request.read().decode("utf-8")
            assert "input_data" in payload
            return httpx.Response(200, json={"status": "learned"})
        if request.url.path == "/ask":
            return httpx.Response(200, json={"prediction": [0.1], "uncertainty": 0.2, "recall_status": "success", "energy": 50.0})
        if request.url.path == "/sleep":
            return httpx.Response(200, json={"status": "slept"})
        return httpx.Response(404, json={})

    transport = _mock_transport(handler)
    teach_result = teach("http://localhost:8000", [0.1] * 10, [0.2] * 5, transport=transport)
    ask_result = ask("http://localhost:8000", [0.1] * 10, memory_key="mem_1", transport=transport)
    sleep_result = sleep("http://localhost:8000", transport=transport)

    assert teach_result["status"] == "learned"
    assert ask_result["recall_status"] == "success"
    assert sleep_result["status"] == "slept"


def test_tenant_and_pilot_flows() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/organizations":
            return httpx.Response(200, json={"org_id": "org_1", "name": "Acme", "created_at": 1.0})
        if request.url.path == "/users":
            return httpx.Response(200, json={"user_id": "usr_1", "org_id": "org_1", "email": "a@b.com", "name": None, "created_at": 1.0})
        if request.url.path == "/record-attempt":
            return httpx.Response(200, json={"user_id": "usr_1", "concept_id": "c1", "correct": True})
        if request.url.path == "/review-queue":
            return httpx.Response(200, json={"user_id": "usr_1", "items": []})
        if request.url.path == "/analytics":
            return httpx.Response(200, json={"scope": "user", "data": {"attempts": 1}})
        if request.url.path == "/pilot/evaluate":
            return httpx.Response(200, json={"go_decision": True, "reasons": ["ok"]})
        if request.url.path == "/pilot/setup":
            return httpx.Response(200, json={"pilot_id": "pilot_1", "control_count": 1, "treatment_count": 1})
        if request.url.path == "/pilot/run":
            return httpx.Response(200, json={"pilot_id": "pilot_1", "status": "configured"})
        if request.url.path == "/pilot/baseline":
            return httpx.Response(200, json={"pilot_id": "pilot_1", "baseline_id": 1})
        if request.url.path == "/memories":
            return httpx.Response(200, json={"items": [], "count": 0, "offset": 0, "limit": 20})
        if request.url.path == "/pilot/history":
            return httpx.Response(200, json={"items": [], "count": 0})
        return httpx.Response(404, json={})

    transport = _mock_transport(handler)
    org = create_organization("http://localhost:8000", name="Acme", transport=transport)
    user = create_user("http://localhost:8000", org_id="org_1", email="a@b.com", transport=transport)
    attempt = record_attempt("http://localhost:8000", user_id="usr_1", concept_id="c1", correct=True, transport=transport)
    queue = get_review_queue("http://localhost:8000", user_id="usr_1", transport=transport)
    analytics = get_analytics("http://localhost:8000", user_id="usr_1", transport=transport)
    pilot = evaluate_pilot(
        "http://localhost:8000",
        payload={
            "sample_size": 100,
            "onboarding_hours": 10.0,
            "retained_mastery_treatment": 0.8,
            "retained_mastery_control": 0.7,
            "forgetting_velocity_treatment": 0.2,
            "forgetting_velocity_control": 0.3,
            "review_efficiency_treatment": 0.7,
            "review_efficiency_control": 0.6,
        },
        transport=transport,
    )
    setup = setup_pilot(
        "http://localhost:8000",
        payload={"org_id": "org_1", "name": "Pilot", "treatment_ratio": 0.5, "random_seed": 42},
        transport=transport,
    )
    run = get_pilot_run("http://localhost:8000", pilot_id="pilot_1", transport=transport)
    baseline = capture_pilot_baseline(
        "http://localhost:8000",
        payload={"pilot_id": "pilot_1", "window_days": 30},
        transport=transport,
    )
    memories = list_memories("http://localhost:8000", offset=0, limit=20, stable=None, transport=transport)
    history = pilot_history("http://localhost:8000", org_id="org_1", limit=20, transport=transport)

    assert org["org_id"] == "org_1"
    assert user["user_id"] == "usr_1"
    assert attempt["correct"] is True
    assert queue["user_id"] == "usr_1"
    assert analytics["scope"] == "user"
    assert pilot["go_decision"] is True
    assert setup["pilot_id"] == "pilot_1"
    assert run["pilot_id"] == "pilot_1"
    assert baseline["baseline_id"] == 1
    assert memories["count"] == 0
    assert history["count"] == 0


def test_record_attempt_can_send_attempted_at() -> None:
    seen_payload = {"attempted_at": None}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/record-attempt":
            body = request.read().decode("utf-8")
            seen_payload["attempted_at"] = "attempted_at" in body
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404, json={})

    transport = _mock_transport(handler)
    result = record_attempt(
        "http://localhost:8000",
        user_id="usr_1",
        concept_id="c1",
        correct=True,
        attempted_at=123.0,
        transport=transport,
    )
    assert result["ok"] is True
    assert seen_payload["attempted_at"] is True


def test_seed_demo_data_flow() -> None:
    called_paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        called_paths.append(request.url.path)
        if request.url.path == "/organizations":
            return httpx.Response(200, json={"org_id": "org_demo", "name": "Demo Org", "created_at": 1.0})
        if request.url.path == "/users":
            return httpx.Response(200, json={"user_id": "usr_demo", "org_id": "org_demo", "email": "demo@example.com", "name": None, "created_at": 1.0})
        if request.url.path == "/record-attempt":
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/concepts":
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/sleep":
            return httpx.Response(200, json={"status": "slept"})
        if request.url.path == "/review-queue":
            return httpx.Response(200, json={"user_id": "usr_demo", "items": []})
        if request.url.path == "/analytics":
            return httpx.Response(200, json={"scope": "user", "data": {"attempts": 20}})
        return httpx.Response(404, json={})

    transport = _mock_transport(handler)

    # monkeypatching wrappers to pass test transport through the seeder call chain
    from src import frontend_app

    original_create_org = frontend_app.create_organization
    original_create_user = frontend_app.create_user
    original_upsert_concept = frontend_app.upsert_concept
    original_record_attempt = frontend_app.record_attempt
    original_sleep = frontend_app.sleep
    original_queue = frontend_app.get_review_queue
    original_analytics = frontend_app.get_analytics

    frontend_app.create_organization = lambda *a, **k: original_create_org(*a, transport=transport, **k)
    frontend_app.create_user = lambda *a, **k: original_create_user(*a, transport=transport, **k)
    frontend_app.upsert_concept = lambda *a, **k: original_upsert_concept(*a, transport=transport, **k)
    frontend_app.record_attempt = lambda *a, **k: original_record_attempt(*a, transport=transport, **k)
    frontend_app.sleep = lambda *a, **k: original_sleep(*a, transport=transport, **k)
    frontend_app.get_review_queue = lambda *a, **k: original_queue(*a, transport=transport, **k)
    frontend_app.get_analytics = lambda *a, **k: original_analytics(*a, transport=transport, **k)

    try:
        result = seed_demo_data(
            "http://localhost:8000",
            org_id="org_demo",
            user_id="usr_demo",
            email="demo@example.com",
            concept_count=20,
            backdate_minutes_step=2.0,
        )
    finally:
        frontend_app.create_organization = original_create_org
        frontend_app.create_user = original_create_user
        frontend_app.upsert_concept = original_upsert_concept
        frontend_app.record_attempt = original_record_attempt
        frontend_app.sleep = original_sleep
        frontend_app.get_review_queue = original_queue
        frontend_app.get_analytics = original_analytics

    assert result["attempts_seeded"] == 20
    assert called_paths.count("/concepts") == 20
    assert called_paths.count("/record-attempt") == 20

