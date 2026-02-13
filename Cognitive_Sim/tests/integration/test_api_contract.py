from fastapi.testclient import TestClient
import uuid

from src.api import app


def test_status_endpoint_contract() -> None:
    with TestClient(app) as client:
        response = client.get("/status")
        assert response.status_code == 200
        payload = response.json()
        assert "energy" in payload
        assert "memory_count" in payload


def test_teach_ask_sleep_contracts() -> None:
    with TestClient(app) as client:
        teach = client.post(
            "/teach",
            json={
                "input_data": [0.1] * 10,
                "target_data": [0.2] * 5,
            },
        )
        assert teach.status_code == 200
        teach_payload = teach.json()
        assert teach_payload["status"] in {"learned", "failed"}

        ask = client.post("/ask", json={"input_data": [0.1] * 10})
        assert ask.status_code == 200
        ask_payload = ask.json()
        assert "prediction" in ask_payload

        sleep = client.post("/sleep")
        assert sleep.status_code == 200
        assert sleep.json()["status"] == "slept"


def test_new_endpoints_memories_and_metrics() -> None:
    with TestClient(app) as client:
        memories = client.get("/memories?limit=10")
        assert memories.status_code == 200
        assert "items" in memories.json()

        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "trends" in metrics.json()


def test_reset_guard() -> None:
    with TestClient(app) as client:
        denied = client.post("/reset")
        assert denied.status_code == 403


def test_multitenant_endpoints_contract() -> None:
    org_id = "org_" + uuid.uuid4().hex[:8]
    user_id = "usr_" + uuid.uuid4().hex[:8]
    with TestClient(app) as client:
        create_org = client.post("/organizations", json={"name": "Acme QA", "org_id": org_id})
        assert create_org.status_code == 200

        create_user = client.post(
            "/users",
            json={
                "org_id": org_id,
                "email": "qa@example.com",
                "user_id": user_id,
            },
        )
        assert create_user.status_code == 200

        attempt = client.post(
            "/record-attempt",
            json={
                "user_id": user_id,
                "concept_id": "concept_1",
                "correct": True,
                "response_ms": 800,
            },
        )
        assert attempt.status_code == 200
        assert "risk_score" in attempt.json()

        queue = client.get(f"/review-queue?user_id={user_id}&limit=10")
        assert queue.status_code == 200
        assert queue.json()["user_id"] == user_id

        user_analytics = client.get(f"/analytics?user_id={user_id}")
        assert user_analytics.status_code == 200
        assert user_analytics.json()["scope"] == "user"

        org_analytics = client.get(f"/analytics?org_id={org_id}")
        assert org_analytics.status_code == 200
        assert org_analytics.json()["scope"] == "organization"


def test_pilot_gate_endpoints_contract() -> None:
    with TestClient(app) as client:
        org_id = "org_pilot_" + uuid.uuid4().hex[:8]
        create_org = client.post("/organizations", json={"name": "Pilot Org", "org_id": org_id})
        assert create_org.status_code == 200

        evaluate = client.post(
            "/pilot/evaluate",
            json={
                "org_id": org_id,
                "sample_size": 300,
                "onboarding_hours": 25.0,
                "retained_mastery_treatment": 0.80,
                "retained_mastery_control": 0.70,
                "forgetting_velocity_treatment": 0.20,
                "forgetting_velocity_control": 0.35,
                "review_efficiency_treatment": 0.74,
                "review_efficiency_control": 0.62,
            },
        )
        assert evaluate.status_code == 200
        payload = evaluate.json()
        assert "go_decision" in payload
        assert payload["sample_size"] == 300

        history = client.get(f"/pilot/history?org_id={org_id}&limit=10")
        assert history.status_code == 200
        h = history.json()
        assert h["count"] >= 1
        assert isinstance(h["items"], list)
