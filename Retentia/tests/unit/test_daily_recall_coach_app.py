import httpx

from src.daily_recall_coach_app import bootstrap_learner, daily_session


def _mock_transport(handler):
    return httpx.MockTransport(handler)


def test_daily_session_request_with_attempt_payload() -> None:
    captured = {"body": ""}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/daily-session":
            captured["body"] = request.read().decode("utf-8")
            return httpx.Response(
                200,
                json={
                    "user_id": "usr_1",
                    "queue": [],
                    "due_count": 0,
                    "total_count": 0,
                    "next_item": None,
                    "analytics": {"attempts": 0},
                    "recorded_attempt": None,
                    "recommended_action": "seed_attempts",
                },
            )
        return httpx.Response(404, json={})

    transport = _mock_transport(handler)
    payload = daily_session(
        "http://localhost:8000",
        user_id="usr_1",
        limit=10,
        window_days=7,
        now=123.0,
        attempt={"concept_id": "c1", "correct": True, "response_ms": 800.0},
        transport=transport,
    )

    assert payload["recommended_action"] == "seed_attempts"
    assert "\"attempt\"" in captured["body"]
    assert "\"concept_id\":\"c1\"" in captured["body"]


def test_bootstrap_learner_is_idempotent_on_existing_entities() -> None:
    calls = {"org": 0, "user": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/organizations":
            calls["org"] += 1
            return httpx.Response(400, json={"detail": "organization_already_exists"})
        if request.url.path == "/users":
            calls["user"] += 1
            return httpx.Response(400, json={"detail": "user_already_exists"})
        return httpx.Response(404, json={})

    transport = _mock_transport(handler)
    result = bootstrap_learner(
        "http://localhost:8000",
        org_id="org_daily",
        org_name="Daily Org",
        user_id="usr_daily",
        email="daily@example.com",
        transport=transport,
    )

    assert calls["org"] == 1
    assert calls["user"] == 1
    assert result["organization"]["status"] == "existing"
    assert result["user"]["status"] == "existing"

