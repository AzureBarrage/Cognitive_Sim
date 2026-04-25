import json
import time
from typing import Any, Dict, List, Optional

import httpx

from src.config import SimulationConfig, load_config


def normalize_base_url(base_url: str) -> str:
    normalized = str(base_url or "").strip()
    if not normalized:
        return "http://localhost:8000"
    return normalized.rstrip("/")


def parse_float_list(raw: str, expected_len: Optional[int] = None) -> List[float]:
    values: List[float] = []
    chunks = [chunk.strip() for chunk in str(raw).split(",") if chunk.strip()]
    for chunk in chunks:
        values.append(float(chunk))

    if expected_len is not None and len(values) != int(expected_len):
        raise ValueError(f"Expected {expected_len} values but received {len(values)}")
    return values


def build_uniform_vector_csv(length: int, value: float = 0.1) -> str:
    return ",".join([str(float(value)) for _ in range(max(0, int(length)))])


def perform_request(
    base_url: str,
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout_seconds: float = 10.0,
    transport: Optional[httpx.BaseTransport] = None,
) -> Dict[str, Any]:
    url = f"{normalize_base_url(base_url)}{path}"
    with httpx.Client(timeout=float(timeout_seconds), transport=transport) as client:
        response = client.request(method=method.upper(), url=url, json=payload, params=params)
    response.raise_for_status()

    try:
        data = response.json()
        if isinstance(data, dict):
            return data
        return {"data": data}
    except Exception:
        return {"raw": response.text}


def get_status(base_url: str, timeout_seconds: float = 10.0, transport: Optional[httpx.BaseTransport] = None) -> Dict[str, Any]:
    return perform_request(base_url, "GET", "/status", timeout_seconds=timeout_seconds, transport=transport)


def get_metrics(base_url: str, timeout_seconds: float = 10.0, transport: Optional[httpx.BaseTransport] = None) -> Dict[str, Any]:
    return perform_request(base_url, "GET", "/metrics", timeout_seconds=timeout_seconds, transport=transport)


def teach(
    base_url: str,
    input_data: List[float],
    target_data: List[float],
    timeout_seconds: float = 10.0,
    transport: Optional[httpx.BaseTransport] = None,
) -> Dict[str, Any]:
    payload = {"input_data": input_data, "target_data": target_data}
    return perform_request(base_url, "POST", "/teach", payload=payload, timeout_seconds=timeout_seconds, transport=transport)


def ask(
    base_url: str,
    input_data: List[float],
    memory_key: Optional[str] = None,
    timeout_seconds: float = 10.0,
    transport: Optional[httpx.BaseTransport] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"input_data": input_data}
    if memory_key:
        payload["memory_key"] = memory_key
    return perform_request(base_url, "POST", "/ask", payload=payload, timeout_seconds=timeout_seconds, transport=transport)


def sleep(base_url: str, timeout_seconds: float = 10.0, transport: Optional[httpx.BaseTransport] = None) -> Dict[str, Any]:
    return perform_request(base_url, "POST", "/sleep", timeout_seconds=timeout_seconds, transport=transport)


def create_organization(
    base_url: str,
    name: str,
    org_id: Optional[str] = None,
    timeout_seconds: float = 10.0,
    transport: Optional[httpx.BaseTransport] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"name": name}
    if org_id:
        payload["org_id"] = org_id
    return perform_request(base_url, "POST", "/organizations", payload=payload, timeout_seconds=timeout_seconds, transport=transport)


def create_user(
    base_url: str,
    org_id: str,
    email: str,
    user_id: Optional[str] = None,
    name: Optional[str] = None,
    timeout_seconds: float = 10.0,
    transport: Optional[httpx.BaseTransport] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"org_id": org_id, "email": email}
    if user_id:
        payload["user_id"] = user_id
    if name:
        payload["name"] = name
    return perform_request(base_url, "POST", "/users", payload=payload, timeout_seconds=timeout_seconds, transport=transport)


def record_attempt(
    base_url: str,
    user_id: str,
    concept_id: str,
    correct: bool,
    response_ms: Optional[float] = None,
    attempted_at: Optional[float] = None,
    timeout_seconds: float = 10.0,
    transport: Optional[httpx.BaseTransport] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "user_id": user_id,
        "concept_id": concept_id,
        "correct": bool(correct),
    }
    if response_ms is not None:
        payload["response_ms"] = float(response_ms)
    if attempted_at is not None:
        payload["attempted_at"] = float(attempted_at)
    return perform_request(base_url, "POST", "/record-attempt", payload=payload, timeout_seconds=timeout_seconds, transport=transport)


def seed_demo_data(
    base_url: str,
    org_id: str,
    user_id: str,
    email: str,
    concept_count: int = 20,
    backdate_minutes_step: float = 1.0,
    timeout_seconds: float = 10.0,
) -> Dict[str, Any]:
    created_org = create_organization(base_url, name="Demo Org", org_id=org_id, timeout_seconds=timeout_seconds)
    created_user = create_user(base_url, org_id=org_id, email=email, user_id=user_id, timeout_seconds=timeout_seconds)

    now = time.time()
    attempts: List[Dict[str, Any]] = []
    step_seconds = max(0.0, float(backdate_minutes_step) * 60.0)
    for idx in range(max(1, int(concept_count))):
        concept_id = f"concept_{idx+1:03d}"
        correct = (idx % 3) != 0
        attempted_at = now - ((max(1, int(concept_count)) - idx) * step_seconds)
        attempts.append(
            record_attempt(
                base_url,
                user_id=user_id,
                concept_id=concept_id,
                correct=correct,
                response_ms=700.0 + (idx * 10.0),
                attempted_at=attempted_at,
                timeout_seconds=timeout_seconds,
            )
        )

    slept = sleep(base_url, timeout_seconds=timeout_seconds)
    queue = get_review_queue(base_url, user_id=user_id, limit=20, timeout_seconds=timeout_seconds)
    analytics = get_analytics(base_url, user_id=user_id, timeout_seconds=timeout_seconds)

    return {
        "organization": created_org,
        "user": created_user,
        "attempts_seeded": len(attempts),
        "latest_attempt": attempts[-1] if attempts else None,
        "sleep": slept,
        "queue": queue,
        "analytics": analytics,
    }


def get_review_queue(
    base_url: str,
    user_id: str,
    limit: int = 20,
    timeout_seconds: float = 10.0,
    transport: Optional[httpx.BaseTransport] = None,
) -> Dict[str, Any]:
    return perform_request(
        base_url,
        "GET",
        "/review-queue",
        params={"user_id": user_id, "limit": int(limit)},
        timeout_seconds=timeout_seconds,
        transport=transport,
    )


def get_analytics(
    base_url: str,
    user_id: Optional[str] = None,
    org_id: Optional[str] = None,
    window_days: Optional[int] = None,
    timeout_seconds: float = 10.0,
    transport: Optional[httpx.BaseTransport] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if user_id:
        params["user_id"] = user_id
    if org_id:
        params["org_id"] = org_id
    if window_days is not None:
        params["window_days"] = int(window_days)
    return perform_request(base_url, "GET", "/analytics", params=params, timeout_seconds=timeout_seconds, transport=transport)


def list_memories(
    base_url: str,
    offset: int = 0,
    limit: int = 50,
    stable: Optional[bool] = None,
    timeout_seconds: float = 10.0,
    transport: Optional[httpx.BaseTransport] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "offset": int(offset),
        "limit": int(limit),
    }
    if stable is not None:
        params["stable"] = bool(stable)
    return perform_request(base_url, "GET", "/memories", params=params, timeout_seconds=timeout_seconds, transport=transport)


def pilot_history(
    base_url: str,
    org_id: Optional[str] = None,
    limit: int = 50,
    timeout_seconds: float = 10.0,
    transport: Optional[httpx.BaseTransport] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"limit": int(limit)}
    if org_id:
        params["org_id"] = org_id
    return perform_request(base_url, "GET", "/pilot/history", params=params, timeout_seconds=timeout_seconds, transport=transport)


def setup_pilot(
    base_url: str,
    payload: Dict[str, Any],
    timeout_seconds: float = 10.0,
    transport: Optional[httpx.BaseTransport] = None,
) -> Dict[str, Any]:
    return perform_request(base_url, "POST", "/pilot/setup", payload=payload, timeout_seconds=timeout_seconds, transport=transport)


def get_pilot_run(
    base_url: str,
    pilot_id: str,
    timeout_seconds: float = 10.0,
    transport: Optional[httpx.BaseTransport] = None,
) -> Dict[str, Any]:
    return perform_request(
        base_url,
        "GET",
        "/pilot/run",
        params={"pilot_id": pilot_id},
        timeout_seconds=timeout_seconds,
        transport=transport,
    )


def capture_pilot_baseline(
    base_url: str,
    payload: Dict[str, Any],
    timeout_seconds: float = 10.0,
    transport: Optional[httpx.BaseTransport] = None,
) -> Dict[str, Any]:
    return perform_request(base_url, "POST", "/pilot/baseline", payload=payload, timeout_seconds=timeout_seconds, transport=transport)


def evaluate_pilot(
    base_url: str,
    payload: Dict[str, Any],
    timeout_seconds: float = 10.0,
    transport: Optional[httpx.BaseTransport] = None,
) -> Dict[str, Any]:
    return perform_request(base_url, "POST", "/pilot/evaluate", payload=payload, timeout_seconds=timeout_seconds, transport=transport)


def _show_result(st: Any, label: str, result: Dict[str, Any]) -> None:
    st.success(f"{label} succeeded")
    st.json(result)


def _safe_action(st: Any, label: str, action: Any) -> None:
    try:
        result = action()
        _show_result(st, label, result)
    except Exception as exc:
        st.error(f"{label} failed: {exc}")


def render_dashboard(st: Any, config: Optional[SimulationConfig] = None) -> None:
    st.set_page_config(page_title="Cognitive_Sim Dashboard", layout="wide")
    cfg = config or load_config("development")

    st.title("Cognitive_Sim Demo Dashboard")
    st.caption("Interactive controls for key API flows and buyer demo operations.")

    base_url = st.sidebar.text_input("API Base URL", value="http://localhost:8000")
    timeout_seconds = st.sidebar.number_input("Request timeout (seconds)", min_value=1.0, max_value=120.0, value=15.0)

    st.sidebar.subheader("Demo Seeder")
    seed_org_id = st.sidebar.text_input("seed org_id", value="org_demo")
    seed_user_id = st.sidebar.text_input("seed user_id", value="usr_demo")
    seed_email = st.sidebar.text_input("seed email", value="demo@example.com")
    seed_concepts = int(st.sidebar.number_input("seed concept count", min_value=5, max_value=200, value=20))
    seed_backdate_step = float(st.sidebar.number_input("seed backdate minutes/attempt", min_value=0.0, max_value=60.0, value=2.0))
    if st.sidebar.button("Seed Demo Data", use_container_width=True):
        _safe_action(
            st,
            "seed-demo-data",
            lambda: seed_demo_data(
                base_url,
                org_id=seed_org_id,
                user_id=seed_user_id,
                email=seed_email,
                concept_count=seed_concepts,
                backdate_minutes_step=seed_backdate_step,
                timeout_seconds=timeout_seconds,
            ),
        )

    input_csv = st.sidebar.text_input(
        "Default input_data CSV",
        value=build_uniform_vector_csv(cfg.network.input_size, value=0.1),
    )
    target_csv = st.sidebar.text_input(
        "Default target_data CSV",
        value=build_uniform_vector_csv(cfg.network.output_size, value=0.2),
    )

    tab_status, tab_learning, tab_tenant, tab_pilot = st.tabs(
        ["Status & Metrics", "Learning Controls", "Tenant Operations", "Pilot Evaluation"]
    )

    with tab_status:
        col_a, col_b = st.columns(2)
        if col_a.button("Refresh /status", use_container_width=True):
            _safe_action(st, "/status", lambda: get_status(base_url, timeout_seconds=timeout_seconds))
        if col_b.button("Refresh /metrics", use_container_width=True):
            _safe_action(st, "/metrics", lambda: get_metrics(base_url, timeout_seconds=timeout_seconds))

        try:
            status_snapshot = get_status(base_url, timeout_seconds=timeout_seconds)
            metrics_snapshot = get_metrics(base_url, timeout_seconds=timeout_seconds)
            memory_count = int(status_snapshot.get("memory_count", 0))
            at_risk = int(status_snapshot.get("at_risk_memories", 0))
            review_skipped = int(metrics_snapshot.get("counters", {}).get("review_skipped", 0))

            if memory_count <= 0:
                st.warning("No memories yet — run Teach, Record Attempt, or Seed Demo Data to initialize concepts.")
            if memory_count > 0 and at_risk <= 0:
                st.info("No items currently due — queue may populate after more attempts or backdated demo seeding.")
            if review_skipped > 0:
                st.info(f"Review skipped events detected ({review_skipped}) — indicates review actions occurred with no due items.")
        except Exception:
            pass

    with tab_learning:
        with st.form("teach_form"):
            st.subheader("Teach")
            teach_input_csv = st.text_input("input_data", value=input_csv)
            teach_target_csv = st.text_input("target_data", value=target_csv)
            teach_submit = st.form_submit_button("Send /teach")
        if teach_submit:
            _safe_action(
                st,
                "/teach",
                lambda: teach(
                    base_url,
                    input_data=parse_float_list(teach_input_csv, expected_len=cfg.network.input_size),
                    target_data=parse_float_list(teach_target_csv, expected_len=cfg.network.output_size),
                    timeout_seconds=timeout_seconds,
                ),
            )

        with st.form("ask_form"):
            st.subheader("Ask")
            ask_input_csv = st.text_input("ask input_data", value=input_csv)
            ask_memory_key = st.text_input("memory_key (optional)", value="")
            ask_submit = st.form_submit_button("Send /ask")
        if ask_submit:
            _safe_action(
                st,
                "/ask",
                lambda: ask(
                    base_url,
                    input_data=parse_float_list(ask_input_csv, expected_len=cfg.network.input_size),
                    memory_key=ask_memory_key or None,
                    timeout_seconds=timeout_seconds,
                ),
            )

        if st.button("Trigger /sleep", use_container_width=True):
            _safe_action(st, "/sleep", lambda: sleep(base_url, timeout_seconds=timeout_seconds))

    with tab_tenant:
        with st.form("org_form"):
            st.subheader("Create Organization")
            org_name = st.text_input("Organization name", value="Acme Learning")
            org_id = st.text_input("Organization ID (optional)", value="")
            org_submit = st.form_submit_button("Send /organizations")
        if org_submit:
            _safe_action(
                st,
                "/organizations",
                lambda: create_organization(base_url, name=org_name, org_id=org_id or None, timeout_seconds=timeout_seconds),
            )

        with st.form("user_form"):
            st.subheader("Create User")
            user_org_id = st.text_input("org_id", value="org_acme")
            user_email = st.text_input("email", value="learner@example.com")
            user_name = st.text_input("name (optional)", value="")
            user_id = st.text_input("user_id (optional)", value="")
            user_submit = st.form_submit_button("Send /users")
        if user_submit:
            _safe_action(
                st,
                "/users",
                lambda: create_user(
                    base_url,
                    org_id=user_org_id,
                    email=user_email,
                    user_id=user_id or None,
                    name=user_name or None,
                    timeout_seconds=timeout_seconds,
                ),
            )

        with st.form("attempt_form"):
            st.subheader("Record Attempt")
            attempt_user_id = st.text_input("attempt user_id", value="usr_001")
            attempt_concept_id = st.text_input("concept_id", value="concept_001")
            attempt_correct = st.checkbox("correct", value=True)
            attempt_response_ms = st.number_input("response_ms", min_value=0.0, value=850.0)
            attempt_submit = st.form_submit_button("Send /record-attempt")
        if attempt_submit:
            _safe_action(
                st,
                "/record-attempt",
                lambda: record_attempt(
                    base_url,
                    user_id=attempt_user_id,
                    concept_id=attempt_concept_id,
                    correct=attempt_correct,
                    response_ms=float(attempt_response_ms),
                    timeout_seconds=timeout_seconds,
                ),
            )

        st.subheader("Review Queue & Analytics")
        queue_col, analytics_col = st.columns(2)
        queue_user = queue_col.text_input("queue user_id", value="usr_001")
        queue_limit = int(queue_col.number_input("queue limit", min_value=1, max_value=200, value=20))
        if queue_col.button("Load /review-queue", use_container_width=True):
            _safe_action(
                st,
                "/review-queue",
                lambda: get_review_queue(base_url, user_id=queue_user, limit=queue_limit, timeout_seconds=timeout_seconds),
            )

        analytics_user = analytics_col.text_input("analytics user_id (optional)", value="usr_001")
        analytics_org = analytics_col.text_input("analytics org_id (optional)", value="")
        analytics_window = int(analytics_col.number_input("window_days", min_value=1, max_value=365, value=30))
        if analytics_col.button("Load /analytics", use_container_width=True):
            _safe_action(
                st,
                "/analytics",
                lambda: get_analytics(
                    base_url,
                    user_id=analytics_user or None,
                    org_id=analytics_org or None,
                    window_days=analytics_window,
                    timeout_seconds=timeout_seconds,
                ),
            )

        st.subheader("Memory Explorer")
        mem_col_a, mem_col_b, mem_col_c = st.columns(3)
        mem_offset = int(mem_col_a.number_input("memory offset", min_value=0, value=0))
        mem_limit = int(mem_col_b.number_input("memory limit", min_value=1, max_value=500, value=20))
        mem_stable_filter = mem_col_c.selectbox("stable filter", options=["all", "stable", "unstable"], index=0)
        if st.button("Load /memories", use_container_width=True):
            stable_value: Optional[bool]
            if mem_stable_filter == "stable":
                stable_value = True
            elif mem_stable_filter == "unstable":
                stable_value = False
            else:
                stable_value = None
            _safe_action(
                st,
                "/memories",
                lambda: list_memories(
                    base_url,
                    offset=mem_offset,
                    limit=mem_limit,
                    stable=stable_value,
                    timeout_seconds=timeout_seconds,
                ),
            )

    with tab_pilot:
        st.subheader("Pilot Setup (Controlled Cohorts)")
        setup_payload = {
            "org_id": st.text_input("setup org_id", value="org_acme"),
            "name": st.text_input("pilot name", value="Support Pilot - Week 1"),
            "treatment_ratio": float(st.number_input("treatment_ratio", min_value=0.01, max_value=0.99, value=0.50)),
            "random_seed": int(st.number_input("pilot random_seed", min_value=1, value=42)),
            "pilot_id": st.text_input("pilot_id (optional)", value="") or None,
        }
        if st.button("Send /pilot/setup", use_container_width=True):
            _safe_action(
                st,
                "/pilot/setup",
                lambda: setup_pilot(base_url, payload=setup_payload, timeout_seconds=timeout_seconds),
            )

        st.subheader("Pilot Baseline Capture")
        baseline_payload = {
            "pilot_id": st.text_input("baseline pilot_id", value="pilot_001"),
            "window_days": int(st.number_input("baseline window_days", min_value=1, max_value=365, value=30)),
        }
        captured_at_raw = st.text_input("baseline captured_at (optional epoch)", value="")
        if captured_at_raw.strip():
            baseline_payload["captured_at"] = float(captured_at_raw)
        if st.button("Send /pilot/baseline", use_container_width=True):
            _safe_action(
                st,
                "/pilot/baseline",
                lambda: capture_pilot_baseline(base_url, payload=baseline_payload, timeout_seconds=timeout_seconds),
            )

        st.subheader("Pilot Run Snapshot")
        pilot_run_id = st.text_input("run pilot_id", value="pilot_001")
        if st.button("Load /pilot/run", use_container_width=True):
            _safe_action(
                st,
                "/pilot/run",
                lambda: get_pilot_run(base_url, pilot_id=pilot_run_id, timeout_seconds=timeout_seconds),
            )

        st.subheader("Pilot Gate Evaluation")
        pilot_payload = {
            "org_id": st.text_input("pilot org_id (optional)", value="org_acme") or None,
            "sample_size": int(st.number_input("sample_size", min_value=1, value=300)),
            "onboarding_hours": float(st.number_input("onboarding_hours", min_value=0.0, value=25.0)),
            "retained_mastery_treatment": float(st.number_input("retained_mastery_treatment", min_value=0.0, max_value=1.0, value=0.80)),
            "retained_mastery_control": float(st.number_input("retained_mastery_control", min_value=0.0, max_value=1.0, value=0.70)),
            "forgetting_velocity_treatment": float(st.number_input("forgetting_velocity_treatment", min_value=0.0, max_value=1.0, value=0.20)),
            "forgetting_velocity_control": float(st.number_input("forgetting_velocity_control", min_value=0.0, max_value=1.0, value=0.35)),
            "review_efficiency_treatment": float(st.number_input("review_efficiency_treatment", min_value=0.0, max_value=1.0, value=0.74)),
            "review_efficiency_control": float(st.number_input("review_efficiency_control", min_value=0.0, max_value=1.0, value=0.62)),
        }
        if st.button("Send /pilot/evaluate", use_container_width=True):
            _safe_action(
                st,
                "/pilot/evaluate",
                lambda: evaluate_pilot(base_url, payload=pilot_payload, timeout_seconds=timeout_seconds),
            )

        st.subheader("Pilot History")
        hist_col_a, hist_col_b = st.columns(2)
        history_org_id = hist_col_a.text_input("history org_id (optional)", value="org_acme")
        history_limit = int(hist_col_b.number_input("history limit", min_value=1, max_value=500, value=20))
        if st.button("Load /pilot/history", use_container_width=True):
            _safe_action(
                st,
                "/pilot/history",
                lambda: pilot_history(
                    base_url,
                    org_id=history_org_id or None,
                    limit=history_limit,
                    timeout_seconds=timeout_seconds,
                ),
            )

    st.divider()
    st.caption(
        "Simulation suite CLI examples: "
        "`python -m src.simulation_suite all --env development --steps 60`, "
        "`python -m src.simulation_suite ab-policy --env testing --steps 40`"
    )


def main() -> None:
    import streamlit as st

    try:
        render_dashboard(st)
    except httpx.HTTPError as exc:
        st.error(f"HTTP request failed: {exc}")
    except Exception as exc:
        st.error(f"Dashboard error: {exc}")
        st.code(json.dumps({"error": str(exc)}, indent=2))


if __name__ == "__main__":
    main()

