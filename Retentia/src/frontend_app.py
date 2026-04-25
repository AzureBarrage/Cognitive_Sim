import json
import time
from typing import Any, Dict, List, Optional

import httpx

from src.config import SimulationConfig, load_config


COMPLIANCE_DEMO_TOPICS: List[Dict[str, Any]] = [
    {
        "concept_id": "hipaa_001",
        "title": "Protected Health Information (PHI)",
        "prompt": "What qualifies as Protected Health Information (PHI)?",
        "answer": "Any individually identifiable health information including medical history, diagnosis, or payment data.",
        "explanation": "PHI includes any data that can be linked to a specific patient.",
        "tags": ["hipaa", "privacy", "healthcare"],
        "difficulty": 1.2,
        "source": "compliance_demo_set",
    },
    {
        "concept_id": "hipaa_002",
        "title": "Minimum Necessary Standard",
        "prompt": "What does the minimum necessary standard require?",
        "answer": "Only the minimum amount of PHI needed should be accessed or disclosed.",
        "explanation": "This reduces exposure risk and limits unnecessary access.",
        "tags": ["hipaa", "access_control", "healthcare"],
        "difficulty": 1.4,
        "source": "compliance_demo_set",
    },
    {
        "concept_id": "sec_001",
        "title": "Phishing Identification",
        "prompt": "What is a key indicator of a phishing email?",
        "answer": "Unexpected requests for sensitive information or urgent actions.",
        "explanation": "Phishing attempts rely on urgency and deception.",
        "tags": ["security", "phishing", "cybersecurity"],
        "difficulty": 1.2,
        "source": "compliance_demo_set",
    },
    {
        "concept_id": "sec_002",
        "title": "Password Security",
        "prompt": "What defines a strong password?",
        "answer": "A long, unique password that is not reused across systems.",
        "explanation": "Weak or reused passwords are a major security vulnerability.",
        "tags": ["security", "authentication", "cybersecurity"],
        "difficulty": 1.1,
        "source": "compliance_demo_set",
    },
]


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


def risk_level(risk_score: float) -> str:
    score = float(risk_score)
    if score >= 0.70:
        return "🔴 High Risk"
    if score >= 0.40:
        return "🟡 Medium Risk"
    return "🟢 Low Risk"


def compliance_status(retention_percentage: float, high_risk_count: int = 0, medium_risk_count: int = 0) -> str:
    retention = float(retention_percentage)
    if high_risk_count > 0 or retention < 0.60:
        return "🔴 Critical"
    if medium_risk_count > 0 or retention < 0.80:
        return "⚠️ At Risk"
    return "✅ Stable"


def enrich_queue_for_compliance(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for item in items:
        copy = dict(item)
        risk_score = float(copy.get("risk_score", 0.0))
        copy["risk_level"] = risk_level(risk_score)
        copy["compliance_topic"] = (copy.get("concept") or {}).get("title") or copy.get("concept_id")
        copy["why_this_is_shown"] = copy.get("reason", "Based on retention modeling and prior knowledge checks.")
        enriched.append(copy)
    return enriched


def compliance_risk_distribution(items: List[Dict[str, Any]]) -> Dict[str, int]:
    distribution = {"high": 0, "medium": 0, "low": 0}
    for item in items:
        score = float(item.get("risk_score", 0.0))
        if score >= 0.70:
            distribution["high"] += 1
        elif score >= 0.40:
            distribution["medium"] += 1
        else:
            distribution["low"] += 1
    return distribution


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


def upsert_concept(
    base_url: str,
    org_id: str,
    concept_id: str,
    title: str,
    prompt: str,
    answer: str,
    explanation: Optional[str] = None,
    tags: Optional[List[str]] = None,
    difficulty: float = 1.0,
    source: Optional[str] = None,
    active: bool = True,
    timeout_seconds: float = 10.0,
    transport: Optional[httpx.BaseTransport] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "org_id": org_id,
        "concept_id": concept_id,
        "title": title,
        "prompt": prompt,
        "answer": answer,
        "tags": tags or [],
        "difficulty": float(difficulty),
        "active": bool(active),
    }
    if explanation:
        payload["explanation"] = explanation
    if source:
        payload["source"] = source
    return perform_request(base_url, "PUT", "/concepts", payload=payload, timeout_seconds=timeout_seconds, transport=transport)


def list_concepts(
    base_url: str,
    org_id: str,
    active: Optional[bool] = True,
    limit: int = 100,
    offset: int = 0,
    timeout_seconds: float = 10.0,
    transport: Optional[httpx.BaseTransport] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"org_id": org_id, "limit": int(limit), "offset": int(offset)}
    if active is not None:
        params["active"] = bool(active)
    return perform_request(base_url, "GET", "/concepts", params=params, timeout_seconds=timeout_seconds, transport=transport)


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


def get_audit_log(
    base_url: str,
    org_id: Optional[str] = None,
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = 100,
    timeout_seconds: float = 10.0,
    transport: Optional[httpx.BaseTransport] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"limit": int(limit)}
    if org_id:
        params["org_id"] = org_id
    if user_id:
        params["user_id"] = user_id
    if action:
        params["action"] = action
    return perform_request(base_url, "GET", "/audit-log", params=params, timeout_seconds=timeout_seconds, transport=transport)


def build_roi_report(
    base_url: str,
    org_id: str,
    learners: int,
    training_hours_saved_per_learner: float,
    cost_per_training_hour: float,
    annual_contract_value: float,
    window_days: int = 30,
    timeout_seconds: float = 10.0,
    transport: Optional[httpx.BaseTransport] = None,
) -> Dict[str, Any]:
    payload = {
        "org_id": org_id,
        "learners": int(learners),
        "training_hours_saved_per_learner": float(training_hours_saved_per_learner),
        "cost_per_training_hour": float(cost_per_training_hour),
        "annual_contract_value": float(annual_contract_value),
        "window_days": int(window_days),
    }
    return perform_request(base_url, "POST", "/roi/report", payload=payload, timeout_seconds=timeout_seconds, transport=transport)


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
    demo_topics = list(COMPLIANCE_DEMO_TOPICS)
    for idx in range(max(1, int(concept_count))):
        topic = demo_topics[idx % len(demo_topics)]
        cycle = idx // len(demo_topics)
        concept_id = str(topic["concept_id"] if cycle == 0 else f"{topic['concept_id']}_r{cycle + 1}")
        upsert_concept(
            base_url,
            org_id=org_id,
            concept_id=concept_id,
            title=str(topic["title"] if cycle == 0 else f"{topic['title']} Reinforcement {cycle + 1}"),
            prompt=str(topic["prompt"]),
            answer=str(topic["answer"]),
            explanation=str(topic["explanation"]),
            tags=list(topic["tags"]),
            difficulty=float(topic["difficulty"]),
            source=str(topic["source"]),
            timeout_seconds=timeout_seconds,
        )
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


def _render_compliance_queue(st: Any, items: List[Dict[str, Any]]) -> None:
    enriched = enrich_queue_for_compliance(items)
    if not enriched:
        st.info("No at-risk knowledge found yet. Seed demo data or record knowledge checks to populate risk.")
        return
    st.caption("Based on retention modeling and prior knowledge checks")
    for item in enriched:
        concept = item.get("concept") or {}
        st.markdown(f"**{item['risk_level']} — {item['compliance_topic']}**")
        st.write(item.get("why_this_is_shown"))
        if concept.get("prompt"):
            st.caption(f"Compliance check: {concept.get('prompt')}")
    if hasattr(st, "dataframe"):
        st.dataframe(
            [
                {
                    "Compliance Topic": item["compliance_topic"],
                    "Risk Level": item["risk_level"],
                    "Compliance Risk Score": item.get("risk_score"),
                    "Retention Confidence": item.get("retention"),
                    "Knowledge Stability": item.get("stability"),
                    "Why this is being shown": item.get("why_this_is_shown"),
                }
                for item in enriched
            ],
            use_container_width=True,
        )
    else:
        st.json(enriched)


def render_dashboard(st: Any, config: Optional[SimulationConfig] = None) -> None:
    st.set_page_config(page_title="Compliance Retention Engine", layout="wide")
    cfg = config or load_config("development")

    st.title("Compliance Retention Engine")
    st.caption("Prevent knowledge decay. Reduce compliance risk.")

    base_url = st.sidebar.text_input("API Base URL", value="http://localhost:8000")
    timeout_seconds = st.sidebar.number_input("Request timeout (seconds)", min_value=1.0, max_value=120.0, value=15.0)

    st.sidebar.subheader("Compliance Demo Seeder")
    seed_org_id = st.sidebar.text_input("seed org_id", value="org_demo")
    seed_user_id = st.sidebar.text_input("seed user_id", value="usr_demo")
    seed_email = st.sidebar.text_input("seed email", value="demo@example.com")
    seed_concepts = int(st.sidebar.number_input("seed compliance topic count", min_value=4, max_value=200, value=20))
    seed_backdate_step = float(st.sidebar.number_input("seed backdate minutes / knowledge check", min_value=0.0, max_value=60.0, value=2.0))
    if st.sidebar.button("Seed Compliance Demo Data", use_container_width=True):
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

    tab_setup, tab_topics, tab_daily, tab_risk, tab_dashboard, tab_roi = st.tabs(
        [
            "Setup (Org + Users)",
            "Compliance Topics",
            "Daily Compliance Check",
            "At-Risk Knowledge",
            "Compliance Risk Dashboard",
            "ROI & Audit",
        ]
    )

    with tab_setup:
        st.subheader("Create a training environment for your organization")
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
                st.warning("No compliance topics have knowledge checks yet — seed demo data or record a knowledge check.")
            if memory_count > 0 and at_risk <= 0:
                st.info("No topics are currently due — at-risk knowledge may populate after more knowledge checks or backdated demo seeding.")
            if review_skipped > 0:
                st.info(f"Review skipped events detected ({review_skipped}) — indicates checks occurred with no due at-risk knowledge.")
        except Exception:
            pass

        st.subheader("Organization Setup")
        with st.form("org_form"):
            org_name = st.text_input("Organization name", value="Acme Compliance")
            org_id = st.text_input("Organization ID (optional)", value="")
            org_submit = st.form_submit_button("Create Organization")
        if org_submit:
            _safe_action(
                st,
                "/organizations",
                lambda: create_organization(base_url, name=org_name, org_id=org_id or None, timeout_seconds=timeout_seconds),
            )

        with st.form("user_form"):
            st.subheader("Employee Learner")
            user_org_id = st.text_input("org_id", value="org_acme")
            user_email = st.text_input("employee email", value="learner@example.com")
            user_name = st.text_input("employee name (optional)", value="")
            user_id = st.text_input("employee user_id (optional)", value="")
            user_submit = st.form_submit_button("Create Employee Learner")
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

    with tab_daily:
        st.subheader("Reinforce knowledge continuously based on retention risk")
        with st.form("attempt_form"):
            st.subheader("Knowledge Check")
            attempt_user_id = st.text_input("employee user_id", value="usr_001")
            attempt_concept_id = st.text_input("compliance topic id", value="hipaa_001")
            attempt_correct = st.checkbox("answered correctly", value=True)
            attempt_response_ms = st.number_input("response_ms", min_value=0.0, value=850.0)
            attempt_submit = st.form_submit_button("Submit Knowledge Check")
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

        st.caption("Developer controls for the underlying cognitive engine")
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

    with tab_topics:
        st.subheader("Define the knowledge employees must retain")
        with st.form("concept_form"):
            concept_org_id = st.text_input("topic org_id", value="org_acme")
            concept_id = st.text_input("compliance topic id", value="hipaa_001")
            concept_title = st.text_input("title", value="Protected Health Information (PHI)")
            concept_prompt = st.text_area("knowledge check prompt", value="What qualifies as Protected Health Information (PHI)?")
            concept_answer = st.text_area("expected answer", value="Any individually identifiable health information including medical history, diagnosis, or payment data.")
            concept_explanation = st.text_area("explanation", value="PHI includes any data that can be linked to a specific patient.")
            concept_tags_raw = st.text_input("tags CSV", value="hipaa,privacy,healthcare")
            concept_difficulty = float(st.number_input("difficulty", min_value=0.0, max_value=10.0, value=1.2))
            concept_source = st.text_input("source", value="compliance_demo_set")
            concept_active = st.checkbox("active", value=True)
            concept_submit = st.form_submit_button("Save Compliance Topic")
        if concept_submit:
            _safe_action(
                st,
                "/concepts",
                lambda: upsert_concept(
                    base_url,
                    org_id=concept_org_id,
                    concept_id=concept_id,
                    title=concept_title,
                    prompt=concept_prompt,
                    answer=concept_answer,
                    explanation=concept_explanation or None,
                    tags=[item.strip() for item in concept_tags_raw.split(",") if item.strip()],
                    difficulty=concept_difficulty,
                    source=concept_source or None,
                    active=concept_active,
                    timeout_seconds=timeout_seconds,
                ),
            )

        list_col_a, list_col_b = st.columns(2)
        list_concept_org = list_col_a.text_input("list topics org_id", value="org_acme")
        list_concept_limit = int(list_col_b.number_input("topic limit", min_value=1, max_value=500, value=50))
        if st.button("Load Compliance Topics", use_container_width=True):
            _safe_action(
                st,
                "/concepts list",
                lambda: list_concepts(base_url, org_id=list_concept_org, limit=list_concept_limit, timeout_seconds=timeout_seconds),
            )

    with tab_risk:
        st.subheader("Identify what employees are most likely to forget")
        risk_user = st.text_input("employee user_id for at-risk knowledge", value="usr_001")
        risk_limit = int(st.number_input("at-risk knowledge limit", min_value=1, max_value=200, value=20))
        if st.button("Load At-Risk Knowledge", use_container_width=True):
            try:
                queue_result = get_review_queue(base_url, user_id=risk_user, limit=risk_limit, timeout_seconds=timeout_seconds)
                _show_result(st, "/review-queue", queue_result)
                _render_compliance_queue(st, queue_result.get("items", []))
            except Exception as exc:
                st.error(f"/review-queue failed: {exc}")

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

    with tab_dashboard:
        st.subheader("Monitor retention and detect compliance risk early")
        analytics_user = st.text_input("dashboard employee user_id (optional)", value="usr_001")
        analytics_org = st.text_input("dashboard org_id (optional)", value="")
        analytics_window = int(st.number_input("dashboard window_days", min_value=1, max_value=365, value=30))
        if st.button("Load Compliance Risk Dashboard", use_container_width=True):
            try:
                analytics_result = get_analytics(
                    base_url,
                    user_id=analytics_user or None,
                    org_id=analytics_org or None,
                    window_days=analytics_window,
                    timeout_seconds=timeout_seconds,
                )
                data = analytics_result.get("data", {})
                retention = float(data.get("retention_percentage", 0.0))
                at_risk_concepts = int(data.get("at_risk_concepts", 0))
                status = compliance_status(retention, high_risk_count=at_risk_concepts)
                col_status, col_retention, col_at_risk = st.columns(3)
                col_status.metric("Compliance Status", status)
                col_retention.metric("Retention Confidence", f"{retention:.0%}")
                col_at_risk.metric("At-Risk Topics", at_risk_concepts)
                _show_result(st, "/analytics", analytics_result)
            except Exception as exc:
                st.error(f"/analytics failed: {exc}")

        st.subheader("Pilot Setup (Controlled Cohorts)")
        setup_payload = {
            "org_id": st.text_input("setup org_id", value="org_acme"),
            "name": st.text_input("pilot name", value="Compliance Retention Pilot - Week 1"),
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

    with tab_roi:
        st.subheader("Quantify impact and maintain audit readiness")
        st.caption("Based on reduced retraining and improved retention outcomes")
        roi_org_id = st.text_input("ROI org_id", value="org_acme")
        roi_col_a, roi_col_b, roi_col_c, roi_col_d = st.columns(4)
        roi_learners = int(roi_col_a.number_input("learners", min_value=0, value=10000))
        roi_hours = float(roi_col_b.number_input("hours saved / learner", min_value=0.0, value=2.5))
        roi_hourly_cost = float(roi_col_c.number_input("cost / training hour", min_value=0.0, value=60.0))
        roi_acv = float(roi_col_d.number_input("annual contract value", min_value=0.0, value=300000.0))
        roi_window = int(st.number_input("ROI analytics window_days", min_value=1, max_value=365, value=30))
        if st.button("Build /roi/report", use_container_width=True):
            _safe_action(
                st,
                "/roi/report",
                lambda: build_roi_report(
                    base_url,
                    org_id=roi_org_id,
                    learners=roi_learners,
                    training_hours_saved_per_learner=roi_hours,
                    cost_per_training_hour=roi_hourly_cost,
                    annual_contract_value=roi_acv,
                    window_days=roi_window,
                    timeout_seconds=timeout_seconds,
                ),
            )

        st.subheader("Audit Log")
        audit_col_a, audit_col_b, audit_col_c = st.columns(3)
        audit_org = audit_col_a.text_input("audit org_id (optional)", value="org_acme")
        audit_user = audit_col_b.text_input("audit user_id (optional)", value="")
        audit_action = audit_col_c.text_input("audit action (optional)", value="")
        audit_limit = int(st.number_input("audit limit", min_value=1, max_value=1000, value=100))
        if st.button("Load /audit-log", use_container_width=True):
            _safe_action(
                st,
                "/audit-log",
                lambda: get_audit_log(
                    base_url,
                    org_id=audit_org or None,
                    user_id=audit_user or None,
                    action=audit_action or None,
                    limit=audit_limit,
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

