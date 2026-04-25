import json
import time
from typing import Any, Dict, Optional

import httpx

from src.frontend_app import create_organization, create_user, perform_request


def daily_session(
    base_url: str,
    user_id: str,
    limit: int = 20,
    window_days: Optional[int] = None,
    now: Optional[float] = None,
    attempt: Optional[Dict[str, Any]] = None,
    timeout_seconds: float = 10.0,
    transport: Optional[httpx.BaseTransport] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "user_id": user_id,
        "limit": int(limit),
    }
    if window_days is not None:
        payload["window_days"] = int(window_days)
    if now is not None:
        payload["now"] = float(now)
    if attempt is not None:
        payload["attempt"] = attempt
    return perform_request(
        base_url,
        "POST",
        "/daily-session",
        payload=payload,
        timeout_seconds=timeout_seconds,
        transport=transport,
    )


def _extract_http_detail(exc: httpx.HTTPStatusError) -> str:
    response = exc.response
    if response is None:
        return str(exc)
    try:
        payload = response.json()
        if isinstance(payload, dict):
            detail = payload.get("detail")
            if detail is not None:
                return str(detail)
    except Exception:
        pass
    text = str(response.text or "").strip()
    return text or str(exc)


def bootstrap_learner(
    base_url: str,
    org_id: str,
    org_name: str,
    user_id: str,
    email: str,
    user_name: Optional[str] = None,
    timeout_seconds: float = 10.0,
    transport: Optional[httpx.BaseTransport] = None,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}

    try:
        org = create_organization(
            base_url,
            name=org_name,
            org_id=org_id,
            timeout_seconds=timeout_seconds,
            transport=transport,
        )
        output["organization"] = {"status": "created", **org}
    except httpx.HTTPStatusError as exc:
        detail = _extract_http_detail(exc)
        if detail == "organization_already_exists":
            output["organization"] = {
                "status": "existing",
                "org_id": org_id,
                "name": org_name,
                "detail": detail,
            }
        else:
            raise

    try:
        user = create_user(
            base_url,
            org_id=org_id,
            email=email,
            user_id=user_id,
            name=user_name,
            timeout_seconds=timeout_seconds,
            transport=transport,
        )
        output["user"] = {"status": "created", **user}
    except httpx.HTTPStatusError as exc:
        detail = _extract_http_detail(exc)
        if detail in {"user_already_exists", "user_email_already_exists"}:
            output["user"] = {
                "status": "existing",
                "org_id": org_id,
                "user_id": user_id,
                "email": email,
                "name": user_name,
                "detail": detail,
            }
        else:
            raise

    return output


def _show_result(st: Any, label: str, result: Dict[str, Any]) -> None:
    st.success(f"{label} succeeded")
    st.json(result)


def _render_session_snapshot(st: Any, session: Dict[str, Any]) -> None:
    st.subheader("Daily Compliance Check Snapshot")
    col_a, col_b, col_c = st.columns(3)

    due_count = int(session.get("due_count", 0))
    total_count = int(session.get("total_count", 0))
    recommended_action = str(session.get("recommended_action", "unknown"))

    if hasattr(col_a, "metric"):
        col_a.metric("At-risk now", due_count)
        col_b.metric("Knowledge items", total_count)
        col_c.metric("Action", recommended_action)
    else:
        st.json(
            {
                "due_count": due_count,
                "total_count": total_count,
                "recommended_action": recommended_action,
            }
        )

    st.subheader("Next compliance topic at risk")
    next_item = session.get("next_item")
    if next_item:
        concept = next_item.get("concept") or {}
        if concept:
            st.markdown(f"**{concept.get('title', next_item.get('concept_id'))}**")
            st.write(concept.get("prompt", ""))
            with st.expander("Answer and explanation"):
                st.write(concept.get("answer", ""))
                if concept.get("explanation"):
                    st.caption(concept.get("explanation"))
        st.info(str(next_item.get("reason", "Based on retention modeling and prior knowledge checks.")))
        st.json(next_item)
    else:
        st.info("No compliance topics available yet. Record a knowledge check to start the compliance retention loop.")

    st.subheader("At-Risk Knowledge")
    queue = session.get("queue") or []
    if queue:
        st.caption("Based on retention modeling and prior knowledge checks")
        if hasattr(st, "dataframe"):
            st.dataframe(queue, use_container_width=True)
        else:
            st.json(queue)
    else:
        st.info("At-risk knowledge is currently empty.")

    st.subheader("Compliance Risk Dashboard")
    st.json(session.get("analytics", {}))


def render_daily_recall_coach(st: Any) -> None:
    st.set_page_config(page_title="Daily Compliance Check", layout="wide")
    st.title("Daily Compliance Check")
    st.caption("Reinforce knowledge continuously based on retention risk.")

    base_url = st.sidebar.text_input("API Base URL", value="http://localhost:8000")
    timeout_seconds = st.sidebar.number_input("Request timeout (seconds)", min_value=1.0, max_value=120.0, value=15.0)

    st.sidebar.subheader("Compliance Learner Bootstrap")
    bootstrap_org_id = st.sidebar.text_input("org_id", value="org_daily_demo")
    bootstrap_org_name = st.sidebar.text_input("organization name", value="Daily Recall Demo Org")
    bootstrap_user_id = st.sidebar.text_input("user_id", value="usr_daily_demo")
    bootstrap_email = st.sidebar.text_input("user email", value="daily.demo@example.com")
    bootstrap_user_name = st.sidebar.text_input("user name (optional)", value="")
    if st.sidebar.button("Bootstrap Learner", use_container_width=True):
        try:
            result = bootstrap_learner(
                base_url,
                org_id=bootstrap_org_id,
                org_name=bootstrap_org_name,
                user_id=bootstrap_user_id,
                email=bootstrap_email,
                user_name=bootstrap_user_name or None,
                timeout_seconds=timeout_seconds,
            )
            _show_result(st, "bootstrap-learner", result)
        except Exception as exc:
            st.error(f"bootstrap-learner failed: {exc}")

    st.subheader("Daily Compliance Check Controls")
    control_col_a, control_col_b, control_col_c = st.columns(3)
    session_user_id = control_col_a.text_input("session user_id", value=bootstrap_user_id)
    session_limit = int(control_col_b.number_input("session queue limit", min_value=1, max_value=200, value=20))
    session_window_days = int(control_col_c.number_input("analytics window_days", min_value=1, max_value=365, value=30))

    if st.button("Load Daily Compliance Check", use_container_width=True):
        try:
            snapshot = daily_session(
                base_url,
                user_id=session_user_id,
                limit=session_limit,
                window_days=session_window_days,
                timeout_seconds=timeout_seconds,
            )
            _show_result(st, "daily-session", snapshot)
            _render_session_snapshot(st, snapshot)
        except Exception as exc:
            st.error(f"daily-session failed: {exc}")

    with st.form("daily_attempt_form"):
        st.subheader("Submit Knowledge Check + Refresh Compliance Risk")
        concept_id = st.text_input("compliance topic id", value="hipaa_001")
        correct = st.checkbox("answered correctly", value=True)
        response_ms = float(st.number_input("response_ms", min_value=0.0, value=850.0))
        attempted_at_mode = st.selectbox("attempted_at", options=["now", "custom epoch"], index=0)
        attempted_at_custom = st.text_input("attempted_at custom epoch (optional)", value="")
        submit_attempt = st.form_submit_button("Submit Attempt")

    if submit_attempt:
        attempt_payload: Dict[str, Any] = {
            "concept_id": concept_id,
            "correct": bool(correct),
            "response_ms": float(response_ms),
        }
        now_epoch = time.time()
        if attempted_at_mode == "custom epoch" and attempted_at_custom.strip():
            attempt_payload["attempted_at"] = float(attempted_at_custom)
        else:
            attempt_payload["attempted_at"] = float(now_epoch)

        try:
            snapshot = daily_session(
                base_url,
                user_id=session_user_id,
                limit=session_limit,
                window_days=session_window_days,
                now=now_epoch,
                attempt=attempt_payload,
                timeout_seconds=timeout_seconds,
            )
            _show_result(st, "submit-attempt", snapshot)
            _render_session_snapshot(st, snapshot)
        except Exception as exc:
            st.error(f"submit-attempt failed: {exc}")

    st.divider()
    st.caption(
        "API flow: `POST /daily-session` (optionally with a knowledge check) returns at-risk knowledge, compliance analytics, and next-step recommendation."
    )


def main() -> None:
    import streamlit as st

    try:
        render_daily_recall_coach(st)
    except httpx.HTTPError as exc:
        st.error(f"HTTP request failed: {exc}")
    except Exception as exc:
        st.error(f"Daily Recall Coach error: {exc}")
        st.code(json.dumps({"error": str(exc)}, indent=2))


if __name__ == "__main__":
    main()

