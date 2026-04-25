from typing import Any, Dict, List, Optional

from src import daily_recall_coach_app


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _Container:
    def __init__(self, parent: "FakeStreamlit"):
        self.parent = parent

    def button(self, label: str, **kwargs):
        return self.parent.button(label, **kwargs)

    def text_input(self, label: str, value: str = "", **kwargs):
        return self.parent.text_input(label, value=value, **kwargs)

    def number_input(self, label: str, value: Any = 0, **kwargs):
        return self.parent.number_input(label, value=value, **kwargs)

    def checkbox(self, label: str, value: bool = False, **kwargs):
        return self.parent.checkbox(label, value=value, **kwargs)

    def selectbox(self, label: str, options, index: int = 0, **kwargs):
        return self.parent.selectbox(label, options=options, index=index, **kwargs)

    def metric(self, label: str, value: Any, delta: Any = None, **kwargs):
        return self.parent.metric(label, value, delta=delta, **kwargs)

    def subheader(self, *args, **kwargs):
        return self.parent.subheader(*args, **kwargs)


class FakeStreamlit:
    def __init__(
        self,
        button_values: Dict[str, bool],
        submit_values: Dict[str, bool],
        text_values: Optional[Dict[str, str]] = None,
        number_values: Optional[Dict[str, Any]] = None,
        checkbox_values: Optional[Dict[str, bool]] = None,
    ):
        self.button_values = dict(button_values)
        self.submit_values = dict(submit_values)
        self.text_values = dict(text_values or {})
        self.number_values = dict(number_values or {})
        self.checkbox_values = dict(checkbox_values or {})
        self.sidebar = _Container(self)
        self.success_messages: List[str] = []
        self.error_messages: List[str] = []
        self.json_payloads: List[Dict[str, Any]] = []
        self.metric_payloads: List[Dict[str, Any]] = []
        self.dataframe_calls: int = 0

    def set_page_config(self, **kwargs):
        return None

    def title(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def subheader(self, *args, **kwargs):
        return None

    def divider(self):
        return None

    def columns(self, count: int):
        return [_Container(self) for _ in range(int(count))]

    def form(self, name: str):
        return _Ctx()

    def text_input(self, label: str, value: str = "", **kwargs):
        return self.text_values.get(label, value)

    def number_input(self, label: str, value: Any = 0, **kwargs):
        return self.number_values.get(label, value)

    def checkbox(self, label: str, value: bool = False, **kwargs):
        return self.checkbox_values.get(label, value)

    def selectbox(self, label: str, options, index: int = 0, **kwargs):
        if label in self.text_values:
            return self.text_values[label]
        return options[index]

    def button(self, label: str, **kwargs):
        return bool(self.button_values.get(label, False))

    def form_submit_button(self, label: str, **kwargs):
        return bool(self.submit_values.get(label, False))

    def success(self, message: str):
        self.success_messages.append(str(message))

    def error(self, message: str):
        self.error_messages.append(str(message))

    def info(self, message: str):
        return None

    def json(self, payload: Dict[str, Any]):
        self.json_payloads.append(payload)

    def metric(self, label: str, value: Any, delta: Any = None, **kwargs):
        self.metric_payloads.append({"label": label, "value": value, "delta": delta})

    def dataframe(self, data, **kwargs):
        self.dataframe_calls += 1


def test_render_daily_recall_coach_wires_actions(monkeypatch) -> None:
    calls: List[str] = []

    def _record(name: str):
        def inner(*args, **kwargs):
            calls.append(name)
            if name == "bootstrap_learner":
                return {
                    "organization": {"status": "created", "org_id": "org_daily_demo"},
                    "user": {"status": "created", "user_id": "usr_daily_demo"},
                }
            return {
                "user_id": "usr_daily_demo",
                "queue": [
                    {
                        "concept_id": "concept_001",
                        "next_review_at": 1700000000.0,
                        "is_due": True,
                        "retention": 0.42,
                        "risk_score": 0.68,
                        "strength": 0.8,
                        "stability": 1.2,
                    }
                ],
                "due_count": 1,
                "total_count": 1,
                "next_item": {"concept_id": "concept_001", "risk_score": 0.68},
                "analytics": {"attempts": 3},
                "recorded_attempt": {"concept_id": "concept_001", "correct": True},
                "recommended_action": "review_due_items",
            }

        return inner

    monkeypatch.setattr(daily_recall_coach_app, "bootstrap_learner", _record("bootstrap_learner"))
    monkeypatch.setattr(daily_recall_coach_app, "daily_session", _record("daily_session"))

    fake_st = FakeStreamlit(
        button_values={
            "Bootstrap Learner": True,
            "Load Daily Compliance Check": True,
        },
        submit_values={
            "Submit Attempt": True,
        },
        text_values={
            "org_id": "org_daily_demo",
            "organization name": "Daily Recall Demo Org",
            "user_id": "usr_daily_demo",
            "user email": "daily.demo@example.com",
            "user name (optional)": "",
            "session user_id": "usr_daily_demo",
            "compliance topic id": "concept_001",
            "attempted_at": "now",
            "attempted_at custom epoch (optional)": "",
        },
        number_values={
            "Request timeout (seconds)": 15.0,
            "session queue limit": 20,
            "analytics window_days": 30,
            "response_ms": 850.0,
        },
        checkbox_values={
            "answered correctly": True,
        },
    )

    daily_recall_coach_app.render_daily_recall_coach(fake_st)

    assert calls.count("bootstrap_learner") == 1
    assert calls.count("daily_session") == 2
    assert len(fake_st.success_messages) >= 3
    assert len(fake_st.metric_payloads) >= 3
    assert fake_st.dataframe_calls >= 1

