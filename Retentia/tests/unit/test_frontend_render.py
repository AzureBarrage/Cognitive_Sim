from typing import Any, Dict, List

from src.config import SimulationConfig
from src import frontend_app


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

    def selectbox(self, label: str, options, index: int = 0, **kwargs):
        return self.parent.selectbox(label, options=options, index=index, **kwargs)

    def subheader(self, *args, **kwargs):
        return self.parent.subheader(*args, **kwargs)

    def metric(self, *args, **kwargs):
        return self.parent.metric(*args, **kwargs)


class FakeStreamlit:
    def __init__(
        self,
        button_values: Dict[str, bool],
        submit_values: Dict[str, bool],
        text_values: Dict[str, str] | None = None,
        number_values: Dict[str, Any] | None = None,
        checkbox_values: Dict[str, bool] | None = None,
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

    def set_page_config(self, **kwargs):
        return None

    def title(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def metric(self, *args, **kwargs):
        return None

    def subheader(self, *args, **kwargs):
        return None

    def divider(self):
        return None

    def tabs(self, labels):
        return [_Ctx() for _ in labels]

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

    def json(self, payload: Dict[str, Any]):
        self.json_payloads.append(payload)

    def write(self, *args, **kwargs):
        return None

    def markdown(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def dataframe(self, *args, **kwargs):
        return None

    def text_area(self, label: str, value: str = "", **kwargs):
        return self.text_values.get(label, value)


def test_render_dashboard_wires_buttons_and_forms(monkeypatch) -> None:
    calls: List[str] = []

    def _record(name: str):
        def inner(*args, **kwargs):
            calls.append(name)
            return {"ok": name}

        return inner

    monkeypatch.setattr(frontend_app, "get_status", _record("get_status"))
    monkeypatch.setattr(frontend_app, "get_metrics", _record("get_metrics"))
    monkeypatch.setattr(frontend_app, "teach", _record("teach"))
    monkeypatch.setattr(frontend_app, "ask", _record("ask"))
    monkeypatch.setattr(frontend_app, "sleep", _record("sleep"))
    monkeypatch.setattr(frontend_app, "create_organization", _record("create_organization"))
    monkeypatch.setattr(frontend_app, "create_user", _record("create_user"))
    monkeypatch.setattr(frontend_app, "upsert_concept", _record("upsert_concept"))
    monkeypatch.setattr(frontend_app, "list_concepts", _record("list_concepts"))
    monkeypatch.setattr(frontend_app, "record_attempt", _record("record_attempt"))
    monkeypatch.setattr(frontend_app, "get_review_queue", _record("get_review_queue"))
    monkeypatch.setattr(frontend_app, "get_analytics", _record("get_analytics"))
    monkeypatch.setattr(frontend_app, "list_memories", _record("list_memories"))
    monkeypatch.setattr(frontend_app, "setup_pilot", _record("setup_pilot"))
    monkeypatch.setattr(frontend_app, "get_pilot_run", _record("get_pilot_run"))
    monkeypatch.setattr(frontend_app, "capture_pilot_baseline", _record("capture_pilot_baseline"))
    monkeypatch.setattr(frontend_app, "evaluate_pilot", _record("evaluate_pilot"))
    monkeypatch.setattr(frontend_app, "pilot_history", _record("pilot_history"))
    monkeypatch.setattr(frontend_app, "seed_demo_data", _record("seed_demo_data"))

    fake_st = FakeStreamlit(
        button_values={
            "Refresh /status": True,
            "Refresh /metrics": True,
            "Trigger /sleep": True,
            "Load Compliance Topics": True,
            "Load At-Risk Knowledge": True,
            "Load Compliance Risk Dashboard": True,
            "Load /memories": True,
            "Send /pilot/setup": True,
            "Send /pilot/baseline": True,
            "Load /pilot/run": True,
            "Send /pilot/evaluate": True,
            "Load /pilot/history": True,
        },
        submit_values={
            "Send /teach": True,
            "Send /ask": True,
            "Create Organization": True,
            "Create Employee Learner": True,
            "Submit Knowledge Check": True,
            "Save Compliance Topic": True,
        },
        text_values={
            "seed org_id": "org_demo",
            "seed user_id": "usr_demo",
            "seed email": "demo@example.com",
        },
        number_values={
            "seed compliance topic count": 20,
            "seed backdate minutes / knowledge check": 2.0,
        },
    )

    fake_st.button_values["Seed Compliance Demo Data"] = True

    frontend_app.render_dashboard(fake_st, config=SimulationConfig())

    assert set(calls) == {
        "get_status",
        "get_metrics",
        "teach",
        "ask",
        "sleep",
        "create_organization",
        "create_user",
        "upsert_concept",
        "list_concepts",
        "record_attempt",
        "get_review_queue",
        "get_analytics",
        "list_memories",
        "setup_pilot",
        "get_pilot_run",
        "capture_pilot_baseline",
        "evaluate_pilot",
        "pilot_history",
        "seed_demo_data",
    }
    assert len(fake_st.success_messages) == 19
    assert len(fake_st.json_payloads) == 19

