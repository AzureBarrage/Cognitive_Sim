from src.tenant.store import TenantMemoryStore


def test_pilot_gate_go_decision_true(tmp_path) -> None:
    store = TenantMemoryStore(str(tmp_path / "pilot.db"))
    result = store.evaluate_pilot(
        retained_mastery_treatment=0.78,
        retained_mastery_control=0.68,
        forgetting_velocity_treatment=0.20,
        forgetting_velocity_control=0.40,
        review_efficiency_treatment=0.72,
        review_efficiency_control=0.60,
        sample_size=1000,
        min_retained_mastery_lift=0.05,
        min_forgetting_velocity_reduction=0.10,
        min_review_efficiency_lift=0.05,
        confidence_z_threshold=1.96,
        min_sample_size=100,
        onboarding_hours=20.0,
        max_onboarding_hours=40.0,
        org_id="org_gate",
    )
    assert result["go_decision"] is True
    assert result["reasons"] == []
    store.close()


def test_pilot_gate_go_decision_false_when_thresholds_not_met(tmp_path) -> None:
    store = TenantMemoryStore(str(tmp_path / "pilot.db"))
    result = store.evaluate_pilot(
        retained_mastery_treatment=0.60,
        retained_mastery_control=0.59,
        forgetting_velocity_treatment=0.40,
        forgetting_velocity_control=0.41,
        review_efficiency_treatment=0.52,
        review_efficiency_control=0.51,
        sample_size=30,
        min_retained_mastery_lift=0.05,
        min_forgetting_velocity_reduction=0.10,
        min_review_efficiency_lift=0.05,
        confidence_z_threshold=1.96,
        min_sample_size=100,
        onboarding_hours=80.0,
        max_onboarding_hours=40.0,
        org_id="org_gate_bad",
    )
    assert result["go_decision"] is False
    assert len(result["reasons"]) >= 1
    history = store.list_pilot_evaluations(org_id="org_gate_bad", limit=10)
    assert len(history) == 1
    store.close()
