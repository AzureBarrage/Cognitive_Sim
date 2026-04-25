from contextlib import asynccontextmanager
import csv
import io
import json
import os
from threading import Lock
import time
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from src.config import load_config
from src.core.agent import CognitiveAgent
from src.core.memory_layer import MemoryLayer
from src.core.network import CognitiveNetwork
from src.core.optimizer import CognitiveOptimizer
from src.environment.dataset_manager import DatasetManager
from src.environment.simulation_env import SimulationEnvironment
from src.tenant.store import TenantMemoryStore
from src.utils.metrics import MetricsTracker
from src.utils.seed import set_global_seed


class TeachRequest(BaseModel):
    input_data: List[float] = Field(..., min_length=1)
    target_data: List[float] = Field(..., min_length=1)
    label: Optional[str] = None


class TeachResponse(BaseModel):
    status: str
    memory_id: Optional[str]
    loss: float
    energy_remaining: float


class PredictionRequest(BaseModel):
    input_data: List[float] = Field(..., min_length=1)
    memory_key: Optional[str] = None


class PredictionResponse(BaseModel):
    prediction: List[float]
    uncertainty: float
    recall_status: str
    energy: float


class AgentStatus(BaseModel):
    energy: float
    total_rewards: float
    memory_count: int
    at_risk_memories: int
    review_success_rate: float
    learn_success_rate: float
    current_lr: float
    uptime_seconds: float


class MemoryListResponse(BaseModel):
    items: List[Dict[str, Any]]
    count: int
    offset: int
    limit: int


class MetricsResponse(BaseModel):
    trends: Dict[str, float]
    counters: Dict[str, int]


class CreateOrganizationRequest(BaseModel):
    name: str = Field(..., min_length=1)
    org_id: Optional[str] = None


class CreateOrganizationResponse(BaseModel):
    org_id: str
    name: str
    created_at: float


class CreateUserRequest(BaseModel):
    org_id: str
    email: str
    name: Optional[str] = None
    user_id: Optional[str] = None


class CreateUserResponse(BaseModel):
    user_id: str
    org_id: str
    email: str
    name: Optional[str]
    created_at: float


class ConceptRequest(BaseModel):
    org_id: str
    concept_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    explanation: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    difficulty: float = Field(1.0, ge=0.0)
    source: Optional[str] = None
    active: bool = True


class ConceptResponse(BaseModel):
    concept_id: str
    org_id: str
    title: str
    prompt: str
    answer: str
    explanation: Optional[str]
    tags: List[str]
    difficulty: float
    source: Optional[str]
    version: int
    active: bool
    created_at: float
    updated_at: float


class ConceptListResponse(BaseModel):
    items: List[Dict[str, Any]]
    count: int
    offset: int
    limit: int


class RecordAttemptRequest(BaseModel):
    user_id: str
    concept_id: str
    correct: bool
    response_ms: Optional[float] = None
    attempted_at: Optional[float] = None


class DailySessionAttempt(BaseModel):
    concept_id: str
    correct: bool
    response_ms: Optional[float] = Field(None, ge=0.0)
    attempted_at: Optional[float] = None


class DailySessionRequest(BaseModel):
    user_id: str
    limit: int = Field(20, ge=1, le=200)
    window_days: Optional[int] = Field(None, ge=1)
    now: Optional[float] = None
    attempt: Optional[DailySessionAttempt] = None


class DailySessionResponse(BaseModel):
    user_id: str
    queue: List[Dict[str, Any]]
    due_count: int
    total_count: int
    next_item: Optional[Dict[str, Any]]
    analytics: Dict[str, Any]
    recorded_attempt: Optional[Dict[str, Any]] = None
    recommended_action: str


class ReviewQueueResponse(BaseModel):
    user_id: str
    items: List[Dict[str, Any]]


class AnalyticsResponse(BaseModel):
    scope: str
    data: Dict[str, Any]


class AuditLogResponse(BaseModel):
    items: List[Dict[str, Any]]
    count: int


class RoiReportRequest(BaseModel):
    org_id: str
    learners: int = Field(..., ge=0)
    training_hours_saved_per_learner: float = Field(..., ge=0.0)
    cost_per_training_hour: float = Field(..., ge=0.0)
    annual_contract_value: float = Field(..., ge=0.0)
    window_days: int = Field(30, ge=1)


class RoiReportResponse(BaseModel):
    org_id: str
    generated_at: float
    window_days: int
    inputs: Dict[str, Any]
    analytics: Dict[str, Any]
    gross_savings: float
    net_savings: float
    roi_multiple: float
    payback_months: float
    executive_summary: str


class PilotSetupRequest(BaseModel):
    org_id: str
    name: str = Field(..., min_length=1)
    treatment_ratio: float = Field(0.5, gt=0.0, lt=1.0)
    random_seed: int = 42
    pilot_id: Optional[str] = None


class PilotSetupResponse(BaseModel):
    pilot_id: str
    org_id: str
    name: str
    status: str
    treatment_ratio: float
    random_seed: int
    control_count: int
    treatment_count: int
    control_user_ids: List[str]
    treatment_user_ids: List[str]
    created_at: float


class PilotRunResponse(BaseModel):
    pilot_id: str
    org_id: str
    name: str
    status: str
    treatment_ratio: float
    random_seed: int
    created_at: float
    control_count: int
    treatment_count: int
    control_user_ids: List[str]
    treatment_user_ids: List[str]
    latest_baseline: Optional[Dict[str, Any]] = None


class PilotBaselineRequest(BaseModel):
    pilot_id: str
    window_days: int = Field(30, ge=1)
    captured_at: Optional[float] = None


class PilotBaselineResponse(BaseModel):
    baseline_id: int
    pilot_id: str
    captured_at: float
    window_days: int
    control_metrics: Dict[str, Any]
    treatment_metrics: Dict[str, Any]
    overall_metrics: Dict[str, Any]


class PilotEvaluateRequest(BaseModel):
    org_id: Optional[str] = None
    sample_size: int = Field(..., ge=1)
    onboarding_hours: float = Field(..., ge=0.0)
    retained_mastery_treatment: float
    retained_mastery_control: float
    forgetting_velocity_treatment: float
    forgetting_velocity_control: float
    review_efficiency_treatment: float
    review_efficiency_control: float


class PilotEvaluateResponse(BaseModel):
    org_id: Optional[str]
    evaluated_at: float
    sample_size: int
    retained_mastery_lift: float
    forgetting_velocity_reduction: float
    review_efficiency_lift: float
    retained_mastery_z: float
    forgetting_velocity_z: float
    review_efficiency_z: float
    onboarding_hours: float
    go_decision: bool
    reasons: List[str]


class PilotHistoryResponse(BaseModel):
    items: List[Dict[str, Any]]
    count: int


STATE: Dict[str, Any] = {}
STATE_LOCK = Lock()


def _ensure_state() -> Dict[str, Any]:
    if "agent" not in STATE:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Agent is not initialized")
    return STATE


def _require_api_key(request: Request, state: Dict[str, Any]) -> None:
    runtime = state["config"].runtime
    if not bool(runtime.require_api_key):
        return
    configured = str(runtime.api_key or "").strip()
    if not configured:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="API key is not configured")
    provided = request.headers.get("x-api-key") or request.headers.get("authorization") or ""
    if provided.lower().startswith("bearer "):
        provided = provided[7:]
    if provided.strip() != configured:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def _rows_to_csv(rows: List[Dict[str, Any]], fieldnames: List[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        flat: Dict[str, Any] = {}
        for field in fieldnames:
            value = row.get(field)
            if isinstance(value, (dict, list)):
                flat[field] = json_dumps(value)
            else:
                flat[field] = value
        writer.writerow(flat)
    return buffer.getvalue()


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    env_name = str(os.getenv("ENV", "development")).strip() or "development"
    config = load_config(env_name)
    set_global_seed(config.seed)

    memory = MemoryLayer(config.memory)
    memory.load_state()

    device = CognitiveNetwork.resolve_device(config.network.device)
    network = CognitiveNetwork(config.network).to(device)
    optimizer = CognitiveOptimizer(network.parameters(), base_lr=config.network.learning_rate, weight_decay=config.network.weight_decay)

    checkpoint = network.load_checkpoint(config.runtime.checkpoint_path)
    if checkpoint and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    agent = CognitiveAgent(memory, network, optimizer, config=config.agent)
    data_manager = DatasetManager(config.model_dump())
    data_manager.load_data(seed=config.seed)
    env = SimulationEnvironment(data_manager.get_train_loader(batch_size=config.data.batch_size))
    metrics = MetricsTracker(
        artifact_dir=config.runtime.artifact_dir,
        flush_every=config.runtime.metrics_flush_every,
        flush_interval_seconds=config.runtime.metrics_flush_interval_seconds,
    )
    tenant_store = TenantMemoryStore(config.runtime.tenant_db_path)

    with STATE_LOCK:
        STATE.clear()
        STATE.update(
            {
                "config": config,
                "agent": agent,
                "metrics": metrics,
                "epoch_start": time.time(),
                "device": device,
                "env": env,
                "tenant_store": tenant_store,
            }
        )

    yield

    with STATE_LOCK:
        if "agent" in STATE:
            STATE["agent"].memory.save_state()
            STATE["agent"].network.save_checkpoint(
                STATE["config"].runtime.checkpoint_path,
                optimizer_state=STATE["agent"].optimizer.state_dict(),
            )
        if "metrics" in STATE:
            STATE["metrics"].flush()
        if "tenant_store" in STATE:
            STATE["tenant_store"].close()


app = FastAPI(title="Cognitive Sim API", lifespan=lifespan)


@app.get("/")
def root(request: Request) -> Dict[str, str]:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
    return {"status": "online", "message": "Cognitive Agent Ready"}


@app.get("/status", response_model=AgentStatus)
def get_status(request: Request) -> AgentStatus:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        agent = state["agent"]
        total_reviews = agent.review_successes + agent.review_failures
        total_learns = agent.learn_successes + agent.learn_failures
        uptime = time.time() - float(state["epoch_start"])

        return AgentStatus(
            energy=float(agent.energy),
            total_rewards=float(agent.total_rewards),
            memory_count=int(len(agent.memory.memories)),
            at_risk_memories=int(agent.memory.get_due_review_count()),
            review_success_rate=float((agent.review_successes / total_reviews) if total_reviews else 0.0),
            learn_success_rate=float((agent.learn_successes / total_learns) if total_learns else 0.0),
            current_lr=float(agent.optimizer.get_current_lr()),
            uptime_seconds=float(uptime),
        )


@app.post("/teach", response_model=TeachResponse)
def teach_agent(request: Request, payload: TeachRequest) -> TeachResponse:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        config = state["config"]
        if len(payload.input_data) != int(config.network.input_size):
            raise HTTPException(status_code=422, detail="input_data length does not match network.input_size")
        if len(payload.target_data) != int(config.network.output_size):
            raise HTTPException(status_code=422, detail="target_data length does not match network.output_size")
        agent = state["agent"]
        device = state["device"]

    input_tensor = torch.tensor([payload.input_data], dtype=torch.float32, device=device)
    target_tensor = torch.tensor([payload.target_data], dtype=torch.float32, device=device)
    result = agent.learn_new(input_tensor, target_tensor)

    with STATE_LOCK:
        state = _ensure_state()
        state["metrics"].inc("learn_events")

    return TeachResponse(
        status=str(result.get("status", "unknown")),
        memory_id=result.get("memory_id"),
        loss=float(result.get("loss", 0.0)),
        energy_remaining=float(agent.energy),
    )


@app.post("/ask", response_model=PredictionResponse)
def ask_agent(request: Request, payload: PredictionRequest) -> PredictionResponse:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        agent = state["agent"]
        config = state["config"]
        device = state["device"]
        if len(payload.input_data) != int(config.network.input_size):
            raise HTTPException(status_code=422, detail="input_data length does not match network.input_size")

    input_tensor = torch.tensor([payload.input_data], dtype=torch.float32, device=device)
    memory_context = None
    recall_status = "no_context_requested"

    if payload.memory_key:
        payload_data = agent.memory.retrieve_memory(payload.memory_key, reinforce=False)
        if payload_data is None:
            recall_status = "forgotten"
        else:
            recall_status = "success"
            memory_context = payload_data.get("input") if isinstance(payload_data, dict) else None
            if torch.is_tensor(memory_context):
                memory_context = memory_context.to(device)

    with torch.no_grad():
        output, meta = agent.network(input_tensor, memory_context=memory_context)

    return PredictionResponse(
        prediction=output.detach().cpu().tolist()[0],
        uncertainty=float(meta["uncertainty"]),
        recall_status=recall_status,
        energy=float(agent.energy),
    )


@app.post("/sleep")
def trigger_sleep(request: Request) -> Dict[str, Any]:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        agent = state["agent"]
        config = state["config"]
        metrics = state["metrics"]

    result = agent.sleep()
    agent.memory.save_state()
    agent.network.save_checkpoint(
        config.runtime.checkpoint_path,
        optimizer_state=agent.optimizer.state_dict(),
    )

    with STATE_LOCK:
        state = _ensure_state()
        metrics = state["metrics"]
        metrics.inc("sleep_events")
        metrics.flush()
    return {"status": "slept", "result": result}


@app.get("/memories", response_model=MemoryListResponse)
def list_memories(
    request: Request,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    stable: Optional[bool] = Query(None),
    last_reviewed_before: Optional[float] = Query(None),
    last_reviewed_after: Optional[float] = Query(None),
    min_strength: Optional[float] = Query(None),
    max_strength: Optional[float] = Query(None),
) -> MemoryListResponse:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        items = state["agent"].memory.list_memories(
            offset=offset,
            limit=limit,
            stable=stable,
            last_reviewed_before=last_reviewed_before,
            last_reviewed_after=last_reviewed_after,
            min_strength=min_strength,
            max_strength=max_strength,
        )
        return MemoryListResponse(items=items, count=len(items), offset=offset, limit=limit)


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics(request: Request) -> MetricsResponse:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        metrics = state["metrics"]
        return MetricsResponse(trends=metrics.trends(window=20), counters=metrics.counters)


@app.post("/reset")
def reset_state(request: Request, dev: bool = Query(False)) -> Dict[str, str]:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        if not dev or not bool(state["config"].runtime.dev_reset_enabled):
            raise HTTPException(status_code=403, detail="reset is disabled")
        state["agent"].memory.reset()
        return {"status": "reset"}


@app.post("/organizations", response_model=CreateOrganizationResponse)
def create_organization(request: Request, payload: CreateOrganizationRequest) -> CreateOrganizationResponse:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        store: TenantMemoryStore = state["tenant_store"]
    try:
        created = store.create_org(name=payload.name, org_id=payload.org_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return CreateOrganizationResponse(**created)


@app.post("/users", response_model=CreateUserResponse)
def create_user(request: Request, payload: CreateUserRequest) -> CreateUserResponse:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        store: TenantMemoryStore = state["tenant_store"]
    try:
        created = store.create_user(
            org_id=payload.org_id,
            email=payload.email,
            name=payload.name,
            user_id=payload.user_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return CreateUserResponse(**created)


@app.put("/concepts", response_model=ConceptResponse)
def upsert_concept(request: Request, payload: ConceptRequest) -> ConceptResponse:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        store: TenantMemoryStore = state["tenant_store"]
    try:
        concept = store.upsert_concept(
            org_id=payload.org_id,
            concept_id=payload.concept_id,
            title=payload.title,
            prompt=payload.prompt,
            answer=payload.answer,
            explanation=payload.explanation,
            tags=payload.tags,
            difficulty=float(payload.difficulty),
            source=payload.source,
            active=bool(payload.active),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ConceptResponse(**concept)


@app.get("/concepts", response_model=ConceptListResponse)
def list_concepts(
    request: Request,
    org_id: str = Query(...),
    active: Optional[bool] = Query(True),
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
) -> ConceptListResponse:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        store: TenantMemoryStore = state["tenant_store"]
    try:
        items = store.list_concepts(org_id=org_id, active=active, limit=limit, offset=offset)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ConceptListResponse(items=items, count=len(items), offset=offset, limit=limit)


@app.get("/concepts/{concept_id}", response_model=ConceptResponse)
def get_concept(request: Request, concept_id: str, org_id: str = Query(...)) -> ConceptResponse:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        store: TenantMemoryStore = state["tenant_store"]
    concept = store.get_concept(org_id=org_id, concept_id=concept_id)
    if concept is None:
        raise HTTPException(status_code=404, detail="concept_not_found")
    return ConceptResponse(**concept)


@app.post("/record-attempt")
def record_attempt(request: Request, payload: RecordAttemptRequest) -> Dict[str, Any]:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        store: TenantMemoryStore = state["tenant_store"]
        memory_cfg = state["config"].memory
    try:
        result = store.record_attempt(
            user_id=payload.user_id,
            concept_id=payload.concept_id,
            correct=payload.correct,
            response_ms=payload.response_ms,
            attempted_at=payload.attempted_at,
            decay_rate=float(memory_cfg.decay_rate),
            retrieval_difficulty=float(memory_cfg.default_difficulty),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.post("/daily-session", response_model=DailySessionResponse)
def run_daily_session(request: Request, payload: DailySessionRequest) -> DailySessionResponse:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        store: TenantMemoryStore = state["tenant_store"]
        runtime = state["config"].runtime
        memory_cfg = state["config"].memory

    reference_time = float(payload.now if payload.now is not None else time.time())
    days = int(payload.window_days or runtime.analytics_window_days)
    recorded_attempt: Optional[Dict[str, Any]] = None

    try:
        if payload.attempt is not None:
            attempted_at = payload.attempt.attempted_at if payload.attempt.attempted_at is not None else reference_time
            recorded_attempt = store.record_attempt(
                user_id=payload.user_id,
                concept_id=payload.attempt.concept_id,
                correct=payload.attempt.correct,
                response_ms=payload.attempt.response_ms,
                attempted_at=float(attempted_at),
                decay_rate=float(memory_cfg.decay_rate),
                retrieval_difficulty=float(memory_cfg.default_difficulty),
            )

        queue = store.get_review_queue(user_id=payload.user_id, limit=int(payload.limit), now=reference_time)
        due_items = [item for item in queue if bool(item.get("is_due"))]
        analytics = store.compute_user_analytics(user_id=payload.user_id, window_days=days)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    next_item: Optional[Dict[str, Any]] = due_items[0] if due_items else (queue[0] if queue else None)
    if due_items:
        recommended_action = "review_due_items"
    elif queue:
        recommended_action = "build_consistency"
    else:
        recommended_action = "seed_attempts"

    with STATE_LOCK:
        state = _ensure_state()
        metrics: MetricsTracker = state["metrics"]
        metrics.inc("daily_sessions")
        if recorded_attempt is not None:
            metrics.inc("daily_session_attempts_recorded")
        try:
            store.log_audit_event(
                action="daily_session_loaded",
                entity_type="daily_session",
                entity_id=payload.user_id,
                org_id=store._user_org_id(payload.user_id),
                user_id=payload.user_id,
                metadata={
                    "due_count": len(due_items),
                    "total_count": len(queue),
                    "recorded_attempt": recorded_attempt is not None,
                    "recommended_action": recommended_action,
                },
                created_at=reference_time,
            )
        except Exception:
            pass

    return DailySessionResponse(
        user_id=payload.user_id,
        queue=queue,
        due_count=int(len(due_items)),
        total_count=int(len(queue)),
        next_item=next_item,
        analytics=analytics,
        recorded_attempt=recorded_attempt,
        recommended_action=recommended_action,
    )


@app.get("/review-queue", response_model=ReviewQueueResponse)
def get_review_queue(
    request: Request,
    user_id: str = Query(...),
    limit: int = Query(50, ge=1, le=500),
) -> ReviewQueueResponse:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        store: TenantMemoryStore = state["tenant_store"]
    try:
        items = store.get_review_queue(user_id=user_id, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ReviewQueueResponse(user_id=user_id, items=items)


@app.get("/analytics", response_model=AnalyticsResponse)
def get_retention_analytics(
    request: Request,
    org_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    window_days: Optional[int] = Query(None, ge=1),
) -> AnalyticsResponse:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        store: TenantMemoryStore = state["tenant_store"]
        days = int(window_days or state["config"].runtime.analytics_window_days)

    if user_id:
        data = store.compute_user_analytics(user_id=user_id, window_days=days)
        return AnalyticsResponse(scope="user", data=data)
    if org_id:
        data = store.compute_org_analytics(org_id=org_id, window_days=days)
        return AnalyticsResponse(scope="organization", data=data)
    raise HTTPException(status_code=422, detail="Provide org_id or user_id")


@app.get("/audit-log", response_model=AuditLogResponse)
def get_audit_log(
    request: Request,
    org_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
) -> AuditLogResponse:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        store: TenantMemoryStore = state["tenant_store"]
    items = store.list_audit_events(org_id=org_id, user_id=user_id, action=action, limit=limit)
    return AuditLogResponse(items=items, count=len(items))


@app.post("/roi/report", response_model=RoiReportResponse)
def build_roi_report(request: Request, payload: RoiReportRequest) -> RoiReportResponse:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        store: TenantMemoryStore = state["tenant_store"]
    try:
        report = store.build_roi_report(
            org_id=payload.org_id,
            learners=int(payload.learners),
            training_hours_saved_per_learner=float(payload.training_hours_saved_per_learner),
            cost_per_training_hour=float(payload.cost_per_training_hour),
            annual_contract_value=float(payload.annual_contract_value),
            window_days=int(payload.window_days),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return RoiReportResponse(**report)


@app.get("/exports/review-queue.csv", response_class=PlainTextResponse)
def export_review_queue_csv(
    request: Request,
    user_id: str = Query(...),
    limit: int = Query(500, ge=1, le=5000),
) -> PlainTextResponse:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        store: TenantMemoryStore = state["tenant_store"]
    try:
        items = store.get_review_queue(user_id=user_id, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    rows = []
    for item in items:
        concept = item.get("concept") or {}
        rows.append(
            {
                **item,
                "concept_title": concept.get("title"),
                "concept_tags": concept.get("tags"),
                "concept": concept,
            }
        )
    csv_text = _rows_to_csv(
        rows,
        [
            "concept_id",
            "concept_title",
            "is_due",
            "retention",
            "risk_score",
            "reason_code",
            "reason",
            "next_review_at",
            "strength",
            "stability",
            "concept_tags",
        ],
    )
    return PlainTextResponse(csv_text, media_type="text/csv")


@app.get("/exports/audit-log.csv", response_class=PlainTextResponse)
def export_audit_log_csv(
    request: Request,
    org_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    limit: int = Query(1000, ge=1, le=5000),
) -> PlainTextResponse:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        store: TenantMemoryStore = state["tenant_store"]
    items = store.list_audit_events(org_id=org_id, user_id=user_id, action=action, limit=limit)
    csv_text = _rows_to_csv(items, ["id", "created_at", "org_id", "user_id", "action", "entity_type", "entity_id", "metadata"])
    return PlainTextResponse(csv_text, media_type="text/csv")


@app.get("/exports/analytics.csv", response_class=PlainTextResponse)
def export_analytics_csv(
    request: Request,
    org_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    window_days: Optional[int] = Query(None, ge=1),
) -> PlainTextResponse:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        store: TenantMemoryStore = state["tenant_store"]
        days = int(window_days or state["config"].runtime.analytics_window_days)

    if user_id:
        data = store.compute_user_analytics(user_id=user_id, window_days=days)
        data["scope"] = "user"
    elif org_id:
        data = store.compute_org_analytics(org_id=org_id, window_days=days)
        data["scope"] = "organization"
    else:
        raise HTTPException(status_code=422, detail="Provide org_id or user_id")
    csv_text = _rows_to_csv([data], list(data.keys()))
    return PlainTextResponse(csv_text, media_type="text/csv")


@app.post("/exports/roi-report.csv", response_class=PlainTextResponse)
def export_roi_report_csv(request: Request, payload: RoiReportRequest) -> PlainTextResponse:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        store: TenantMemoryStore = state["tenant_store"]
    try:
        report = store.build_roi_report(
            org_id=payload.org_id,
            learners=int(payload.learners),
            training_hours_saved_per_learner=float(payload.training_hours_saved_per_learner),
            cost_per_training_hour=float(payload.cost_per_training_hour),
            annual_contract_value=float(payload.annual_contract_value),
            window_days=int(payload.window_days),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    row = {
        "org_id": report["org_id"],
        "generated_at": report["generated_at"],
        "window_days": report["window_days"],
        "learners": report["inputs"]["learners"],
        "training_hours_saved_per_learner": report["inputs"]["training_hours_saved_per_learner"],
        "cost_per_training_hour": report["inputs"]["cost_per_training_hour"],
        "annual_contract_value": report["inputs"]["annual_contract_value"],
        "gross_savings": report["gross_savings"],
        "net_savings": report["net_savings"],
        "roi_multiple": report["roi_multiple"],
        "payback_months": report["payback_months"],
        "executive_summary": report["executive_summary"],
    }
    csv_text = _rows_to_csv([row], list(row.keys()))
    return PlainTextResponse(csv_text, media_type="text/csv")


@app.post("/pilot/setup", response_model=PilotSetupResponse)
def setup_pilot(request: Request, payload: PilotSetupRequest) -> PilotSetupResponse:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        store: TenantMemoryStore = state["tenant_store"]
    try:
        result = store.setup_pilot(
            org_id=payload.org_id,
            name=payload.name,
            treatment_ratio=float(payload.treatment_ratio),
            random_seed=int(payload.random_seed),
            pilot_id=payload.pilot_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return PilotSetupResponse(**result)


@app.get("/pilot/run", response_model=PilotRunResponse)
def get_pilot_run(request: Request, pilot_id: str = Query(...)) -> PilotRunResponse:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        store: TenantMemoryStore = state["tenant_store"]
    try:
        result = store.get_pilot_run(pilot_id=pilot_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return PilotRunResponse(**result)


@app.post("/pilot/baseline", response_model=PilotBaselineResponse)
def capture_pilot_baseline(request: Request, payload: PilotBaselineRequest) -> PilotBaselineResponse:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        store: TenantMemoryStore = state["tenant_store"]
    try:
        result = store.capture_pilot_baseline(
            pilot_id=payload.pilot_id,
            window_days=int(payload.window_days),
            captured_at=payload.captured_at,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return PilotBaselineResponse(**result)


@app.post("/pilot/evaluate", response_model=PilotEvaluateResponse)
def evaluate_pilot(request: Request, payload: PilotEvaluateRequest) -> PilotEvaluateResponse:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        store: TenantMemoryStore = state["tenant_store"]
        runtime = state["config"].runtime
    result = store.evaluate_pilot(
        retained_mastery_treatment=float(payload.retained_mastery_treatment),
        retained_mastery_control=float(payload.retained_mastery_control),
        forgetting_velocity_treatment=float(payload.forgetting_velocity_treatment),
        forgetting_velocity_control=float(payload.forgetting_velocity_control),
        review_efficiency_treatment=float(payload.review_efficiency_treatment),
        review_efficiency_control=float(payload.review_efficiency_control),
        sample_size=int(payload.sample_size),
        min_retained_mastery_lift=float(runtime.pilot_min_retained_mastery_lift),
        min_forgetting_velocity_reduction=float(runtime.pilot_min_forgetting_velocity_reduction),
        min_review_efficiency_lift=float(runtime.pilot_min_review_efficiency_lift),
        confidence_z_threshold=float(runtime.pilot_confidence_z_threshold),
        min_sample_size=int(runtime.pilot_min_sample_size),
        onboarding_hours=float(payload.onboarding_hours),
        max_onboarding_hours=float(runtime.pilot_max_onboarding_hours),
        org_id=payload.org_id,
    )
    return PilotEvaluateResponse(**result)


@app.get("/pilot/history", response_model=PilotHistoryResponse)
def pilot_history(
    request: Request,
    org_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
) -> PilotHistoryResponse:
    with STATE_LOCK:
        state = _ensure_state()
        _require_api_key(request, state)
        store: TenantMemoryStore = state["tenant_store"]
    items = store.list_pilot_evaluations(org_id=org_id, limit=limit)
    return PilotHistoryResponse(items=items, count=len(items))


def main() -> None:
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
