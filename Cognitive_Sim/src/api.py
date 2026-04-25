from contextlib import asynccontextmanager
from threading import Lock
import time
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query, status
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


class RecordAttemptRequest(BaseModel):
    user_id: str
    concept_id: str
    correct: bool
    response_ms: Optional[float] = None
    attempted_at: Optional[float] = None


class ReviewQueueResponse(BaseModel):
    user_id: str
    items: List[Dict[str, Any]]


class AnalyticsResponse(BaseModel):
    scope: str
    data: Dict[str, Any]


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config("development")
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
    metrics = MetricsTracker(artifact_dir=config.runtime.artifact_dir)
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
        if "tenant_store" in STATE:
            STATE["tenant_store"].close()


app = FastAPI(title="Cognitive Simulation API", lifespan=lifespan)


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "online", "message": "Cognitive Agent Ready"}


@app.get("/status", response_model=AgentStatus)
def get_status() -> AgentStatus:
    with STATE_LOCK:
        state = _ensure_state()
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
def teach_agent(request: TeachRequest) -> TeachResponse:
    with STATE_LOCK:
        state = _ensure_state()
        agent = state["agent"]
        config = state["config"]
        if len(request.input_data) != int(config.network.input_size):
            raise HTTPException(status_code=422, detail="input_data length does not match network.input_size")
        if len(request.target_data) != int(config.network.output_size):
            raise HTTPException(status_code=422, detail="target_data length does not match network.output_size")

        device = state["device"]
        input_tensor = torch.tensor([request.input_data], dtype=torch.float32, device=device)
        target_tensor = torch.tensor([request.target_data], dtype=torch.float32, device=device)
        result = agent.learn_new(input_tensor, target_tensor)
        state["metrics"].inc("learn_events")

        return TeachResponse(
            status=str(result.get("status", "unknown")),
            memory_id=result.get("memory_id"),
            loss=float(result.get("loss", 0.0)),
            energy_remaining=float(agent.energy),
        )


@app.post("/ask", response_model=PredictionResponse)
def ask_agent(request: PredictionRequest) -> PredictionResponse:
    with STATE_LOCK:
        state = _ensure_state()
        agent = state["agent"]
        config = state["config"]
        if len(request.input_data) != int(config.network.input_size):
            raise HTTPException(status_code=422, detail="input_data length does not match network.input_size")

        device = state["device"]
        input_tensor = torch.tensor([request.input_data], dtype=torch.float32, device=device)
        memory_context = None
        recall_status = "no_context_requested"

        if request.memory_key:
            payload = agent.memory.retrieve_memory(request.memory_key, reinforce=False)
            if payload is None:
                recall_status = "forgotten"
            else:
                recall_status = "success"
                memory_context = payload.get("input") if isinstance(payload, dict) else None
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
def trigger_sleep() -> Dict[str, Any]:
    with STATE_LOCK:
        state = _ensure_state()
        result = state["agent"].sleep()
        state["metrics"].inc("sleep_events")
        state["agent"].memory.save_state()
        state["agent"].network.save_checkpoint(
            state["config"].runtime.checkpoint_path,
            optimizer_state=state["agent"].optimizer.state_dict(),
        )
        return {"status": "slept", "result": result}


@app.get("/memories", response_model=MemoryListResponse)
def list_memories(
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
def get_metrics() -> MetricsResponse:
    with STATE_LOCK:
        state = _ensure_state()
        metrics = state["metrics"]
        return MetricsResponse(trends=metrics.trends(window=20), counters=metrics.counters)


@app.post("/reset")
def reset_state(dev: bool = Query(False)) -> Dict[str, str]:
    with STATE_LOCK:
        state = _ensure_state()
        if not dev or not bool(state["config"].runtime.dev_reset_enabled):
            raise HTTPException(status_code=403, detail="reset is disabled")
        state["agent"].memory.reset()
        return {"status": "reset"}


@app.post("/organizations", response_model=CreateOrganizationResponse)
def create_organization(request: CreateOrganizationRequest) -> CreateOrganizationResponse:
    with STATE_LOCK:
        state = _ensure_state()
        store: TenantMemoryStore = state["tenant_store"]
        try:
            created = store.create_org(name=request.name, org_id=request.org_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return CreateOrganizationResponse(**created)


@app.post("/users", response_model=CreateUserResponse)
def create_user(request: CreateUserRequest) -> CreateUserResponse:
    with STATE_LOCK:
        state = _ensure_state()
        store: TenantMemoryStore = state["tenant_store"]
        try:
            created = store.create_user(
                org_id=request.org_id,
                email=request.email,
                name=request.name,
                user_id=request.user_id,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return CreateUserResponse(**created)


@app.post("/record-attempt")
def record_attempt(request: RecordAttemptRequest) -> Dict[str, Any]:
    with STATE_LOCK:
        state = _ensure_state()
        store: TenantMemoryStore = state["tenant_store"]
        try:
            result = store.record_attempt(
                user_id=request.user_id,
                concept_id=request.concept_id,
                correct=request.correct,
                response_ms=request.response_ms,
                attempted_at=request.attempted_at,
                decay_rate=float(state["config"].memory.decay_rate),
                retrieval_difficulty=float(state["config"].memory.default_difficulty),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return result


@app.get("/review-queue", response_model=ReviewQueueResponse)
def get_review_queue(user_id: str = Query(...), limit: int = Query(50, ge=1, le=500)) -> ReviewQueueResponse:
    with STATE_LOCK:
        state = _ensure_state()
        store: TenantMemoryStore = state["tenant_store"]
        try:
            items = store.get_review_queue(user_id=user_id, limit=limit)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return ReviewQueueResponse(user_id=user_id, items=items)


@app.get("/analytics", response_model=AnalyticsResponse)
def get_retention_analytics(
    org_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    window_days: Optional[int] = Query(None, ge=1),
) -> AnalyticsResponse:
    with STATE_LOCK:
        state = _ensure_state()
        store: TenantMemoryStore = state["tenant_store"]
        days = int(window_days or state["config"].runtime.analytics_window_days)

        if user_id:
            data = store.compute_user_analytics(user_id=user_id, window_days=days)
            return AnalyticsResponse(scope="user", data=data)
        if org_id:
            data = store.compute_org_analytics(org_id=org_id, window_days=days)
            return AnalyticsResponse(scope="organization", data=data)
        raise HTTPException(status_code=422, detail="Provide org_id or user_id")


@app.post("/pilot/setup", response_model=PilotSetupResponse)
def setup_pilot(request: PilotSetupRequest) -> PilotSetupResponse:
    with STATE_LOCK:
        state = _ensure_state()
        store: TenantMemoryStore = state["tenant_store"]
        try:
            result = store.setup_pilot(
                org_id=request.org_id,
                name=request.name,
                treatment_ratio=float(request.treatment_ratio),
                random_seed=int(request.random_seed),
                pilot_id=request.pilot_id,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return PilotSetupResponse(**result)


@app.get("/pilot/run", response_model=PilotRunResponse)
def get_pilot_run(pilot_id: str = Query(...)) -> PilotRunResponse:
    with STATE_LOCK:
        state = _ensure_state()
        store: TenantMemoryStore = state["tenant_store"]
        try:
            result = store.get_pilot_run(pilot_id=pilot_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return PilotRunResponse(**result)


@app.post("/pilot/baseline", response_model=PilotBaselineResponse)
def capture_pilot_baseline(request: PilotBaselineRequest) -> PilotBaselineResponse:
    with STATE_LOCK:
        state = _ensure_state()
        store: TenantMemoryStore = state["tenant_store"]
        try:
            result = store.capture_pilot_baseline(
                pilot_id=request.pilot_id,
                window_days=int(request.window_days),
                captured_at=request.captured_at,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return PilotBaselineResponse(**result)


@app.post("/pilot/evaluate", response_model=PilotEvaluateResponse)
def evaluate_pilot(request: PilotEvaluateRequest) -> PilotEvaluateResponse:
    with STATE_LOCK:
        state = _ensure_state()
        store: TenantMemoryStore = state["tenant_store"]
        runtime = state["config"].runtime
        result = store.evaluate_pilot(
            retained_mastery_treatment=float(request.retained_mastery_treatment),
            retained_mastery_control=float(request.retained_mastery_control),
            forgetting_velocity_treatment=float(request.forgetting_velocity_treatment),
            forgetting_velocity_control=float(request.forgetting_velocity_control),
            review_efficiency_treatment=float(request.review_efficiency_treatment),
            review_efficiency_control=float(request.review_efficiency_control),
            sample_size=int(request.sample_size),
            min_retained_mastery_lift=float(runtime.pilot_min_retained_mastery_lift),
            min_forgetting_velocity_reduction=float(runtime.pilot_min_forgetting_velocity_reduction),
            min_review_efficiency_lift=float(runtime.pilot_min_review_efficiency_lift),
            confidence_z_threshold=float(runtime.pilot_confidence_z_threshold),
            min_sample_size=int(runtime.pilot_min_sample_size),
            onboarding_hours=float(request.onboarding_hours),
            max_onboarding_hours=float(runtime.pilot_max_onboarding_hours),
            org_id=request.org_id,
        )
        return PilotEvaluateResponse(**result)


@app.get("/pilot/history", response_model=PilotHistoryResponse)
def pilot_history(org_id: Optional[str] = Query(None), limit: int = Query(50, ge=1, le=500)) -> PilotHistoryResponse:
    with STATE_LOCK:
        state = _ensure_state()
        store: TenantMemoryStore = state["tenant_store"]
        items = store.list_pilot_evaluations(org_id=org_id, limit=limit)
        return PilotHistoryResponse(items=items, count=len(items))


def main() -> None:
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
