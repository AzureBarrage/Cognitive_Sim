# Cognitive Simulation (Cognitive_Sim)

Cognitive_Sim is a biologically-inspired agent simulation with:

- Ebbinghaus-style memory decay
- spaced repetition review scheduling
- energy economy (learn/review/sleep tradeoffs)
- PyTorch neural core
- FastAPI + CLI interfaces
- run artifacts (JSONL event log + summary report)

---

## Repository Layout

- Core runtime: `Cognitive_Sim/src/`
- Configs: `Cognitive_Sim/configs/`
- Tests: `Cognitive_Sim/tests/`
- Dependency sets: `Cognitive_Sim/requirements/`
- Changelog: `Cognitive_Sim/CHANGELOG.md`

---

## Local Quickstart

```bash
cd Cognitive_Sim
python -m pip install -r requirements/dev.txt
python -m pytest -q
```

Run deterministic simulation:

```bash
python -m src.main run --env development --steps 50 --seed 42 --fresh
```

Validate configuration and invariants:

```bash
python -m src.main verify --env development --seed 42
```

Run API server:

```bash
python -m src.api
```

---

## Docker Quickstart

```bash
cd Cognitive_Sim
docker-compose up --build
```

API base URL: `http://localhost:8000`

---

## CLI Commands

Run simulation:

```bash
python -m src.main run --env development --steps 100 --seed 42
```

Verify runtime assumptions:

```bash
python -m src.main verify --env testing --seed 123
```

---

## API Endpoints

- `GET /status` — energy, rewards, memory count, success rates, uptime
- `POST /teach` — train on one input/target sample
- `POST /ask` — inference with optional memory context
- `POST /sleep` — force consolidation and checkpoint persistence
- `GET /memories` — paged memory listing with filters
- `GET /metrics` — trend metrics and event counters
- `POST /reset` — dev-guarded reset endpoint
- `POST /organizations` — create tenant organization
- `POST /users` — create user under organization
- `POST /record-attempt` — write per-user concept attempt and update memory state
- `GET /review-queue` — get prioritized concept review queue per user
- `GET /analytics` — user/org retention analytics (money layer)
- `POST /pilot/evaluate` — objective go/no-go evaluation with confidence checks
- `GET /pilot/history` — historical pilot gate decisions

Examples:

```bash
curl http://localhost:8000/status
```

```bash
curl -X POST http://localhost:8000/teach \
  -H "Content-Type: application/json" \
  -d '{"input_data":[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],"target_data":[0.2,0.2,0.2,0.2,0.2]}'
```

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"input_data":[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]}'
```

```bash
curl "http://localhost:8000/memories?limit=20&stable=true"
```

```bash
curl http://localhost:8000/metrics
```

```bash
curl -X POST http://localhost:8000/organizations \
  -H "Content-Type: application/json" \
  -d '{"name":"Acme Learning","org_id":"org_acme"}'
```

```bash
curl -X POST http://localhost:8000/users \
  -H "Content-Type: application/json" \
  -d '{"org_id":"org_acme","email":"learner@example.com","user_id":"usr_001"}'
```

```bash
curl -X POST http://localhost:8000/record-attempt \
  -H "Content-Type: application/json" \
  -d '{"user_id":"usr_001","concept_id":"cardiology_001","correct":true,"response_ms":850}'
```

```bash
curl "http://localhost:8000/review-queue?user_id=usr_001&limit=20"
```

```bash
curl "http://localhost:8000/analytics?user_id=usr_001"
```

```bash
curl "http://localhost:8000/analytics?org_id=org_acme"
```

```bash
curl -X POST http://localhost:8000/pilot/evaluate \
  -H "Content-Type: application/json" \
  -d '{"org_id":"org_acme","sample_size":300,"onboarding_hours":25.0,"retained_mastery_treatment":0.80,"retained_mastery_control":0.70,"forgetting_velocity_treatment":0.20,"forgetting_velocity_control":0.35,"review_efficiency_treatment":0.74,"review_efficiency_control":0.62}'
```

```bash
curl "http://localhost:8000/pilot/history?org_id=org_acme&limit=20"
```

---

## Configuration

Primary environment files:

- `Cognitive_Sim/configs/development.yaml`
- `Cognitive_Sim/configs/testing.yaml`
- `Cognitive_Sim/configs/production.yaml`

Important sections:

- `memory`: decay model, stability, spacing/relearning knobs
- `network`: dimensions, loss type, clipping, device
- `agent`: energy economy and policy parameters
- `data`: deterministic/stochastic toy environment settings
- `runtime`: checkpoint path and run-artifact output path

Multi-tenant runtime options:

- `runtime.tenant_db_path`: SQLite file for tenant/user memory state and attempts
- `runtime.analytics_window_days`: default analytics horizon for `/analytics`
- `runtime.pilot_min_retained_mastery_lift`: minimum lift threshold for pilot pass
- `runtime.pilot_min_forgetting_velocity_reduction`: minimum forgetting reduction threshold
- `runtime.pilot_min_review_efficiency_lift`: minimum review efficiency lift threshold
- `runtime.pilot_confidence_z_threshold`: minimum z-score confidence threshold
- `runtime.pilot_min_sample_size`: minimum cohort size for gate decision
- `runtime.pilot_max_onboarding_hours`: implementation economics guardrail

---

## Artifacts and Persistence

Typical outputs after a run:

- Model checkpoint: `Cognitive_Sim/data/network_checkpoint.pt`
- Memory index + payload blobs: under `Cognitive_Sim/data/`
- Run metrics JSONL + summary: under `Cognitive_Sim/logs/runs/`

---

## Testing

Run all tests:

```bash
cd Cognitive_Sim
python -m pytest -q
```

Current test suite covers:

- memory decay and spaced repetition behavior
- policy boundary decisions
- persistence roundtrip
- API contract checks (`/status`, `/teach`, `/ask`, `/sleep`, `/memories`, `/metrics`, `/reset`)
- deterministic short simulation integration run

---

## Change History

See `Cognitive_Sim/CHANGELOG.md` for full details of the latest audit and implementation pass.
