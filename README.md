# Cognitive Sim

Cognitive Sim is a biologically-inspired agent simulation and recall-training platform with:

- Ebbinghaus-style memory decay
- spaced repetition review scheduling
- energy economy (learn/review/sleep tradeoffs)
- PyTorch neural core
- FastAPI + CLI interfaces
- Streamlit learner/admin frontends
- multi-tenant learning workflows for organizations, users, concepts, attempts, review queues, analytics, audit logs, ROI reports, and pilot gates
- run artifacts (JSONL event log + summary report)

---

## Repository Layout

- Core runtime: `Cognitive_Sim/src/`
- Configs: `Cognitive_Sim/configs/`
- Tests: `Cognitive_Sim/tests/`
- Dependency sets: `Cognitive_Sim/requirements/`
- Changelog: `Cognitive_Sim/CHANGELOG.md`
- Runtime data/artifacts: `Cognitive_Sim/data/` and `Cognitive_Sim/logs/`

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

Run Daily Recall Coach frontend:

```bash
streamlit run src/daily_recall_coach_app.py
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

Run simulation suite scenarios:

```bash
python -m src.simulation_suite all --env development --steps 60 --seed 42
```

```bash
python -m src.simulation_suite all --env testing --steps 60 --seed 42 --sim-step-seconds 60
```

```bash
python -m src.simulation_suite ab-policy --env testing --steps 40 --seed 123
```

Launch Streamlit frontend dashboard:

```bash
streamlit run src/frontend_app.py
```

Launch Daily Recall Coach via package script:

```bash
cognitive-sim-daily-recall-ui
```

Dashboard includes:
- status and metrics refresh
- teach / ask / sleep controls
- tenant org/user/attempt flows
- review queue + analytics
- memory explorer (`/memories`)
- pilot evaluation + pilot history (`/pilot/history`)
- one-click demo data seeding (sidebar: `Seed Demo Data`) with backdated attempts
- demo readiness warnings (no memories, no due items, review-skipped signals)

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
- `PUT /concepts` — create or update tenant-owned learning content concepts
- `GET /concepts` / `GET /concepts/{concept_id}` — list or retrieve learning content concepts
- `POST /record-attempt` — write per-user concept attempt and update memory state
- `POST /daily-session` — single-call daily workflow (optional attempt write + refreshed queue + analytics)
- `GET /review-queue` — get prioritized concept review queue per user
- `GET /analytics` — user/org retention analytics (money layer)
- `GET /audit-log` — tenant/user action audit trail
- `POST /roi/report` — executive ROI report for B2B business cases
- `GET /exports/review-queue.csv`, `GET /exports/analytics.csv`, `GET /exports/audit-log.csv`, `POST /exports/roi-report.csv` — CSV exports
- `POST /pilot/setup` — create a randomized control/treatment pilot cohort from tenant users
- `GET /pilot/run` — retrieve pilot cohort assignment and latest baseline snapshot
- `POST /pilot/baseline` — capture pilot baseline metrics for control/treatment cohorts
- `POST /pilot/evaluate` — objective go/no-go evaluation with confidence checks
- `GET /pilot/history` — historical pilot gate decisions

Authentication:

- When `runtime.require_api_key` is true (see [`production.yaml`](Cognitive_Sim/configs/production.yaml:1)), send the key via `x-api-key: <value>` or `Authorization: Bearer <value>`.

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
curl -X PUT http://localhost:8000/concepts \
  -H "Content-Type: application/json" \
  -d '{"org_id":"org_acme","concept_id":"cardiology_001","title":"Recognize unstable angina","prompt":"Which symptoms require escalation?","answer":"Chest pain at rest, ECG changes, or elevated biomarkers.","explanation":"Reinforces safe triage and escalation behavior.","tags":["clinical","triage"],"difficulty":2.0,"source":"cardiology_playbook"}'
```

```bash
curl "http://localhost:8000/concepts?org_id=org_acme&limit=20"
```

```bash
curl -X POST http://localhost:8000/record-attempt \
  -H "Content-Type: application/json" \
  -d '{"user_id":"usr_001","concept_id":"cardiology_001","correct":true,"response_ms":850}'
```

```bash
curl -X POST http://localhost:8000/daily-session \
  -H "Content-Type: application/json" \
  -d '{"user_id":"usr_001","limit":20,"window_days":30}'
```

```bash
curl -X POST http://localhost:8000/daily-session \
  -H "Content-Type: application/json" \
  -d '{"user_id":"usr_001","limit":20,"attempt":{"concept_id":"cardiology_001","correct":true,"response_ms":810}}'
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
curl -X POST http://localhost:8000/roi/report \
  -H "Content-Type: application/json" \
  -d '{"org_id":"org_acme","learners":10000,"training_hours_saved_per_learner":2.5,"cost_per_training_hour":60,"annual_contract_value":300000,"window_days":30}'
```

```bash
curl "http://localhost:8000/audit-log?org_id=org_acme&limit=20"
```

```bash
curl "http://localhost:8000/exports/review-queue.csv?user_id=usr_001&limit=20"
```

```bash
curl -X POST http://localhost:8000/pilot/setup \
  -H "Content-Type: application/json" \
  -d '{"org_id":"org_acme","name":"Q2 onboarding pilot","treatment_ratio":0.5,"random_seed":42}'
```

```bash
curl "http://localhost:8000/pilot/run?pilot_id=pilot_001"
```

```bash
curl -X POST http://localhost:8000/pilot/baseline \
  -H "Content-Type: application/json" \
  -d '{"pilot_id":"pilot_001","window_days":30}'
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
  - `runtime.require_api_key` + `runtime.api_key` to enforce API auth
  - `runtime.metrics_flush_every` + `runtime.metrics_flush_interval_seconds` to buffer metric writes
  - `memory.payload_save_limit` to cap memory payload writes per save for large stores

Multi-tenant runtime options:

- `runtime.tenant_db_path`: SQLite file for tenant/user memory state and attempts
- `runtime.analytics_window_days`: default analytics horizon for `/analytics` and `/daily-session`
- `runtime.pilot_min_retained_mastery_lift`: minimum lift threshold for pilot pass
- `runtime.pilot_min_forgetting_velocity_reduction`: minimum forgetting reduction threshold
- `runtime.pilot_min_review_efficiency_lift`: minimum review efficiency lift threshold
- `runtime.pilot_confidence_z_threshold`: minimum z-score confidence threshold
- `runtime.pilot_min_sample_size`: minimum cohort size for gate decision
- `runtime.pilot_max_onboarding_hours`: implementation economics guardrail

---

## Daily Recall Coach workflow

The Daily Recall Coach app is a focused learner workflow for recurring daily practice.

1. Bootstrap learner identity using tenant APIs (`/organizations`, `/users`) from the app sidebar.
2. Load a daily session with `/daily-session` to receive:
   - prioritized queue
   - due count and next best concept
   - learner analytics snapshot
   - recommended action (`review_due_items`, `build_consistency`, `seed_attempts`)
3. Submit an attempt through the same endpoint by including `attempt` in the request body.

This keeps the primary UX as a single round-trip per learner action while preserving compatibility with existing endpoints.

---

## Functionality Checklist

The reclaimed project currently documents and supports:

- Core cognitive simulation: memory decay, stability tracking, spaced review scheduling, energy/reward tradeoffs, sleep consolidation, deterministic seeding, and checkpoint persistence.
- Neural/runtime layer: PyTorch network inference/training, optimizer scheduling, metrics buffering, JSONL run logs, and summary reports.
- CLI workflows: simulation runs, verification checks, simulation-suite scenarios, policy A/B runs, reproducibility checks, and persistence/restart checks.
- API workflows: health/status, teach, ask, sleep, memory browsing, metrics, guarded reset, and optional API-key authentication.
- Tenant workflows: organizations, users, concepts, attempts, daily learner sessions, review queues, analytics, audit logs, and tenant SQLite persistence.
- Business workflows: ROI report generation and CSV exports for review queues, analytics, audit logs, and ROI reports.
- Pilot workflows: pilot setup, randomized control/treatment assignment, baseline capture, objective go/no-go evaluation, and pilot history.
- Frontends: full Streamlit dashboard, Daily Recall Coach, demo-data seeding, review UX, analytics, memory explorer, pilot evaluation, and pilot history views.
- Operations: environment-specific YAML configuration, dependency sets, Docker/Docker Compose startup, and automated unit/integration tests.

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
- simulation-suite runs (policy A/B comparison, reproducibility checks, persistence/restart)

---

## Change History

See `Cognitive_Sim/CHANGELOG.md` for full details of the latest audit and implementation pass.
