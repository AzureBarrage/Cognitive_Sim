## Cognitive_Sim API Reference

Base URL: `http://localhost:8000`

### Authentication

If `runtime.require_api_key` is enabled in your environment config, supply the key via:

- `x-api-key: <value>`
- `Authorization: Bearer <value>`

### Endpoints

#### `GET /status`

Returns agent status, energy, memory counts, and uptime.

#### `POST /teach`

Trains on one input/target sample.

Payload:

```
{
  "input_data": [0.1, 0.1, ...],
  "target_data": [0.2, 0.2, ...],
  "label": "optional"
}
```

#### `POST /ask`

Runs inference with optional memory context.

Payload:

```
{
  "input_data": [0.1, 0.1, ...],
  "memory_key": "optional"
}
```

#### `POST /sleep`

Forces consolidation and checkpoint persistence.

#### `GET /memories`

Query params:

- `offset` (int, default 0)
- `limit` (int, default 50, max 500)
- `stable` (bool)
- `last_reviewed_before` (float epoch)
- `last_reviewed_after` (float epoch)
- `min_strength` (float)
- `max_strength` (float)

#### `GET /metrics`

Returns recent trend averages and event counters.

#### `POST /reset`

Dev-guarded reset. Requires `dev=true` and `runtime.dev_reset_enabled`.

#### `POST /organizations`

Creates a tenant organization.

Payload:

```
{
  "name": "Acme Learning",
  "org_id": "optional"
}
```

#### `POST /users`

Creates a user under an organization.

Payload:

```
{
  "org_id": "org_acme",
  "email": "learner@example.com",
  "name": "optional",
  "user_id": "optional"
}
```

#### `PUT /concepts`

Creates or updates a tenant-owned learning content concept. This is the B2B content layer used by review queues and daily sessions.

Payload:

```
{
  "org_id": "org_acme",
  "concept_id": "support_001",
  "title": "Handle billing escalation",
  "prompt": "What steps should an agent follow?",
  "answer": "Confirm identity, review context, explain options, and document resolution.",
  "explanation": "Reinforces the approved support escalation playbook.",
  "tags": ["support", "billing"],
  "difficulty": 1.5,
  "source": "support_playbook",
  "active": true
}
```

#### `GET /concepts`

Query params:

- `org_id` (required)
- `active` (bool, optional)
- `offset` (int, default 0)
- `limit` (int, default 100, max 500)

#### `GET /concepts/{concept_id}`

Query params:

- `org_id` (required)

#### `POST /record-attempt`

Writes per-user concept attempt and updates memory state.

Payload:

```
{
  "user_id": "usr_001",
  "concept_id": "cardiology_001",
  "correct": true,
  "response_ms": 850,
  "attempted_at": 1700000000.0
}
```

#### `GET /review-queue`

Query params:

- `user_id` (required)
- `limit` (int, default 50, max 500)

#### `GET /analytics`

Query params:

- `user_id` or `org_id` (one required)
- `window_days` (int)

#### `GET /audit-log`

Returns tenant/user audit events for buyer-trust and enterprise operations.

Query params:

- `org_id` (optional)
- `user_id` (optional)
- `action` (optional)
- `limit` (int, default 100, max 1000)

#### `POST /roi/report`

Builds an executive ROI report from retention analytics plus training economics.

Payload:

```
{
  "org_id": "org_acme",
  "learners": 10000,
  "training_hours_saved_per_learner": 2.5,
  "cost_per_training_hour": 60.0,
  "annual_contract_value": 300000.0,
  "window_days": 30
}
```

#### CSV exports

- `GET /exports/review-queue.csv?user_id=usr_001&limit=20`
- `GET /exports/analytics.csv?org_id=org_acme`
- `GET /exports/audit-log.csv?org_id=org_acme&limit=100`
- `POST /exports/roi-report.csv`

#### `POST /pilot/setup`

Assigns treatment/control cohorts.

Payload:

```
{
  "org_id": "org_acme",
  "name": "Support Pilot",
  "treatment_ratio": 0.5,
  "random_seed": 42,
  "pilot_id": "optional"
}
```

#### `GET /pilot/run`

Query params:

- `pilot_id` (required)

#### `POST /pilot/baseline`

Payload:

```
{
  "pilot_id": "pilot_001",
  "window_days": 30,
  "captured_at": 1700000000.0
}
```

#### `POST /pilot/evaluate`

Evaluates pilot gate thresholds.

Payload:

```
{
  "org_id": "optional",
  "sample_size": 300,
  "onboarding_hours": 25.0,
  "retained_mastery_treatment": 0.80,
  "retained_mastery_control": 0.70,
  "forgetting_velocity_treatment": 0.20,
  "forgetting_velocity_control": 0.35,
  "review_efficiency_treatment": 0.74,
  "review_efficiency_control": 0.62
}
```

#### `GET /pilot/history`

Query params:

- `org_id` (optional)
- `limit` (int, default 50, max 500)
