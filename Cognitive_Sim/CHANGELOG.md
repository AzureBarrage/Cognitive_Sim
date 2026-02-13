# Changelog

## 0.2.0 - Repository audit + P0 implementation

### Added
- New architecture interfaces in `src/core/interfaces.py` for memory, policy, environment, and brain abstractions.
- Pluggable policies in `src/core/policy.py`:
  - Heuristic policy (energy + due memories + entropy)
  - Contextual bandit policy (reward-driven)
- Deterministic + stochastic toy environments in `src/environment/dataset_manager.py`.
- Explicit environment API (`reset`, `step`, `get_next_flashcard`) in `src/environment/simulation_env.py`.
- Global reproducible seeding utility in `src/utils/seed.py`.
- Structured run artifact metrics (`JSONL` + `summary.json`) in `src/utils/metrics.py`.
- API endpoints:
  - `GET /memories` with paging + filters
  - `GET /metrics`
  - guarded `POST /reset`
- New tests:
  - unit: decay curve, scheduler progression, policy behavior, optimizer/network behavior, agent boundaries
  - integration: deterministic short simulation, API contract, scheduler behavior

### Changed
- Rebuilt memory subsystem in `src/core/memory_layer.py`:
  - typed `MemoryRecord` schema
  - numerically stable Ebbinghaus retention computation
  - spaced repetition review updates (SM-2 inspired)
  - forgotten/relearn penalty behavior
  - persistence + listing/filtering + thread-safe locking
- Rebuilt neural core in `src/core/network.py`:
  - device-aware operation
  - explicit `train_step()` / `eval_step()`
  - gradient clipping + configurable loss and optimizer behavior
  - checkpoint save/load support
- Rebuilt agent orchestration in `src/core/agent.py`:
  - policy-driven action selection
  - balanced energy economy hooks
  - consistent reward updates and action metrics
- Rebuilt CLI runtime in `src/main.py`:
  - deterministic run support
  - stronger verify command with invariants
  - integrated checkpointing + artifact summary output
- Rebuilt FastAPI app in `src/api.py`:
  - strict request/response models
  - validation and status codes
  - thread-safe global state with lock
- Config model extensions in `src/config.py` + all YAML env files.
- Packaging/deployment updates:
  - requirements split now populated (`base/dev/test/prod`)
  - Dockerfile moved to multi-stage build
  - docker-compose includes logs mount and explicit uvicorn command

### Fixed
- Removed duplicated class block previously present in memory layer.
- Added fallback behavior when `psutil` is unavailable (`src/performance_monitor.py`).
- Corrected pyproject package discovery for current project layout.
