import json
import math
import random
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


class TenantMemoryStore:
    """SQLite-backed multi-tenant memory and attempt analytics store."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS organizations (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    org_id TEXT NOT NULL,
                    email TEXT NOT NULL,
                    name TEXT,
                    created_at REAL NOT NULL,
                    UNIQUE(org_id, email),
                    FOREIGN KEY(org_id) REFERENCES organizations(id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS concepts (
                    id TEXT NOT NULL,
                    org_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    explanation TEXT,
                    tags TEXT NOT NULL,
                    difficulty REAL NOT NULL,
                    source TEXT,
                    version INTEGER NOT NULL,
                    active INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(org_id, id),
                    FOREIGN KEY(org_id) REFERENCES organizations(id)
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_concepts_org_active ON concepts(org_id, active, updated_at)")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_memory_state (
                    user_id TEXT NOT NULL,
                    concept_id TEXT NOT NULL,
                    strength REAL NOT NULL,
                    stability REAL NOT NULL,
                    last_reviewed REAL NOT NULL,
                    decay_rate REAL NOT NULL,
                    retrieval_difficulty REAL NOT NULL,
                    attempts INTEGER NOT NULL,
                    successes INTEGER NOT NULL,
                    failures INTEGER NOT NULL,
                    next_review_at REAL NOT NULL,
                    forgotten INTEGER NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(user_id, concept_id),
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS attempt_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    concept_id TEXT NOT NULL,
                    correct INTEGER NOT NULL,
                    response_ms REAL,
                    attempted_at REAL NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_attempt_user_time ON attempt_events(user_id, attempted_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_attempt_org_user ON users(org_id, id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mem_user_next ON user_memory_state(user_id, next_review_at)")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    org_id TEXT,
                    user_id TEXT,
                    action TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT,
                    metadata TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(org_id) REFERENCES organizations(id),
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_org_time ON audit_events(org_id, created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_user_time ON audit_events(user_id, created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_action_time ON audit_events(action, created_at)")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS pilot_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    org_id TEXT,
                    evaluated_at REAL NOT NULL,
                    sample_size INTEGER NOT NULL,
                    retained_mastery_lift REAL NOT NULL,
                    forgetting_velocity_reduction REAL NOT NULL,
                    review_efficiency_lift REAL NOT NULL,
                    retained_mastery_z REAL NOT NULL,
                    forgetting_velocity_z REAL NOT NULL,
                    review_efficiency_z REAL NOT NULL,
                    onboarding_hours REAL NOT NULL,
                    go_decision INTEGER NOT NULL,
                    reasons TEXT NOT NULL,
                    FOREIGN KEY(org_id) REFERENCES organizations(id)
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pilot_org_time ON pilot_evaluations(org_id, evaluated_at)")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS pilot_runs (
                    id TEXT PRIMARY KEY,
                    org_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    treatment_ratio REAL NOT NULL,
                    random_seed INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    FOREIGN KEY(org_id) REFERENCES organizations(id)
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pilot_runs_org_time ON pilot_runs(org_id, created_at)")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS pilot_cohort_assignments (
                    pilot_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    cohort TEXT NOT NULL,
                    assigned_at REAL NOT NULL,
                    PRIMARY KEY(pilot_id, user_id),
                    FOREIGN KEY(pilot_id) REFERENCES pilot_runs(id),
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_pilot_cohort_lookup ON pilot_cohort_assignments(pilot_id, cohort)"
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS pilot_baselines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pilot_id TEXT NOT NULL,
                    captured_at REAL NOT NULL,
                    window_days INTEGER NOT NULL,
                    control_users INTEGER NOT NULL,
                    treatment_users INTEGER NOT NULL,
                    control_metrics TEXT NOT NULL,
                    treatment_metrics TEXT NOT NULL,
                    overall_metrics TEXT NOT NULL,
                    FOREIGN KEY(pilot_id) REFERENCES pilot_runs(id)
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pilot_baselines_time ON pilot_baselines(pilot_id, captured_at)")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _safe_retention(strength: float, stability: float, decay_rate: float, elapsed: float) -> float:
        stability = max(1e-6, float(stability))
        exponent = -((float(decay_rate) * max(0.0, float(elapsed))) / stability)
        exponent = max(-700.0, min(700.0, exponent))
        return max(0.0, min(1.0, float(strength) * math.exp(exponent)))

    def _org_exists(self, org_id: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM organizations WHERE id = ?", (org_id,)).fetchone()
        return row is not None

    def _user_exists(self, user_id: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone()
        return row is not None

    def _user_org_id(self, user_id: str) -> Optional[str]:
        row = self._conn.execute("SELECT org_id FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            return None
        return str(row["org_id"])

    def _ensure_org_exists(self, org_id: Optional[str], name: Optional[str] = None, created_at: Optional[float] = None) -> None:
        if not org_id:
            return
        if self._org_exists(org_id):
            return
        now = float(time.time() if created_at is None else created_at)
        self._conn.execute(
            "INSERT INTO organizations(id, name, created_at) VALUES (?, ?, ?)",
            (org_id, name or org_id, now),
        )

    @staticmethod
    def _decode_json_list(raw: Any) -> List[str]:
        if raw is None:
            return []
        try:
            value = json.loads(str(raw))
            if isinstance(value, list):
                return [str(item) for item in value]
        except Exception:
            return []
        return []

    @staticmethod
    def _concept_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "concept_id": str(row["id"]),
            "org_id": str(row["org_id"]),
            "title": str(row["title"]),
            "prompt": str(row["prompt"]),
            "answer": str(row["answer"]),
            "explanation": row["explanation"],
            "tags": TenantMemoryStore._decode_json_list(row["tags"]),
            "difficulty": float(row["difficulty"]),
            "source": row["source"],
            "version": int(row["version"]),
            "active": bool(row["active"]),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    @staticmethod
    def _audit_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        try:
            metadata = json.loads(str(row["metadata"] or "{}"))
        except Exception:
            metadata = {}
        return {
            "id": int(row["id"]),
            "org_id": row["org_id"],
            "user_id": row["user_id"],
            "action": str(row["action"]),
            "entity_type": str(row["entity_type"]),
            "entity_id": row["entity_id"],
            "metadata": metadata,
            "created_at": float(row["created_at"]),
        }

    def log_audit_event(
        self,
        action: str,
        entity_type: str,
        entity_id: Optional[str] = None,
        org_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[float] = None,
    ) -> Dict[str, Any]:
        now = float(time.time() if created_at is None else created_at)
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO audit_events(org_id, user_id, action, entity_type, entity_id, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    org_id,
                    user_id,
                    action,
                    entity_type,
                    entity_id,
                    json.dumps(metadata or {}, sort_keys=True),
                    now,
                ),
            )
            self._conn.commit()
            row = self._conn.execute("SELECT * FROM audit_events WHERE id = ?", (int(cursor.lastrowid),)).fetchone()
        return self._audit_row_to_dict(row)

    def list_audit_events(
        self,
        org_id: Optional[str] = None,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM audit_events"
        clauses: List[str] = []
        params: List[Any] = []
        if org_id:
            clauses.append("org_id = ?")
            params.append(org_id)
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if action:
            clauses.append("action = ?")
            params.append(action)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(int(limit))

        with self._lock:
            rows = self._conn.execute(query, tuple(params)).fetchall()
        return [self._audit_row_to_dict(row) for row in rows]

    def upsert_concept(
        self,
        org_id: str,
        concept_id: str,
        title: str,
        prompt: str,
        answer: str,
        explanation: Optional[str] = None,
        tags: Optional[List[str]] = None,
        difficulty: float = 1.0,
        source: Optional[str] = None,
        active: bool = True,
    ) -> Dict[str, Any]:
        if not concept_id.strip():
            raise ValueError("concept_id_required")
        now = time.time()
        clean_tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
        with self._lock:
            if not self._org_exists(org_id):
                raise ValueError("organization_not_found")
            existing = self._conn.execute(
                "SELECT * FROM concepts WHERE org_id = ? AND id = ?",
                (org_id, concept_id),
            ).fetchone()
            version = int(existing["version"] + 1) if existing is not None else 1
            created_at = float(existing["created_at"]) if existing is not None else now
            self._conn.execute(
                """
                INSERT OR REPLACE INTO concepts(
                    id, org_id, title, prompt, answer, explanation, tags, difficulty,
                    source, version, active, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    concept_id,
                    org_id,
                    title,
                    prompt,
                    answer,
                    explanation,
                    json.dumps(clean_tags),
                    float(difficulty),
                    source,
                    int(version),
                    1 if active else 0,
                    created_at,
                    now,
                ),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM concepts WHERE org_id = ? AND id = ?",
                (org_id, concept_id),
            ).fetchone()
        concept = self._concept_row_to_dict(row)
        self.log_audit_event(
            action="concept_upserted",
            entity_type="concept",
            entity_id=concept_id,
            org_id=org_id,
            metadata={"version": concept["version"], "active": concept["active"], "tags": concept["tags"]},
        )
        return concept

    def get_concept(self, org_id: str, concept_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM concepts WHERE org_id = ? AND id = ?",
                (org_id, concept_id),
            ).fetchone()
        return self._concept_row_to_dict(row) if row is not None else None

    def list_concepts(self, org_id: str, active: Optional[bool] = True, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        query = "SELECT * FROM concepts WHERE org_id = ?"
        params: List[Any] = [org_id]
        if active is not None:
            query += " AND active = ?"
            params.append(1 if active else 0)
        query += " ORDER BY updated_at DESC, id ASC LIMIT ? OFFSET ?"
        params.extend([int(limit), int(offset)])

        with self._lock:
            rows = self._conn.execute(query, tuple(params)).fetchall()
        return [self._concept_row_to_dict(row) for row in rows]

    def _concepts_for_user(self, user_id: str, concept_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not concept_ids:
            return {}
        org_id = self._user_org_id(user_id)
        if org_id is None:
            return {}
        placeholders = ",".join(["?"] * len(concept_ids))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM concepts WHERE org_id = ? AND id IN ({placeholders})",
                tuple([org_id] + concept_ids),
            ).fetchall()
        return {str(row["id"]): self._concept_row_to_dict(row) for row in rows}

    def create_org(self, name: str, org_id: Optional[str] = None) -> Dict[str, Any]:
        now = time.time()
        oid = org_id or ("org_" + uuid.uuid4().hex[:16])
        with self._lock:
            if self._org_exists(oid):
                raise ValueError("organization_already_exists")
            self._conn.execute("INSERT INTO organizations(id, name, created_at) VALUES (?, ?, ?)", (oid, name, now))
            self._conn.commit()
        self.log_audit_event(
            action="organization_created",
            entity_type="organization",
            entity_id=oid,
            org_id=oid,
            metadata={"name": name},
            created_at=now,
        )
        return {"org_id": oid, "name": name, "created_at": now}

    def create_user(self, org_id: str, email: str, name: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        now = time.time()
        uid = user_id or ("usr_" + uuid.uuid4().hex[:16])
        with self._lock:
            if not self._org_exists(org_id):
                raise ValueError("organization_not_found")
            if self._user_exists(uid):
                raise ValueError("user_already_exists")
            existing_email = self._conn.execute(
                "SELECT 1 FROM users WHERE org_id = ? AND email = ?",
                (org_id, email),
            ).fetchone()
            if existing_email is not None:
                raise ValueError("user_email_already_exists")
            self._conn.execute(
                "INSERT INTO users(id, org_id, email, name, created_at) VALUES (?, ?, ?, ?, ?)",
                (uid, org_id, email, name, now),
            )
            self._conn.commit()
        self.log_audit_event(
            action="user_created",
            entity_type="user",
            entity_id=uid,
            org_id=org_id,
            user_id=uid,
            metadata={"email": email, "name": name},
            created_at=now,
        )
        return {"user_id": uid, "org_id": org_id, "email": email, "name": name, "created_at": now}

    def _get_state(self, user_id: str, concept_id: str) -> Optional[sqlite3.Row]:
        row = self._conn.execute(
            "SELECT * FROM user_memory_state WHERE user_id = ? AND concept_id = ?",
            (user_id, concept_id),
        ).fetchone()
        return row

    def record_attempt(
        self,
        user_id: str,
        concept_id: str,
        correct: bool,
        response_ms: Optional[float] = None,
        decay_rate: float = 0.1,
        retrieval_difficulty: float = 2.5,
        attempted_at: Optional[float] = None,
    ) -> Dict[str, Any]:
        now = float(time.time() if attempted_at is None else attempted_at)
        audit_org_id: Optional[str] = None

        with self._lock:
            if not self._user_exists(user_id):
                raise ValueError("user_not_found")
            audit_org_id = self._user_org_id(user_id)
            self._conn.execute(
                "INSERT INTO attempt_events(user_id, concept_id, correct, response_ms, attempted_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, concept_id, 1 if correct else 0, response_ms, now),
            )

            state = self._get_state(user_id, concept_id)
            if state is None:
                strength = 1.0
                stability = 1.0
                attempts = 0
                successes = 0
                failures = 0
                last_reviewed = now
                interval = 30.0
                difficulty = float(retrieval_difficulty)
            else:
                strength = float(state["strength"])
                stability = float(state["stability"])
                attempts = int(state["attempts"])
                successes = int(state["successes"])
                failures = int(state["failures"])
                last_reviewed = float(state["last_reviewed"])
                interval = max(5.0, float(state["next_review_at"]) - float(last_reviewed))
                difficulty = float(state["retrieval_difficulty"])
                decay_rate = float(state["decay_rate"])

            attempts += 1
            elapsed = max(0.0, now - last_reviewed)
            retention = self._safe_retention(strength, stability, decay_rate, elapsed)

            if correct:
                successes += 1
                quality = int(max(0, min(5, round(retention * 5))))
                ef_delta = 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
                difficulty = max(1.3, min(3.0, difficulty + ef_delta))
                strength = min(1.0, strength + 0.06 + quality / 100.0)
                stability = max(0.1, stability * (1.0 + 0.12 * max(1, quality)))
                interval = max(30.0, interval * max(1.3, difficulty))
                forgotten = 0
            else:
                failures += 1
                difficulty = max(1.3, difficulty - 0.2)
                strength = max(0.1, strength * 0.75)
                stability = max(0.1, stability / 1.3)
                interval = max(10.0, interval * 0.5)
                forgotten = 1

            next_review_at = now + interval

            self._conn.execute(
                """
                INSERT OR REPLACE INTO user_memory_state(
                    user_id, concept_id, strength, stability, last_reviewed, decay_rate,
                    retrieval_difficulty, attempts, successes, failures, next_review_at,
                    forgotten, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    concept_id,
                    float(strength),
                    float(stability),
                    now,
                    float(decay_rate),
                    float(difficulty),
                    int(attempts),
                    int(successes),
                    int(failures),
                    float(next_review_at),
                    int(forgotten),
                    now,
                ),
            )
            self._conn.commit()

            risk = 1.0 - self._safe_retention(strength, stability, decay_rate, 0.0)
            risk = min(1.0, risk + (0.2 if forgotten else 0.0))

            result = {
                "user_id": user_id,
                "concept_id": concept_id,
                "correct": bool(correct),
                "attempts": attempts,
                "successes": successes,
                "failures": failures,
                "strength": float(strength),
                "stability": float(stability),
                "decay_rate": float(decay_rate),
                "retrieval_difficulty": float(difficulty),
                "next_review_at": float(next_review_at),
                "forgotten": bool(forgotten),
                "risk_score": float(risk),
            }

        self.log_audit_event(
            action="attempt_recorded",
            entity_type="attempt",
            entity_id=concept_id,
            org_id=audit_org_id,
            user_id=user_id,
            metadata={
                "concept_id": concept_id,
                "correct": bool(correct),
                "response_ms": response_ms,
                "attempts": attempts,
                "risk_score": float(risk),
            },
            created_at=now,
        )
        return result

    def get_review_queue(self, user_id: str, limit: int = 50, now: Optional[float] = None) -> List[Dict[str, Any]]:
        reference_time = float(time.time() if now is None else now)
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM user_memory_state
                WHERE user_id = ?
                ORDER BY next_review_at ASC
                LIMIT ?
                """,
                (user_id, int(limit)),
            ).fetchall()

        queue: List[Dict[str, Any]] = []
        for row in rows:
            elapsed = max(0.0, reference_time - float(row["last_reviewed"]))
            retention = self._safe_retention(
                float(row["strength"]),
                float(row["stability"]),
                float(row["decay_rate"]),
                elapsed,
            )
            risk = min(1.0, (1.0 - retention) + (0.2 if int(row["forgotten"]) else 0.0))
            is_due = float(row["next_review_at"]) <= reference_time
            if is_due:
                reason_code = "review_due"
            elif int(row["forgotten"]):
                reason_code = "prior_miss_reinforcement"
            elif retention < 0.6:
                reason_code = "retention_risk"
            else:
                reason_code = "scheduled_reinforcement"
            queue.append(
                {
                    "concept_id": row["concept_id"],
                    "next_review_at": float(row["next_review_at"]),
                    "is_due": is_due,
                    "retention": float(retention),
                    "risk_score": float(risk),
                    "strength": float(row["strength"]),
                    "stability": float(row["stability"]),
                    "reason_code": reason_code,
                    "reason": self._review_reason(reason_code, retention, risk),
                }
            )
        queue.sort(key=lambda item: ((not item["is_due"]), -item["risk_score"], item["next_review_at"]))
        concepts = self._concepts_for_user(user_id, [str(item["concept_id"]) for item in queue])
        for item in queue:
            item["concept"] = concepts.get(str(item["concept_id"]))
        return queue

    @staticmethod
    def _review_reason(reason_code: str, retention: float, risk: float) -> str:
        if reason_code == "review_due":
            return "This concept is due now based on its spaced-review interval."
        if reason_code == "prior_miss_reinforcement":
            return "This concept was previously missed and should be reinforced."
        if reason_code == "retention_risk":
            return f"Estimated retention is below the target range ({retention:.0%})."
        return f"Scheduled reinforcement keeps the risk score controlled ({risk:.0%})."

    def _user_ids_for_org(self, org_id: str) -> List[str]:
        with self._lock:
            rows = self._conn.execute("SELECT id FROM users WHERE org_id = ?", (org_id,)).fetchall()
        return [str(row["id"]) for row in rows]

    @staticmethod
    def _aggregate_user_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not metrics:
            return {
                "users": 0,
                "attempts": 0,
                "retention_percentage": 0.0,
                "forgetting_velocity": 0.0,
                "unstable_concept_ratio": 0.0,
                "review_efficiency_score": 0.0,
                "avg_concept_half_life_hours": 0.0,
                "at_risk_concepts": 0,
            }

        count = len(metrics)
        return {
            "users": int(count),
            "attempts": int(sum(int(m.get("attempts", 0)) for m in metrics)),
            "retention_percentage": float(sum(float(m.get("retention_percentage", 0.0)) for m in metrics) / count),
            "forgetting_velocity": float(sum(float(m.get("forgetting_velocity", 0.0)) for m in metrics) / count),
            "unstable_concept_ratio": float(sum(float(m.get("unstable_concept_ratio", 0.0)) for m in metrics) / count),
            "review_efficiency_score": float(sum(float(m.get("review_efficiency_score", 0.0)) for m in metrics) / count),
            "avg_concept_half_life_hours": float(sum(float(m.get("avg_concept_half_life_hours", 0.0)) for m in metrics) / count),
            "at_risk_concepts": int(sum(int(m.get("at_risk_concepts", 0)) for m in metrics)),
        }

    def _compute_user_metrics_map(
        self,
        user_ids: List[str],
        window_days: int,
        reference_time: float,
    ) -> Dict[str, Dict[str, Any]]:
        return {
            user_id: self._compute_user_analytics_at(
                user_id=user_id,
                window_days=window_days,
                reference_time=reference_time,
            )
            for user_id in user_ids
        }

    def setup_pilot(
        self,
        org_id: str,
        name: str,
        treatment_ratio: float = 0.5,
        random_seed: int = 42,
        pilot_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        ratio = float(treatment_ratio)
        if ratio <= 0.0 or ratio >= 1.0:
            raise ValueError("treatment_ratio must be > 0 and < 1")

        with self._lock:
            org = self._conn.execute("SELECT id FROM organizations WHERE id = ?", (org_id,)).fetchone()
            if org is None:
                raise ValueError("organization_not_found")

        users = self._user_ids_for_org(org_id)
        if len(users) < 2:
            raise ValueError("pilot_setup_requires_at_least_two_users")

        ordered_users = list(users)
        random.Random(int(random_seed)).shuffle(ordered_users)

        treatment_count = int(round(len(ordered_users) * ratio))
        treatment_count = max(1, min(len(ordered_users) - 1, treatment_count))
        treatment_users = ordered_users[:treatment_count]
        control_users = ordered_users[treatment_count:]

        pid = pilot_id or ("pilot_" + uuid.uuid4().hex[:16])
        now = time.time()

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO pilot_runs(id, org_id, name, created_at, treatment_ratio, random_seed, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (pid, org_id, name, now, ratio, int(random_seed), "configured"),
            )
            for uid in control_users:
                self._conn.execute(
                    "INSERT INTO pilot_cohort_assignments(pilot_id, user_id, cohort, assigned_at) VALUES (?, ?, ?, ?)",
                    (pid, uid, "control", now),
                )
            for uid in treatment_users:
                self._conn.execute(
                    "INSERT INTO pilot_cohort_assignments(pilot_id, user_id, cohort, assigned_at) VALUES (?, ?, ?, ?)",
                    (pid, uid, "treatment", now),
            )
            self._conn.commit()

        self.log_audit_event(
            action="pilot_configured",
            entity_type="pilot_run",
            entity_id=pid,
            org_id=org_id,
            metadata={
                "name": name,
                "treatment_ratio": ratio,
                "control_count": len(control_users),
                "treatment_count": len(treatment_users),
            },
            created_at=now,
        )

        return {
            "pilot_id": pid,
            "org_id": org_id,
            "name": name,
            "status": "configured",
            "treatment_ratio": ratio,
            "random_seed": int(random_seed),
            "control_count": int(len(control_users)),
            "treatment_count": int(len(treatment_users)),
            "control_user_ids": control_users,
            "treatment_user_ids": treatment_users,
            "created_at": now,
        }

    def get_pilot_run(self, pilot_id: str) -> Dict[str, Any]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM pilot_runs WHERE id = ?", (pilot_id,)).fetchone()
            if row is None:
                raise ValueError("pilot_not_found")

            assignments = self._conn.execute(
                "SELECT user_id, cohort FROM pilot_cohort_assignments WHERE pilot_id = ? ORDER BY user_id ASC",
                (pilot_id,),
            ).fetchall()
            baseline_row = self._conn.execute(
                "SELECT * FROM pilot_baselines WHERE pilot_id = ? ORDER BY captured_at DESC LIMIT 1",
                (pilot_id,),
            ).fetchone()

        control_user_ids = [str(item["user_id"]) for item in assignments if str(item["cohort"]) == "control"]
        treatment_user_ids = [str(item["user_id"]) for item in assignments if str(item["cohort"]) == "treatment"]

        latest_baseline: Optional[Dict[str, Any]] = None
        if baseline_row is not None:
            latest_baseline = {
                "baseline_id": int(baseline_row["id"]),
                "captured_at": float(baseline_row["captured_at"]),
                "window_days": int(baseline_row["window_days"]),
                "control_users": int(baseline_row["control_users"]),
                "treatment_users": int(baseline_row["treatment_users"]),
                "control_metrics": json.loads(str(baseline_row["control_metrics"])),
                "treatment_metrics": json.loads(str(baseline_row["treatment_metrics"])),
                "overall_metrics": json.loads(str(baseline_row["overall_metrics"])),
            }

        return {
            "pilot_id": str(row["id"]),
            "org_id": str(row["org_id"]),
            "name": str(row["name"]),
            "status": str(row["status"]),
            "treatment_ratio": float(row["treatment_ratio"]),
            "random_seed": int(row["random_seed"]),
            "created_at": float(row["created_at"]),
            "control_count": int(len(control_user_ids)),
            "treatment_count": int(len(treatment_user_ids)),
            "control_user_ids": control_user_ids,
            "treatment_user_ids": treatment_user_ids,
            "latest_baseline": latest_baseline,
        }

    def capture_pilot_baseline(
        self,
        pilot_id: str,
        window_days: int = 30,
        captured_at: Optional[float] = None,
    ) -> Dict[str, Any]:
        reference_time = float(time.time() if captured_at is None else captured_at)
        days = max(1, int(window_days))

        run = self.get_pilot_run(pilot_id)
        control_users = list(run["control_user_ids"])
        treatment_users = list(run["treatment_user_ids"])

        if not control_users or not treatment_users:
            raise ValueError("pilot_requires_control_and_treatment_users")

        all_user_ids = list(dict.fromkeys(control_users + treatment_users))
        metrics_by_user = self._compute_user_metrics_map(
            user_ids=all_user_ids,
            window_days=days,
            reference_time=reference_time,
        )
        control_metrics = self._aggregate_user_metrics([metrics_by_user[uid] for uid in control_users])
        treatment_metrics = self._aggregate_user_metrics([metrics_by_user[uid] for uid in treatment_users])
        overall_metrics = self._aggregate_user_metrics([metrics_by_user[uid] for uid in all_user_ids])

        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO pilot_baselines(
                    pilot_id, captured_at, window_days, control_users, treatment_users,
                    control_metrics, treatment_metrics, overall_metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pilot_id,
                    reference_time,
                    days,
                    int(len(control_users)),
                    int(len(treatment_users)),
                    json.dumps(control_metrics),
                    json.dumps(treatment_metrics),
                    json.dumps(overall_metrics),
                ),
            )
            self._conn.commit()
            baseline_id = int(cursor.lastrowid)

        self.log_audit_event(
            action="pilot_baseline_captured",
            entity_type="pilot_baseline",
            entity_id=str(baseline_id),
            org_id=str(run["org_id"]),
            metadata={
                "pilot_id": pilot_id,
                "window_days": days,
                "control_users": len(control_users),
                "treatment_users": len(treatment_users),
            },
            created_at=reference_time,
        )

        return {
            "baseline_id": baseline_id,
            "pilot_id": pilot_id,
            "captured_at": reference_time,
            "window_days": days,
            "control_metrics": control_metrics,
            "treatment_metrics": treatment_metrics,
            "overall_metrics": overall_metrics,
        }

    def _compute_user_analytics_at(self, user_id: str, window_days: int, reference_time: float) -> Dict[str, Any]:
        since = float(reference_time) - (float(window_days) * 86400.0)
        with self._lock:
            attempts_row = self._conn.execute(
                """
                SELECT COUNT(*) as total,
                       SUM(correct) as successes
                FROM attempt_events
                WHERE user_id = ? AND attempted_at >= ? AND attempted_at <= ?
                """,
                (user_id, since, float(reference_time)),
            ).fetchone()
            states = self._conn.execute(
                "SELECT * FROM user_memory_state WHERE user_id = ?",
                (user_id,),
            ).fetchall()

        total = int(attempts_row["total"] or 0)
        successes = int(attempts_row["successes"] or 0)
        failures = max(0, total - successes)

        if not states:
            return {
                "user_id": user_id,
                "attempts": total,
                "retention_percentage": 0.0,
                "forgetting_velocity": 0.0,
                "unstable_concept_ratio": 0.0,
                "review_efficiency_score": 0.0,
                "avg_concept_half_life_hours": 0.0,
                "at_risk_concepts": 0,
            }

        retentions: List[float] = []
        half_lives: List[float] = []
        at_risk = 0
        unstable = 0
        for row in states:
            stability = float(row["stability"])
            decay_rate = float(row["decay_rate"])
            elapsed = float(reference_time) - float(row["last_reviewed"])
            retention = self._safe_retention(float(row["strength"]), stability, decay_rate, elapsed)
            retentions.append(retention)
            if retention < 0.4:
                at_risk += 1
            if retention < 0.6:
                unstable += 1
            half_life = (math.log(2.0) * max(1e-6, stability)) / max(1e-6, decay_rate)
            half_lives.append(half_life / 3600.0)

        retention_percentage = (successes / total) if total else 0.0
        forgetting_velocity = failures / max(1.0, float(window_days))
        review_efficiency = (successes / max(1, total)) * (1.0 - (at_risk / max(1, len(states))))

        return {
            "user_id": user_id,
            "attempts": total,
            "retention_percentage": float(retention_percentage),
            "forgetting_velocity": float(forgetting_velocity),
            "unstable_concept_ratio": float(unstable / max(1, len(states))),
            "review_efficiency_score": float(review_efficiency),
            "avg_concept_half_life_hours": float(sum(half_lives) / max(1, len(half_lives))),
            "at_risk_concepts": int(at_risk),
        }

    def compute_user_analytics(self, user_id: str, window_days: int = 30) -> Dict[str, Any]:
        return self._compute_user_analytics_at(user_id=user_id, window_days=window_days, reference_time=time.time())

    def compute_org_analytics(self, org_id: str, window_days: int = 30) -> Dict[str, Any]:
        user_ids = self._user_ids_for_org(org_id)
        if not user_ids:
            return {
                "org_id": org_id,
                "users": 0,
                "retention_percentage": 0.0,
                "forgetting_velocity": 0.0,
                "unstable_concept_ratio": 0.0,
                "review_efficiency_score": 0.0,
                "avg_concept_half_life_hours": 0.0,
                "at_risk_concepts": 0,
            }

        reference_time = time.time()
        metrics_by_user = self._compute_user_metrics_map(
            user_ids=user_ids,
            window_days=window_days,
            reference_time=reference_time,
        )
        aggregated = self._aggregate_user_metrics(list(metrics_by_user.values()))
        return {
            "org_id": org_id,
            "users": int(aggregated["users"]),
            "retention_percentage": float(aggregated["retention_percentage"]),
            "forgetting_velocity": float(aggregated["forgetting_velocity"]),
            "unstable_concept_ratio": float(aggregated["unstable_concept_ratio"]),
            "review_efficiency_score": float(aggregated["review_efficiency_score"]),
            "avg_concept_half_life_hours": float(aggregated["avg_concept_half_life_hours"]),
            "at_risk_concepts": int(aggregated["at_risk_concepts"]),
        }

    def build_roi_report(
        self,
        org_id: str,
        learners: int,
        training_hours_saved_per_learner: float,
        cost_per_training_hour: float,
        annual_contract_value: float,
        window_days: int = 30,
    ) -> Dict[str, Any]:
        if not self._org_exists(org_id):
            raise ValueError("organization_not_found")
        learner_count = max(0, int(learners))
        hours_saved = max(0.0, float(training_hours_saved_per_learner))
        hourly_cost = max(0.0, float(cost_per_training_hour))
        contract_value = max(0.0, float(annual_contract_value))
        gross_savings = learner_count * hours_saved * hourly_cost
        net_savings = gross_savings - contract_value
        roi_multiple = (gross_savings / contract_value) if contract_value > 0.0 else 0.0
        payback_months = (12.0 / roi_multiple) if roi_multiple > 0.0 else 0.0
        analytics = self.compute_org_analytics(org_id=org_id, window_days=window_days)
        report = {
            "org_id": org_id,
            "generated_at": time.time(),
            "window_days": int(window_days),
            "inputs": {
                "learners": learner_count,
                "training_hours_saved_per_learner": hours_saved,
                "cost_per_training_hour": hourly_cost,
                "annual_contract_value": contract_value,
            },
            "analytics": analytics,
            "gross_savings": float(gross_savings),
            "net_savings": float(net_savings),
            "roi_multiple": float(roi_multiple),
            "payback_months": float(payback_months),
            "executive_summary": (
                f"Estimated annual savings are {gross_savings:,.2f} against an annual contract value of "
                f"{contract_value:,.2f}, producing an ROI multiple of {roi_multiple:.2f}x."
            ),
        }
        self.log_audit_event(
            action="roi_report_generated",
            entity_type="roi_report",
            entity_id=org_id,
            org_id=org_id,
            metadata={
                "learners": learner_count,
                "gross_savings": gross_savings,
                "annual_contract_value": contract_value,
                "roi_multiple": roi_multiple,
            },
        )
        return report

    @staticmethod
    def _z_for_diff(a: float, b: float, n: int) -> float:
        sample = max(1, int(n))
        variance = (max(1e-9, abs(a)) + max(1e-9, abs(b))) / sample
        std = math.sqrt(max(1e-12, variance))
        return (float(a) - float(b)) / std

    def evaluate_pilot(
        self,
        retained_mastery_treatment: float,
        retained_mastery_control: float,
        forgetting_velocity_treatment: float,
        forgetting_velocity_control: float,
        review_efficiency_treatment: float,
        review_efficiency_control: float,
        sample_size: int,
        min_retained_mastery_lift: float,
        min_forgetting_velocity_reduction: float,
        min_review_efficiency_lift: float,
        confidence_z_threshold: float,
        min_sample_size: int,
        onboarding_hours: float,
        max_onboarding_hours: float,
        org_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        lift_retained = float(retained_mastery_treatment - retained_mastery_control)
        reduction_forgetting = float(forgetting_velocity_control - forgetting_velocity_treatment)
        lift_efficiency = float(review_efficiency_treatment - review_efficiency_control)

        z_retained = self._z_for_diff(retained_mastery_treatment, retained_mastery_control, sample_size)
        z_forgetting = self._z_for_diff(forgetting_velocity_control, forgetting_velocity_treatment, sample_size)
        z_efficiency = self._z_for_diff(review_efficiency_treatment, review_efficiency_control, sample_size)

        reasons: List[str] = []
        if int(sample_size) < int(min_sample_size):
            reasons.append("sample_size_below_minimum")
        if lift_retained < float(min_retained_mastery_lift):
            reasons.append("retained_mastery_lift_below_threshold")
        if reduction_forgetting < float(min_forgetting_velocity_reduction):
            reasons.append("forgetting_velocity_reduction_below_threshold")
        if lift_efficiency < float(min_review_efficiency_lift):
            reasons.append("review_efficiency_lift_below_threshold")
        if z_retained < float(confidence_z_threshold):
            reasons.append("retained_mastery_confidence_below_threshold")
        if z_forgetting < float(confidence_z_threshold):
            reasons.append("forgetting_velocity_confidence_below_threshold")
        if z_efficiency < float(confidence_z_threshold):
            reasons.append("review_efficiency_confidence_below_threshold")
        if float(onboarding_hours) > float(max_onboarding_hours):
            reasons.append("onboarding_hours_above_limit")

        go = 1 if not reasons else 0
        now = time.time()
        reasons_str = ",".join(reasons)
        with self._lock:
            self._ensure_org_exists(org_id, name=f"Pilot {org_id}" if org_id else None, created_at=now)
            self._conn.execute(
                """
                INSERT INTO pilot_evaluations(
                    org_id, evaluated_at, sample_size,
                    retained_mastery_lift, forgetting_velocity_reduction, review_efficiency_lift,
                    retained_mastery_z, forgetting_velocity_z, review_efficiency_z,
                    onboarding_hours, go_decision, reasons
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    org_id,
                    now,
                    int(sample_size),
                    lift_retained,
                    reduction_forgetting,
                    lift_efficiency,
                    float(z_retained),
                    float(z_forgetting),
                    float(z_efficiency),
                    float(onboarding_hours),
                    int(go),
                    reasons_str,
                ),
            )
            self._conn.commit()

        self.log_audit_event(
            action="pilot_evaluated",
            entity_type="pilot_evaluation",
            entity_id=org_id,
            org_id=org_id,
            metadata={
                "sample_size": int(sample_size),
                "go_decision": bool(go),
                "retained_mastery_lift": lift_retained,
                "forgetting_velocity_reduction": reduction_forgetting,
                "review_efficiency_lift": lift_efficiency,
                "reasons": reasons,
            },
            created_at=now,
        )

        return {
            "org_id": org_id,
            "evaluated_at": now,
            "sample_size": int(sample_size),
            "retained_mastery_lift": lift_retained,
            "forgetting_velocity_reduction": reduction_forgetting,
            "review_efficiency_lift": lift_efficiency,
            "retained_mastery_z": float(z_retained),
            "forgetting_velocity_z": float(z_forgetting),
            "review_efficiency_z": float(z_efficiency),
            "onboarding_hours": float(onboarding_hours),
            "go_decision": bool(go),
            "reasons": reasons,
        }

    def list_pilot_evaluations(self, org_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        query = "SELECT * FROM pilot_evaluations"
        params: List[Any] = []
        if org_id:
            query += " WHERE org_id = ?"
            params.append(org_id)
        query += " ORDER BY evaluated_at DESC LIMIT ?"
        params.append(int(limit))

        with self._lock:
            rows = self._conn.execute(query, tuple(params)).fetchall()

        output: List[Dict[str, Any]] = []
        for row in rows:
            reasons_str = str(row["reasons"] or "")
            output.append(
                {
                    "id": int(row["id"]),
                    "org_id": row["org_id"],
                    "evaluated_at": float(row["evaluated_at"]),
                    "sample_size": int(row["sample_size"]),
                    "retained_mastery_lift": float(row["retained_mastery_lift"]),
                    "forgetting_velocity_reduction": float(row["forgetting_velocity_reduction"]),
                    "review_efficiency_lift": float(row["review_efficiency_lift"]),
                    "retained_mastery_z": float(row["retained_mastery_z"]),
                    "forgetting_velocity_z": float(row["forgetting_velocity_z"]),
                    "review_efficiency_z": float(row["review_efficiency_z"]),
                    "onboarding_hours": float(row["onboarding_hours"]),
                    "go_decision": bool(row["go_decision"]),
                    "reasons": [item for item in reasons_str.split(",") if item],
                }
            )
        return output
