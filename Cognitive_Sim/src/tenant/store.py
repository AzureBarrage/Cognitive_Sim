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

    def create_org(self, name: str, org_id: Optional[str] = None) -> Dict[str, Any]:
        now = time.time()
        oid = org_id or ("org_" + uuid.uuid4().hex[:16])
        with self._lock:
            self._conn.execute(
                "INSERT INTO organizations(id, name, created_at) VALUES (?, ?, ?)",
                (oid, name, now),
            )
            self._conn.commit()
        return {"org_id": oid, "name": name, "created_at": now}

    def create_user(self, org_id: str, email: str, name: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        now = time.time()
        uid = user_id or ("usr_" + uuid.uuid4().hex[:16])
        with self._lock:
            self._conn.execute(
                "INSERT INTO users(id, org_id, email, name, created_at) VALUES (?, ?, ?, ?, ?)",
                (uid, org_id, email, name, now),
            )
            self._conn.commit()
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

        with self._lock:
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

            return {
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
            queue.append(
                {
                    "concept_id": row["concept_id"],
                    "next_review_at": float(row["next_review_at"]),
                    "is_due": float(row["next_review_at"]) <= reference_time,
                    "retention": float(retention),
                    "risk_score": float(risk),
                    "strength": float(row["strength"]),
                    "stability": float(row["stability"]),
                }
            )
        queue.sort(key=lambda item: ((not item["is_due"]), -item["risk_score"], item["next_review_at"]))
        return queue

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

        control_metrics = self._aggregate_user_metrics(
            [self._compute_user_analytics_at(user_id=uid, window_days=days, reference_time=reference_time) for uid in control_users]
        )
        treatment_metrics = self._aggregate_user_metrics(
            [self._compute_user_analytics_at(user_id=uid, window_days=days, reference_time=reference_time) for uid in treatment_users]
        )
        overall_metrics = self._aggregate_user_metrics(
            [self._compute_user_analytics_at(user_id=uid, window_days=days, reference_time=reference_time) for uid in (control_users + treatment_users)]
        )

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

        metrics = [self.compute_user_analytics(user_id=uid, window_days=window_days) for uid in user_ids]
        aggregated = self._aggregate_user_metrics(metrics)
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
