import heapq
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.config import MemoryConfig
from src.utils.logger import logger


@dataclass
class MemoryRecord:
    key: str
    created_at: float
    last_reviewed: float
    last_access: float
    strength: float
    stability: float
    decay_rate: float
    retrieval_difficulty: float
    interval_seconds: float
    repetitions: int
    lapses: int
    next_review_at: float
    forgotten: bool
    access_count: int
    data_path: str
    embedding: Optional[List[float]] = None


class MemoryLayer:
    """Persistent memory store with Ebbinghaus decay and spaced repetition scheduling."""

    def __init__(self, config: MemoryConfig):
        self.initial_retention = float(config.initial_retention)
        self.decay_rate = float(config.decay_rate)
        self.stability_threshold = float(config.stability_threshold)
        self.review_threshold = float(config.review_threshold)
        self.recall_failure_retention = float(config.recall_failure_retention)
        self.min_stability = float(config.min_stability)
        self.default_difficulty = float(config.default_difficulty)
        self.initial_interval_seconds = float(config.initial_interval_seconds)
        self.relearn_penalty_factor = float(config.relearn_penalty_factor)

        self.store_dir = Path(config.store_dir)
        self.index_path = Path(config.index_path)
        self.eager_load = bool(config.eager_load)
        self.legacy_metadata_path = Path("data/memory_store.json")

        self.memories: Dict[str, MemoryRecord] = {}
        self._payload_cache: Dict[str, Any] = {}

        self._review_heap: List[Tuple[float, int, str]] = []
        self._scheduled_due: Dict[str, float] = {}
        self._heap_seq = 0
        self._lock = RLock()

    @staticmethod
    def _cpuify(obj: Any) -> Any:
        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: MemoryLayer._cpuify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            converted = [MemoryLayer._cpuify(v) for v in obj]
            return converted if isinstance(obj, list) else tuple(converted)
        return obj

    def _memory_blob_path(self, memory_id: str) -> Path:
        return self.store_dir / (memory_id + ".pt")

    def _calculate_retention(self, record: MemoryRecord, now: Optional[float] = None) -> float:
        reference_time = time.time() if now is None else float(now)
        elapsed = max(0.0, reference_time - float(record.last_reviewed))
        stability = max(self.min_stability, float(record.stability))
        exponent = -((float(record.decay_rate) * elapsed) / stability)
        exponent = max(-700.0, min(700.0, exponent))
        return float(math.exp(exponent)) * float(record.strength)

    def _next_review_at(self, record: MemoryRecord, threshold: Optional[float] = None) -> float:
        threshold_value = self.review_threshold if threshold is None else float(threshold)
        threshold_value = float(np.clip(threshold_value, 1e-6, 0.999999))
        base = max(self.min_stability, float(record.stability))
        if float(record.decay_rate) <= 0:
            return float("inf")
        due_in = (-math.log(threshold_value) * base) / float(record.decay_rate)
        due_in = max(float(record.interval_seconds), due_in)
        return float(record.last_reviewed + due_in)

    def advance_time(self, delta_seconds: float) -> None:
        delta = float(delta_seconds)
        if delta <= 0:
            return
        with self._lock:
            for record in self.memories.values():
                record.last_reviewed -= delta
                record.last_access -= delta
                record.created_at -= delta
                record.next_review_at -= delta
                self._scheduled_due[record.key] = record.next_review_at
            self._review_heap = [
                (due_at - delta, seq, mem_id)
                for due_at, seq, mem_id in self._review_heap
            ]
            heapq.heapify(self._review_heap)

    def _schedule_review(self, memory_id: str, threshold: Optional[float] = None) -> None:
        record = self.memories.get(memory_id)
        if record is None:
            return

        due_at = self._next_review_at(record, threshold=threshold)
        record.next_review_at = due_at
        self._scheduled_due[memory_id] = due_at
        self._heap_seq += 1
        heapq.heappush(self._review_heap, (due_at, self._heap_seq, memory_id))

    def _cleanup_heap(self) -> None:
        while self._review_heap:
            due_at, _, mem_id = self._review_heap[0]
            if mem_id in self.memories and self._scheduled_due.get(mem_id) == due_at:
                return
            heapq.heappop(self._review_heap)

    def _serialize_record(self, record: MemoryRecord) -> Dict[str, Any]:
        return asdict(record)

    def _deserialize_record(self, memory_id: str, payload: Dict[str, Any]) -> MemoryRecord:
        return MemoryRecord(
            key=payload.get("key", memory_id),
            created_at=float(payload.get("created_at", time.time())),
            last_reviewed=float(payload.get("last_reviewed", payload.get("last_access", time.time()))),
            last_access=float(payload.get("last_access", time.time())),
            strength=float(payload.get("strength", 1.0)),
            stability=max(self.min_stability, float(payload.get("stability", 1.0))),
            decay_rate=float(payload.get("decay_rate", self.decay_rate)),
            retrieval_difficulty=float(payload.get("retrieval_difficulty", self.default_difficulty)),
            interval_seconds=float(payload.get("interval_seconds", self.initial_interval_seconds)),
            repetitions=int(payload.get("repetitions", 0)),
            lapses=int(payload.get("lapses", 0)),
            next_review_at=float(payload.get("next_review_at", time.time())),
            forgotten=bool(payload.get("forgotten", False)),
            access_count=int(payload.get("access_count", 0)),
            data_path=str(payload.get("data_path", self._memory_blob_path(memory_id))),
            embedding=payload.get("embedding"),
        )

    def add_memory(
        self,
        memory_id: str,
        data: Any,
        initial_stability: float = 1.0,
        embedding: Optional[List[float]] = None,
    ) -> None:
        now = time.time()
        with self._lock:
            record = MemoryRecord(
                key=memory_id,
                created_at=now,
                last_reviewed=now,
                last_access=now,
                strength=1.0,
                stability=max(self.min_stability, float(initial_stability)),
                decay_rate=self.decay_rate,
                retrieval_difficulty=self.default_difficulty,
                interval_seconds=self.initial_interval_seconds,
                repetitions=0,
                lapses=0,
                next_review_at=now + self.initial_interval_seconds,
                forgotten=False,
                access_count=1,
                data_path=str(self._memory_blob_path(memory_id)),
                embedding=embedding,
            )
            self.memories[memory_id] = record
            self._payload_cache[memory_id] = data
            self._schedule_review(memory_id)

    def _load_payload(self, memory_id: str) -> Optional[Any]:
        if memory_id in self._payload_cache:
            return self._payload_cache[memory_id]
        record = self.memories.get(memory_id)
        if record is None:
            return None
        data_path = Path(record.data_path)
        if not data_path.exists():
            return None
        try:
            payload = torch.load(data_path, map_location="cpu")
        except Exception as exc:
            logger.warning("Failed to load memory payload for %s from %s: %s", memory_id, data_path, exc)
            return None
        self._payload_cache[memory_id] = payload
        return payload

    def retrieve_memory(self, memory_id: str, reinforce: bool = True) -> Optional[Any]:
        with self._lock:
            record = self.memories.get(memory_id)
            if record is None:
                return None

            retention = self._calculate_retention(record)
            if retention < self.recall_failure_retention:
                if reinforce:
                    self.review_memory(memory_id, success=False)
                return None

            payload = self._load_payload(memory_id)
            if payload is None:
                return None

            if reinforce:
                self.review_memory(memory_id, success=True)
            return payload

    def review_memory(self, memory_id: str, success: bool) -> None:
        with self._lock:
            record = self.memories.get(memory_id)
            if record is None:
                return

            now = time.time()
            retention = self._calculate_retention(record, now=now)
            quality = int(np.clip(round(retention * 5.0), 0, 5)) if success else 0

            if success:
                record.repetitions += 1
                record.forgotten = False
                if record.repetitions == 1:
                    interval = self.initial_interval_seconds
                elif record.repetitions == 2:
                    interval = self.initial_interval_seconds * 6.0
                else:
                    interval = record.interval_seconds * max(1.3, record.retrieval_difficulty)

                ef_delta = 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
                record.retrieval_difficulty = float(max(1.3, min(3.0, record.retrieval_difficulty + ef_delta)))
                record.interval_seconds = max(self.initial_interval_seconds, interval)
                record.strength = float(min(1.0, record.strength + 0.06 + (quality / 100.0)))
                record.stability = float(max(self.min_stability, record.stability * (1.0 + 0.12 * max(1, quality))))
                record.last_reviewed = now
                record.last_access = now
                record.access_count += 1
            else:
                record.lapses += 1
                record.repetitions = 0
                record.forgotten = True
                record.interval_seconds = max(1.0, self.initial_interval_seconds / 2.0)
                record.retrieval_difficulty = float(max(1.3, record.retrieval_difficulty - 0.2))
                record.strength = float(max(0.1, record.strength * 0.7))
                record.stability = float(max(self.min_stability, record.stability / self.relearn_penalty_factor))
                record.last_access = now

            self._schedule_review(memory_id)

    def has_at_risk_memory(self, threshold: Optional[float] = None) -> bool:
        threshold_value = self.review_threshold if threshold is None else float(threshold)
        with self._lock:
            self._cleanup_heap()
            now = time.time()
            if self._review_heap:
                due_at, _, _ = self._review_heap[0]
                if due_at <= now:
                    return True
            for record in self.memories.values():
                if self._calculate_retention(record, now=now) < threshold_value:
                    return True
            return False

    def get_at_risk_memories(self, threshold: Optional[float] = None, limit: int = 1000) -> List[str]:
        threshold_value = self.review_threshold if threshold is None else float(threshold)
        with self._lock:
            self._cleanup_heap()
            now = time.time()
            due_ids: List[str] = []

            popped: List[Tuple[float, int, str]] = []
            while self._review_heap and len(due_ids) < limit:
                due_at, seq, mem_id = heapq.heappop(self._review_heap)
                popped.append((due_at, seq, mem_id))
                if due_at > now:
                    continue
                if mem_id in self.memories and self._scheduled_due.get(mem_id) == due_at:
                    due_ids.append(mem_id)
            for entry in popped:
                heapq.heappush(self._review_heap, entry)

            if len(due_ids) >= limit:
                return due_ids

            for memory_id, record in self.memories.items():
                if memory_id in due_ids:
                    continue
                if self._calculate_retention(record, now=now) < threshold_value:
                    due_ids.append(memory_id)
                    if len(due_ids) >= limit:
                        break
            return due_ids

    def get_due_review_count(self, limit: int = 1000) -> int:
        return len(self.get_at_risk_memories(limit=limit))

    def memory_stats(self) -> Dict[str, float]:
        with self._lock:
            if not self.memories:
                return {
                    "count": 0.0,
                    "stable": 0.0,
                    "unstable": 0.0,
                    "forgotten": 0.0,
                    "avg_retention": 0.0,
                }

            now = time.time()
            retentions = [self._calculate_retention(r, now=now) for r in self.memories.values()]
            stable = sum(1 for r in retentions if r >= self.stability_threshold)
            forgotten = sum(1 for r in self.memories.values() if r.forgotten)
            return {
                "count": float(len(self.memories)),
                "stable": float(stable),
                "unstable": float(len(self.memories) - stable),
                "forgotten": float(forgotten),
                "avg_retention": float(np.mean(retentions)),
            }

    def get_average_stability(self) -> float:
        with self._lock:
            if not self.memories:
                return 1.0
            return float(np.mean([record.stability for record in self.memories.values()]))

    def consolidate_due_memories(self, limit: int = 250, boost: float = 0.25) -> int:
        with self._lock:
            due_ids = self.get_at_risk_memories(limit=limit)
            count = 0
            for memory_id in due_ids:
                record = self.memories.get(memory_id)
                if record is None:
                    continue
                record.stability = float(max(self.min_stability, record.stability * (1.0 + float(boost))))
                record.strength = float(min(1.0, record.strength + 0.02))
                record.forgotten = False
                self._schedule_review(memory_id)
                count += 1
            return count

    def list_memories(
        self,
        offset: int = 0,
        limit: int = 50,
        stable: Optional[bool] = None,
        last_reviewed_before: Optional[float] = None,
        last_reviewed_after: Optional[float] = None,
        min_strength: Optional[float] = None,
        max_strength: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            now = time.time()
            rows: List[Dict[str, Any]] = []
            for record in self.memories.values():
                retention = self._calculate_retention(record, now=now)
                is_stable = retention >= self.stability_threshold
                if stable is not None and stable != is_stable:
                    continue
                if last_reviewed_before is not None and record.last_reviewed >= float(last_reviewed_before):
                    continue
                if last_reviewed_after is not None and record.last_reviewed <= float(last_reviewed_after):
                    continue
                if min_strength is not None and record.strength < float(min_strength):
                    continue
                if max_strength is not None and record.strength > float(max_strength):
                    continue
                rows.append(
                    {
                        "key": record.key,
                        "last_reviewed": record.last_reviewed,
                        "strength": record.strength,
                        "stability": record.stability,
                        "decay_rate": record.decay_rate,
                        "retrieval_difficulty": record.retrieval_difficulty,
                        "repetitions": record.repetitions,
                        "lapses": record.lapses,
                        "forgotten": record.forgotten,
                        "retention": retention,
                        "next_review_at": record.next_review_at,
                    }
                )

            rows.sort(key=lambda item: item["next_review_at"])
            start = max(0, int(offset))
            end = start + max(1, int(limit))
            return rows[start:end]

    def reset(self) -> None:
        with self._lock:
            self.memories.clear()
            self._payload_cache.clear()
            self._review_heap.clear()
            self._scheduled_due.clear()
            self._heap_seq = 0

    def save_state(self) -> None:
        with self._lock:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.store_dir.mkdir(parents=True, exist_ok=True)

            index_dump: Dict[str, Dict[str, Any]] = {}
            for memory_id, record in self.memories.items():
                data_path = Path(record.data_path)
                payload = self._payload_cache.get(memory_id)
                if payload is not None:
                    try:
                        torch.save(self._cpuify(payload), data_path)
                    except Exception as exc:
                        logger.warning("Failed to save memory payload for %s: %s", memory_id, exc)
                index_dump[memory_id] = self._serialize_record(record)

            with open(self.index_path, "w", encoding="utf-8") as handle:
                json.dump(index_dump, handle, indent=2)

    def load_state(self) -> None:
        with self._lock:
            self.reset()

            if self.index_path.exists():
                with open(self.index_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)

                for memory_id, rec_payload in payload.items():
                    record = self._deserialize_record(memory_id, rec_payload)
                    self.memories[memory_id] = record
                    self._schedule_review(memory_id)
                    if self.eager_load:
                        self._load_payload(memory_id)
                logger.info("Memory state loaded from %s with %d items", self.index_path, len(self.memories))
                return

            if self.legacy_metadata_path.exists():
                with open(self.legacy_metadata_path, "r", encoding="utf-8") as handle:
                    legacy = json.load(handle)
                for memory_id, rec_payload in legacy.items():
                    record = self._deserialize_record(memory_id, rec_payload)
                    self.memories[memory_id] = record
                    self._schedule_review(memory_id)
                logger.info("Legacy memory metadata loaded: %d items", len(self.memories))
