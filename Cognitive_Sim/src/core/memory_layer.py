import json
import math
import heapq
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.config import MemoryConfig
from src.utils.logger import logger


class MemoryLayer:
    """A 'Smart Database' implementing an Ebbinghaus-style forgetting curve.

    Persistence model
    - Writes a JSON *index* with metadata (timestamps, stability, blob path)
    - Writes per-memory payload blobs with `torch.save` (safe for tensors)
    - Loads index on startup; payloads are lazy-loaded on demand
    """

    def __init__(self, config: MemoryConfig):
        self.initial_retention = config.initial_retention
        self.decay_rate = config.decay_rate
        self.stability_threshold = config.stability_threshold

        # Policy knobs (kept in memory layer so scheduling can be efficient)
        self.review_threshold = float(getattr(config, "review_threshold", 0.4))
        self.recall_failure_retention = float(getattr(config, "recall_failure_retention", 0.15))

        # Persistence settings
        self.store_dir = Path(getattr(config, "store_dir", "data/memory_store"))
        self.index_path = Path(getattr(config, "index_path", "data/memory_index.json"))
        self.eager_load = bool(getattr(config, "eager_load", False))

        # Legacy path from earlier template (metadata-only JSON)
        self.legacy_metadata_path = Path("data/memory_store.json")

        # Memory store:
        # {memory_id: {'data': Any|None, 'data_path': str, 'created_at': float,
        #             'last_access': float, 'stability': float, 'access_count': int}}
        self.memories: Dict[str, Dict[str, Any]] = {}

        # Priority queue for "due" reviews.
        # heap entries: (next_review_at_epoch_seconds, seq, memory_id)
        self._review_heap: List[Tuple[float, int, str]] = []
        self._scheduled_due: Dict[str, float] = {}
        self._heap_seq = 0

    def _memory_blob_path(self, memory_id: str) -> Path:
        return self.store_dir / f"{memory_id}.pt"

    @staticmethod
    def _cpuify(obj: Any) -> Any:
        """Move tensors (and nested tensors) to CPU for safe serialization."""

        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: MemoryLayer._cpuify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            converted = [MemoryLayer._cpuify(v) for v in obj]
            return converted if isinstance(obj, list) else tuple(converted)
        return obj

    def _calculate_retention(self, memory: Dict[str, Any]) -> float:
        """R = exp(-(decay_rate * t)/stability)."""

        elapsed = time.time() - float(memory["last_access"])
        stability = max(1e-6, float(memory["stability"]))
        return float(np.exp(-(float(self.decay_rate) * elapsed) / stability))

    def _next_review_at(self, memory: Dict[str, Any], threshold: float) -> float:
        """Compute the time when retention will cross below `threshold`."""

        last_access = float(memory.get("last_access", time.time()))
        stability = max(1e-6, float(memory.get("stability", 1.0)))
        if float(self.decay_rate) <= 0:
            return float("inf")

        threshold = float(np.clip(threshold, 1e-6, 0.999999))
        due_in = (-math.log(threshold) * stability) / float(self.decay_rate)
        return last_access + max(0.0, due_in)

    def _schedule_review(self, memory_id: str, threshold: Optional[float] = None) -> None:
        if memory_id not in self.memories:
            return

        threshold = self.review_threshold if threshold is None else float(threshold)
        due_at = self._next_review_at(self.memories[memory_id], threshold)
        self._scheduled_due[memory_id] = due_at

        self._heap_seq += 1
        heapq.heappush(self._review_heap, (due_at, self._heap_seq, memory_id))

    def _cleanup_heap(self) -> None:
        """Remove stale heap entries after rescheduling."""

        while self._review_heap:
            due_at, _, mem_id = self._review_heap[0]
            if mem_id in self.memories and self._scheduled_due.get(mem_id) == due_at:
                return
            heapq.heappop(self._review_heap)

    def add_memory(self, memory_id: str, data: Any, initial_stability: float = 1.0):
        """Store a new memory trace."""

        current_time = time.time()
        self.memories[memory_id] = {
            "data": data,
            "data_path": str(self._memory_blob_path(memory_id)),
            "created_at": current_time,
            "last_access": current_time,
            "stability": float(initial_stability),
            "access_count": 1,
        }
        self._schedule_review(memory_id)
        logger.debug(f"Memory added: {memory_id}")

    def retrieve_memory(self, memory_id: str) -> Optional[Any]:
        """Retrieve memory if it hasn't decayed below recall-failure threshold.

        On successful retrieval, updates stability (spaced repetition reinforcement).
        """

        if memory_id not in self.memories:
            return None

        memory = self.memories[memory_id]

        # Lazy-load payload if necessary
        if memory.get("data") is None:
            data_path = Path(memory.get("data_path", self._memory_blob_path(memory_id)))
            if data_path.exists():
                try:
                    memory["data"] = torch.load(data_path, map_location="cpu")
                except Exception as e:
                    logger.warning(f"Failed to load memory payload for {memory_id} from {data_path}: {e}")
                    memory["data"] = None

        retention = self._calculate_retention(memory)

        # BIOLOGICAL CONSTRAINT: recall failure
        if retention < self.recall_failure_retention:
            logger.info(f"Recall Failed for {memory_id} (R={retention:.2f})")
            return None

        # Reinforce on successful access
        self._reinforce_memory(memory_id)
        return memory.get("data")

    def _reinforce_memory(self, memory_id: str) -> None:
        """Strengthen memory stability upon access (Spaced Repetition principle)."""

        memory = self.memories[memory_id]
        current_time = time.time()
        elapsed = current_time - float(memory["last_access"])

        # Stability increase depends on difficulty of retrieval (time passed)
        stability_gain = 0.1 * elapsed

        memory["stability"] = float(memory["stability"]) + float(stability_gain)
        memory["last_access"] = current_time
        memory["access_count"] = int(memory["access_count"]) + 1
        self._schedule_review(memory_id)

    def has_at_risk_memory(self, threshold: Optional[float] = None) -> bool:
        """Fast check: is any memory currently below `threshold`?"""

        threshold = self.review_threshold if threshold is None else float(threshold)

        # If caller uses a different threshold than configured, fallback to early-exit scan.
        if threshold != self.review_threshold:
            for memory in self.memories.values():
                if self._calculate_retention(memory) < threshold:
                    return True
            return False

        self._cleanup_heap()
        if not self._review_heap:
            return False
        due_at, _, _ = self._review_heap[0]
        return due_at <= time.time()

    def get_at_risk_memories(self, threshold: Optional[float] = None, limit: int = 1000) -> List[str]:
        """Identify memories that are fading.

        - If `threshold` matches configured review_threshold, uses the review heap.
        - Otherwise falls back to scanning.
        """

        threshold = self.review_threshold if threshold is None else float(threshold)

        if threshold != self.review_threshold:
            at_risk: List[str] = []
            for mem_id, memory in self.memories.items():
                if self._calculate_retention(memory) < threshold:
                    at_risk.append(mem_id)
                    if len(at_risk) >= limit:
                        break
            return at_risk

        self._cleanup_heap()
        now = time.time()

        popped: List[Tuple[float, int, str]] = []
        due_ids: List[str] = []

        while self._review_heap and len(due_ids) < limit:
            due_at, seq, mem_id = self._review_heap[0]
            if due_at > now:
                break
            heapq.heappop(self._review_heap)
            popped.append((due_at, seq, mem_id))
            if mem_id in self.memories and self._scheduled_due.get(mem_id) == due_at:
                due_ids.append(mem_id)

        for entry in popped:
            heapq.heappush(self._review_heap, entry)

        return due_ids

    def get_due_review_count(self, limit: int = 1000) -> int:
        return len(self.get_at_risk_memories(limit=limit))

    def save_state(self) -> None:
        """Persist memory index (JSON) and per-memory payload blobs."""

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        index_dump: Dict[str, Dict[str, Any]] = {}
        for mem_id, mem in self.memories.items():
            data_path = Path(mem.get("data_path", self._memory_blob_path(mem_id)))

            payload = mem.get("data")
            if payload is not None:
                try:
                    torch.save(self._cpuify(payload), data_path)
                except Exception as e:
                    logger.warning(f"Failed to persist memory payload for {mem_id} to {data_path}: {e}")

            index_dump[mem_id] = {
                "created_at": float(mem.get("created_at", time.time())),
                "last_access": float(mem.get("last_access", time.time())),
                "stability": float(mem.get("stability", 1.0)),
                "access_count": int(mem.get("access_count", 0)),
                "data_path": str(data_path),
            }

        with open(self.index_path, "w") as f:
            json.dump(index_dump, f, indent=2)

        logger.info(f"Memory State saved to {self.index_path} (blobs in {self.store_dir})")

    def load_state(self) -> None:
        """Load memory metadata.

        Preferred: `index_path`. Legacy: `legacy_metadata_path` (metadata-only).
        """

        self.memories = {}
        self._review_heap.clear()
        self._scheduled_due.clear()
        self._heap_seq = 0

        if self.index_path.exists():
            with open(self.index_path, "r") as f:
                index = json.load(f)

            for mem_id, meta in index.items():
                rec: Dict[str, Any] = {
                    "created_at": float(meta.get("created_at", time.time())),
                    "last_access": float(meta.get("last_access", time.time())),
                    "stability": float(meta.get("stability", 1.0)),
                    "access_count": int(meta.get("access_count", 0)),
                    "data_path": str(meta.get("data_path", self._memory_blob_path(mem_id))),
                    "data": None,
                }

                if self.eager_load:
                    data_path = Path(rec["data_path"])
                    rec["data"] = torch.load(data_path, map_location="cpu") if data_path.exists() else None

                self.memories[mem_id] = rec
                self._schedule_review(mem_id)

            logger.info(f"Memory State loaded: {len(self.memories)} items from {self.index_path}.")
            return

        # Legacy metadata-only format
        if self.legacy_metadata_path.exists():
            with open(self.legacy_metadata_path, "r") as f:
                legacy = json.load(f)

            for mem_id, meta in legacy.items():
                self.memories[mem_id] = {
                    "created_at": float(meta.get("created_at", time.time())),
                    "last_access": float(meta.get("last_access", time.time())),
                    "stability": float(meta.get("stability", 1.0)),
                    "access_count": int(meta.get("access_count", 0)),
                    "data_path": str(self._memory_blob_path(mem_id)),
                    "data": None,
                }
                self._schedule_review(mem_id)

            logger.info(
                f"Legacy memory metadata loaded: {len(self.memories)} items from {self.legacy_metadata_path}. "
                f"(payloads missing; next save will write {self.index_path})"
            )

import json
import math
import heapq
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.config import MemoryConfig
from src.utils.logger import logger


class MemoryLayer:
    """A 'Smart Database' implementing an Ebbinghaus-style forgetting curve.

    Persistence model
    - Writes a JSON *index* with metadata (timestamps, stability, blob path)
    - Writes per-memory payload blobs with `torch.save` (safe for tensors)
    - Loads index on startup; payloads are lazy-loaded on demand
    """

    def __init__(self, config: MemoryConfig):
        self.initial_retention = config.initial_retention
        self.decay_rate = config.decay_rate
        self.stability_threshold = config.stability_threshold

        # Policy knobs (kept in memory layer so scheduling can be efficient)
        self.review_threshold = float(getattr(config, "review_threshold", 0.4))
        self.recall_failure_retention = float(getattr(config, "recall_failure_retention", 0.15))

        # Persistence settings
        self.store_dir = Path(getattr(config, "store_dir", "data/memory_store"))
        self.index_path = Path(getattr(config, "index_path", "data/memory_index.json"))
        self.eager_load = bool(getattr(config, "eager_load", False))

        # Legacy path from earlier template (metadata-only JSON)
        self.legacy_metadata_path = Path("data/memory_store.json")

        # Memory store:
        # {memory_id: {'data': Any|None, 'data_path': str, 'created_at': float,
        #             'last_access': float, 'stability': float, 'access_count': int}}
        self.memories: Dict[str, Dict[str, Any]] = {}

        # Priority queue for "due" reviews.
        # heap entries: (next_review_at_epoch_seconds, seq, memory_id)
        self._review_heap: List[Tuple[float, int, str]] = []
        self._scheduled_due: Dict[str, float] = {}
        self._heap_seq = 0

    def _memory_blob_path(self, memory_id: str) -> Path:
        return self.store_dir / f"{memory_id}.pt"

    @staticmethod
    def _cpuify(obj: Any) -> Any:
        """Move tensors (and nested tensors) to CPU for safe serialization."""

        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: MemoryLayer._cpuify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            converted = [MemoryLayer._cpuify(v) for v in obj]
            return converted if isinstance(obj, list) else tuple(converted)
        return obj

    def _calculate_retention(self, memory: Dict[str, Any]) -> float:
        """R = exp(-(decay_rate * t)/stability)."""

        elapsed = time.time() - float(memory["last_access"])
        stability = max(1e-6, float(memory["stability"]))
        return float(np.exp(-(float(self.decay_rate) * elapsed) / stability))

    def _next_review_at(self, memory: Dict[str, Any], threshold: float) -> float:
        """Compute the time when retention will cross below `threshold`."""

        last_access = float(memory.get("last_access", time.time()))
        stability = max(1e-6, float(memory.get("stability", 1.0)))
        if float(self.decay_rate) <= 0:
            return float("inf")

        threshold = float(np.clip(threshold, 1e-6, 0.999999))
        due_in = (-math.log(threshold) * stability) / float(self.decay_rate)
        return last_access + max(0.0, due_in)

    def _schedule_review(self, memory_id: str, threshold: Optional[float] = None) -> None:
        if memory_id not in self.memories:
            return

        threshold = self.review_threshold if threshold is None else float(threshold)
        due_at = self._next_review_at(self.memories[memory_id], threshold)
        self._scheduled_due[memory_id] = due_at

        self._heap_seq += 1
        heapq.heappush(self._review_heap, (due_at, self._heap_seq, memory_id))

    def _cleanup_heap(self) -> None:
        """Remove stale heap entries after rescheduling."""

        while self._review_heap:
            due_at, _, mem_id = self._review_heap[0]
            if mem_id in self.memories and self._scheduled_due.get(mem_id) == due_at:
                return
            heapq.heappop(self._review_heap)

    def add_memory(self, memory_id: str, data: Any, initial_stability: float = 1.0):
        """Store a new memory trace."""

        current_time = time.time()
        self.memories[memory_id] = {
            "data": data,
            "data_path": str(self._memory_blob_path(memory_id)),
            "created_at": current_time,
            "last_access": current_time,
            "stability": float(initial_stability),
            "access_count": 1,
        }
        self._schedule_review(memory_id)
        logger.debug(f"Memory added: {memory_id}")

    def retrieve_memory(self, memory_id: str) -> Optional[Any]:
        """Retrieve memory if it hasn't decayed below recall-failure threshold.

        On successful retrieval, updates stability (spaced repetition reinforcement).
        """

        if memory_id not in self.memories:
            return None

        memory = self.memories[memory_id]

        # Lazy-load payload if necessary
        if memory.get("data") is None:
            data_path = Path(memory.get("data_path", self._memory_blob_path(memory_id)))
            if data_path.exists():
                try:
                    memory["data"] = torch.load(data_path, map_location="cpu")
                except Exception as e:
                    logger.warning(f"Failed to load memory payload for {memory_id} from {data_path}: {e}")
                    memory["data"] = None

        retention = self._calculate_retention(memory)

        # BIOLOGICAL CONSTRAINT: recall failure
        if retention < self.recall_failure_retention:
            logger.info(f"Recall Failed for {memory_id} (R={retention:.2f})")
            return None

        # Reinforce on successful access
        self._reinforce_memory(memory_id)
        return memory.get("data")

    def _reinforce_memory(self, memory_id: str) -> None:
        """Strengthen memory stability upon access (Spaced Repetition principle)."""

        memory = self.memories[memory_id]
        current_time = time.time()
        elapsed = current_time - float(memory["last_access"])

        # Stability increase depends on difficulty of retrieval (time passed)
        stability_gain = 0.1 * elapsed

        memory["stability"] = float(memory["stability"]) + float(stability_gain)
        memory["last_access"] = current_time
        memory["access_count"] = int(memory["access_count"]) + 1
        self._schedule_review(memory_id)

    def has_at_risk_memory(self, threshold: Optional[float] = None) -> bool:
        """Fast check: is any memory currently below `threshold`?"""

        threshold = self.review_threshold if threshold is None else float(threshold)

        # If caller uses a different threshold than configured, fallback to early-exit scan.
        if threshold != self.review_threshold:
            for memory in self.memories.values():
                if self._calculate_retention(memory) < threshold:
                    return True
            return False

        self._cleanup_heap()
        if not self._review_heap:
            return False
        due_at, _, _ = self._review_heap[0]
        return due_at <= time.time()

    def get_at_risk_memories(self, threshold: Optional[float] = None, limit: int = 1000) -> List[str]:
        """Identify memories that are fading.

        - If `threshold` matches configured review_threshold, uses the review heap.
        - Otherwise falls back to scanning.
        """

        threshold = self.review_threshold if threshold is None else float(threshold)

        if threshold != self.review_threshold:
            at_risk: List[str] = []
            for mem_id, memory in self.memories.items():
                if self._calculate_retention(memory) < threshold:
                    at_risk.append(mem_id)
                    if len(at_risk) >= limit:
                        break
            return at_risk

        self._cleanup_heap()
        now = time.time()

        popped: List[Tuple[float, int, str]] = []
        due_ids: List[str] = []

        while self._review_heap and len(due_ids) < limit:
            due_at, seq, mem_id = self._review_heap[0]
            if due_at > now:
                break
            heapq.heappop(self._review_heap)
            popped.append((due_at, seq, mem_id))
            if mem_id in self.memories and self._scheduled_due.get(mem_id) == due_at:
                due_ids.append(mem_id)

        for entry in popped:
            heapq.heappush(self._review_heap, entry)

        return due_ids

    def get_due_review_count(self, limit: int = 1000) -> int:
        return len(self.get_at_risk_memories(limit=limit))

    def save_state(self) -> None:
        """Persist memory index (JSON) and per-memory payload blobs."""

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        index_dump: Dict[str, Dict[str, Any]] = {}
        for mem_id, mem in self.memories.items():
            data_path = Path(mem.get("data_path", self._memory_blob_path(mem_id)))

            payload = mem.get("data")
            if payload is not None:
                try:
                    torch.save(self._cpuify(payload), data_path)
                except Exception as e:
                    logger.warning(f"Failed to persist memory payload for {mem_id} to {data_path}: {e}")

            index_dump[mem_id] = {
                "created_at": float(mem.get("created_at", time.time())),
                "last_access": float(mem.get("last_access", time.time())),
                "stability": float(mem.get("stability", 1.0)),
                "access_count": int(mem.get("access_count", 0)),
                "data_path": str(data_path),
            }

        with open(self.index_path, "w") as f:
            json.dump(index_dump, f, indent=2)

        logger.info(f"Memory State saved to {self.index_path} (blobs in {self.store_dir})")

    def load_state(self) -> None:
        """Load memory metadata.

        Preferred: `index_path`. Legacy: `legacy_metadata_path` (metadata-only).
        """

        self.memories = {}
        self._review_heap.clear()
        self._scheduled_due.clear()
        self._heap_seq = 0

        if self.index_path.exists():
            with open(self.index_path, "r") as f:
                index = json.load(f)

            for mem_id, meta in index.items():
                rec: Dict[str, Any] = {
                    "created_at": float(meta.get("created_at", time.time())),
                    "last_access": float(meta.get("last_access", time.time())),
                    "stability": float(meta.get("stability", 1.0)),
                    "access_count": int(meta.get("access_count", 0)),
                    "data_path": str(meta.get("data_path", self._memory_blob_path(mem_id))),
                    "data": None,
                }

                if self.eager_load:
                    data_path = Path(rec["data_path"])
                    rec["data"] = torch.load(data_path, map_location="cpu") if data_path.exists() else None

                self.memories[mem_id] = rec
                self._schedule_review(mem_id)

            logger.info(f"Memory State loaded: {len(self.memories)} items from {self.index_path}.")
            return

        # Legacy metadata-only format
        if self.legacy_metadata_path.exists():
            with open(self.legacy_metadata_path, "r") as f:
                legacy = json.load(f)

            for mem_id, meta in legacy.items():
                self.memories[mem_id] = {
                    "created_at": float(meta.get("created_at", time.time())),
                    "last_access": float(meta.get("last_access", time.time())),
                    "stability": float(meta.get("stability", 1.0)),
                    "access_count": int(meta.get("access_count", 0)),
                    "data_path": str(self._memory_blob_path(mem_id)),
                    "data": None,
                }
                self._schedule_review(mem_id)

            logger.info(
                f"Legacy memory metadata loaded: {len(self.memories)} items from {self.legacy_metadata_path}. "
                f"(payloads missing; next save will write {self.index_path})"
            )
