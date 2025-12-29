from typing import List, Dict, Any, Optional
import heapq
import time
from dataclasses import dataclass, field

@dataclass(order=True)
class Task:
    priority: float
    task_id: str
    action_type: str
    payload: Any = field(compare=False)
    created_at: float = field(default_factory=time.time, compare=False)

class CognitiveScheduler:
    """
    Priority queue scheduler that manages task execution based on
    memory stability and system needs.
    """
    def __init__(self):
        self.task_queue: List[Task] = []
        
    def add_task(self, task_id: str, action_type: str, payload: Any, priority: float):
        """
        Add a task to the queue. Lower priority value = higher urgency.
        """
        task = Task(priority, task_id, action_type, payload)
        heapq.heappush(self.task_queue, task)
        
    def get_next_task(self) -> Optional[Task]:
        """Pop the highest priority task."""
        if not self.task_queue:
            return None
        return heapq.heappop(self.task_queue)
        
    def prioritize_memory_refresh(self, memory_ids: List[str], current_stabilities: List[float]):
        """
        Dynamically schedule refresh tasks for unstable memories.
        """
        for mem_id, stability in zip(memory_ids, current_stabilities):
            # Lower stability -> Higher priority (lower number)
            # Priority range: 0 (critical) to 10 (background)
            priority = max(0.0, min(10.0, stability * 10.0))
            
            if stability < 0.3: # Critical threshold
                self.add_task(f"refresh_{mem_id}", "memory_refresh", mem_id, priority)
    
    def queue_size(self) -> int:
        return len(self.task_queue)
