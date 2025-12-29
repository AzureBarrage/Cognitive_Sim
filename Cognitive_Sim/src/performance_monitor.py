import psutil
import time
from typing import Dict

class PerformanceMonitor:
    """
    Monitors system resource usage (CPU, Memory).
    """
    def __init__(self):
        self.process = psutil.Process()

    def get_resource_usage(self) -> Dict[str, float]:
        """
        Returns current CPU and Memory usage of the process.
        """
        try:
            with self.process.oneshot():
                cpu_percent = self.process.cpu_percent(interval=None)
                memory_info = self.process.memory_info()
                memory_percent = self.process.memory_percent()
                
                return {
                    "cpu_percent": cpu_percent,
                    "memory_rss_mb": memory_info.rss / (1024 * 1024), # Convert to MB
                    "memory_percent": memory_percent
                }
        except Exception as e:
            return {
                "error": str(e),
                "cpu_percent": 0.0,
                "memory_rss_mb": 0.0,
                "memory_percent": 0.0
            }
