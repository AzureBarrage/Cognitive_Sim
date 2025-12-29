from typing import Dict
from .performance_monitor import PerformanceMonitor

class SystemHealth:
    """
    Evaluates the overall health of the system based on resource usage and simulation metrics.
    """
    def __init__(self, memory_threshold_mb: float = 1024.0, cpu_threshold_percent: float = 80.0):
        self.monitor = PerformanceMonitor()
        self.memory_threshold_mb = memory_threshold_mb
        self.cpu_threshold_percent = cpu_threshold_percent

    def check_health(self) -> Dict[str, str]:
        """
        Checks system health and returns a status report.
        """
        usage = self.monitor.get_resource_usage()
        
        status = "HEALTHY"
        issues = []

        if usage["memory_rss_mb"] > self.memory_threshold_mb:
            status = "WARNING"
            issues.append(f"High Memory Usage: {usage['memory_rss_mb']:.2f} MB")

        if usage["cpu_percent"] > self.cpu_threshold_percent:
            status = "WARNING"
            issues.append(f"High CPU Usage: {usage['cpu_percent']:.1f}%")

        return {
            "status": status,
            "issues": issues,
            "metrics": usage
        }
