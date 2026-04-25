class CognitiveSimError(Exception):
    """Base exception for the Cognitive Simulation framework."""
    pass

class ConfigurationError(CognitiveSimError):
    """Raised when there is an issue with the configuration."""
    pass

class MemoryError(CognitiveSimError):
    """Raised when there is an issue with the Memory Layer."""
    pass

class NetworkError(CognitiveSimError):
    """Raised when there is an issue with the Cognitive Network."""
    pass

class OptimizationError(CognitiveSimError):
    """Raised when there is an issue during optimization."""
    pass
