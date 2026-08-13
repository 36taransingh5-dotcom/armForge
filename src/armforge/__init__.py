"""ArmForge -- an Arm-aware inference configuration engine.

ArmForge reads what an Arm CPU can actually do -- its instruction set
extensions, its vector lengths, its core topology -- predicts which inference
configuration should win on that specific silicon, and then proves or refutes
the prediction by measurement.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
