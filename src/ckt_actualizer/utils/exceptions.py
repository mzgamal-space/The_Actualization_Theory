"""
exceptions.py — Custom Exceptions for CKT Actualizer Engine
=============================================================
Author : Antigravity (Advanced Agentic Coding)

Defines custom exception classes for robust error handling across the engine.
"""

class CKTActualizerError(Exception):
    """Base exception for all CKT Actualizer Engine errors."""
    pass

class CausalViolationError(CKTActualizerError):
    """Raised when a candidate thought violates fundamental causal laws (Pipeline A)."""
    pass

class InvalidPrimeProfileError(CKTActualizerError):
    """Raised when an invalid Prime dimension profile is provided."""
    pass

class AnalogyLibraryEmptyError(CKTActualizerError):
    """Raised when the FDSA analogy library is empty but a query is made."""
    pass
