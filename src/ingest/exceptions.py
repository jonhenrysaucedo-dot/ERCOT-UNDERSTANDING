"""Typed exceptions for the ingest layer.

All ingest modules raise from this hierarchy so callers can distinguish
recoverable from fatal errors without catching generic Exception.
"""


class QuantumDartError(Exception):
    """Base exception for all quantum_dart errors."""


class MissingDataError(QuantumDartError):
    """Required data source is absent (NULL upstream — per CLAUDE.md §3)."""


class StaleDataError(QuantumDartError):
    """Data exists but is too old to be trusted for a given as_of_date."""


class ExternalDataError(QuantumDartError):
    """External API / HTTP failure."""


class WalkForwardViolation(QuantumDartError):
    """A feature or model used data that was not available at as_of_timestamp.

    Raised when the as_of_timestamp contract (CLAUDE.md §2) is violated.
    This is a programmer error, not a data error — fix the code.
    """


class ComplianceTagError(QuantumDartError):
    """A REAL/SYNTHETIC/NULL tag is missing or propagated incorrectly."""


class ERCOTParseError(QuantumDartError):
    """ERCOT MIS file could not be parsed."""
