"""Shared failure type for optimization evidence modules."""


class OptimizationContractError(ValueError):
    """Raised when an optimization report could overstate measured evidence."""
