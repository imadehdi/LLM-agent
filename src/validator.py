from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from typing import Optional


class ValidationResult(BaseModel):
    passed: bool
    checks: dict[str, bool]
    errors: list[str]
    warnings: list[str]


def validate_solution(
    weights: list[float],
    max_weight: float = 0.15,
    target_return: Optional[float] = None,
    mu: Optional[list[float]] = None,
    Sigma: Optional[list[list[float]]] = None,
    tol: float = 1e-4,
) -> ValidationResult:
    w = np.array(weights)
    checks = {}
    errors = []
    warnings = []

    # 1. Sum to one
    checks["sum_to_one"] = bool(abs(w.sum() - 1.0) < tol)
    if not checks["sum_to_one"]:
        errors.append(f"Weights sum to {w.sum():.6f}, expected 1.0")

    # 2. Long only
    checks["long_only"] = bool(np.all(w >= -tol))
    if not checks["long_only"]:
        errors.append(f"Negative weights found: min={w.min():.6f}")

    # 3. Max weight
    checks["max_weight"] = bool(np.all(w <= max_weight + tol))
    if not checks["max_weight"]:
        errors.append(f"Weight exceeds max {max_weight}: max={w.max():.6f}")

    # 4. Target return (optional)
    if target_return is not None and mu is not None:
        mu_arr = np.array(mu)
        portfolio_return = float(mu_arr @ w)
        checks["target_return"] = bool(portfolio_return >= target_return - tol)
        if not checks["target_return"]:
            errors.append(
                f"Return {portfolio_return:.4f} below target {target_return:.4f}"
            )
    
    # 5. Variance positive (optional)
    if Sigma is not None:
        S = np.array(Sigma)
        variance = float(w @ S @ w)
        checks["variance_positive"] = bool(variance >= -tol)
        if variance < 0:
            warnings.append(f"Negative variance detected: {variance:.6f}")

    passed = len(errors) == 0
    return ValidationResult(passed=passed, checks=checks, errors=errors, warnings=warnings)