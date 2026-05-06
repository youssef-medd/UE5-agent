from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sandbox.code_validator import CodeValidator

logger = logging.getLogger(__name__)


@dataclass
class DryRunReport:
    code: str
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    simulated_output: str = ""


class DryRunner:
    """Simulates code execution without sending anything to UE5."""

    def __init__(self) -> None:
        self._validator = CodeValidator()

    def run(self, code: str) -> DryRunReport:
        validation = self._validator.validate(code)
        lines = code.strip().splitlines()

        simulated = (
            f"[DRY RUN] Would execute {len(lines)} lines of UE5 Python.\n"
            f"Validation: {'PASS' if validation.valid else 'FAIL'}\n"
        )
        if validation.warnings:
            simulated += "Warnings:\n" + "\n".join(f"  - {w}" for w in validation.warnings)

        logger.info(
            "DryRunner: %d lines, valid=%s, warnings=%d",
            len(lines), validation.valid, len(validation.warnings),
        )

        return DryRunReport(
            code=code,
            valid=validation.valid,
            errors=validation.errors,
            warnings=validation.warnings,
            simulated_output=simulated,
        )
