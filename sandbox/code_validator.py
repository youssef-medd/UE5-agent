from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_BANNED_CALLS = {
    "eval", "exec", "compile", "__import__",
    "open", "os.system", "subprocess.run", "subprocess.Popen",
    "shutil.rmtree", "os.remove", "os.unlink",
}

_BANNED_IMPORTS = {"subprocess", "socket", "ctypes", "pickle"}


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class CodeValidator:
    """AST-based static analysis for generated UE5 Python scripts."""

    def validate(self, code: str) -> ValidationResult:
        result = ValidationResult(valid=True)

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            result.valid = False
            result.errors.append(f"Syntax error: {exc}")
            return result

        for node in ast.walk(tree):
            self._check_node(node, result)

        if result.errors:
            result.valid = False

        return result

    def _check_node(self, node: ast.AST, result: ValidationResult) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _BANNED_IMPORTS:
                    result.errors.append(f"Banned import: {alias.name}")

        elif isinstance(node, ast.ImportFrom):
            if node.module in _BANNED_IMPORTS:
                result.errors.append(f"Banned import: {node.module}")

        elif isinstance(node, ast.Call):
            call_name = self._get_call_name(node)
            if call_name in _BANNED_CALLS:
                result.errors.append(f"Banned call: {call_name}")

        elif isinstance(node, ast.While):
            if isinstance(node.test, ast.Constant) and node.test.value is True:
                result.warnings.append("Infinite while loop detected — ensure a break condition exists")

    def _get_call_name(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            parts = []
            n = node.func
            while isinstance(n, ast.Attribute):
                parts.append(n.attr)
                n = n.value
            if isinstance(n, ast.Name):
                parts.append(n.id)
            return ".".join(reversed(parts))
        return ""
