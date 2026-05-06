from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_HISTORY_DIR = Path("logs/agent_runs")


@dataclass
class TaskRecord:
    task_id: str
    prompt: str
    steps: list[dict] = field(default_factory=list)
    final_code: str = ""
    success: bool = False
    error: str = ""
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0


class TaskHistory:
    """Persists task records to JSON files in logs/agent_runs/."""

    def save(self, record: TaskRecord) -> Path:
        _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        path = _HISTORY_DIR / f"{record.task_id}.json"
        with open(path, "w") as f:
            json.dump(asdict(record), f, indent=2)
        logger.debug("Saved task record: %s", path)
        return path

    def load(self, task_id: str) -> TaskRecord | None:
        path = _HISTORY_DIR / f"{task_id}.json"
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return TaskRecord(**data)
        except Exception as exc:
            logger.error("Could not load task %s: %s", task_id, exc)
            return None

    def list_recent(self, n: int = 10) -> list[TaskRecord]:
        if not _HISTORY_DIR.exists():
            return []
        files = sorted(_HISTORY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        records = []
        for f in files[:n]:
            try:
                with open(f) as fh:
                    records.append(TaskRecord(**json.load(fh)))
            except Exception:
                continue
        return records
