from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventKind(str, Enum):
    TASK_CREATED = "task.created"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"

    AGENT_THINKING = "agent.thinking"
    AGENT_DONE = "agent.done"
    AGENT_ERROR = "agent.error"

    CODE_GENERATED = "code.generated"
    CODE_REVIEWED = "code.reviewed"
    CODE_EXECUTED = "code.executed"

    UE5_REQUEST = "ue5.request"
    UE5_RESPONSE = "ue5.response"
    UE5_ERROR = "ue5.error"


@dataclass
class Event:
    kind: EventKind
    payload: dict[str, Any] = field(default_factory=dict)
    task_id: str = ""
    agent: str = ""
    timestamp: float = field(default_factory=time.time)
