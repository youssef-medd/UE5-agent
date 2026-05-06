from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ParsedTask:
    prompt: str
    intent: str = ""
    entities: dict[str, str] = field(default_factory=dict)
    flags: set[str] = field(default_factory=set)


_FLAG_PATTERNS = {
    "dry_run": re.compile(r"\b(dry.run|simulate|preview|no.?write)\b", re.I),
    "destructive": re.compile(r"\b(delete|destroy|remove|clear|wipe)\b", re.I),
    "blueprint": re.compile(r"\b(blueprint|bp)\b", re.I),
    "spawn": re.compile(r"\b(spawn|add|place|create|insert)\b", re.I),
    "move": re.compile(r"\b(move|teleport|relocate|set location)\b", re.I),
}

_INTENT_PATTERNS = [
    ("spawn_actor", re.compile(r"\b(spawn|add|place|create)\b.*(actor|mesh|light|object|blueprint)", re.I)),
    ("delete_actor", re.compile(r"\b(delete|destroy|remove)\b.*(actor|mesh|object)", re.I)),
    ("create_blueprint", re.compile(r"\b(create|make|build)\b.*(blueprint|bp)\b", re.I)),
    ("modify_property", re.compile(r"\b(set|change|update|modify)\b.*(property|value|color|intensity|scale)", re.I)),
    ("run_script", re.compile(r"\b(run|execute|apply|do)\b", re.I)),
]


def parse_task(prompt: str) -> ParsedTask:
    task = ParsedTask(prompt=prompt)

    for flag_name, pattern in _FLAG_PATTERNS.items():
        if pattern.search(prompt):
            task.flags.add(flag_name)

    for intent_name, pattern in _INTENT_PATTERNS:
        if pattern.search(prompt):
            task.intent = intent_name
            break

    if not task.intent:
        task.intent = "run_script"

    return task
